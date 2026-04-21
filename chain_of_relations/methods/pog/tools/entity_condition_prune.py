#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   entity_condition_prune.py
@Time    :   2026/03/06
@Author  :   liuchenhui
@Desc    :   None
"""


from dataclasses import dataclass, field
import ast
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from chain_of_relations.schema import Entity, Relation


@dataclass
class EntityConditionPruneInput:
	question: str
	current_question: str
	topic_entity_name: str
	condition: str
	candidate_groups: Dict[str, Dict[str, Any]]
	user_prompt: str
	entity_width: int = 3
	sample_entity_threshold: int = 500
	temperature: float = 0.3
	max_tokens: int = 1024
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class EntityConditionPruneOutput:
	success: bool
	pruned_entities: List[Entity]
	pruned_scores: List[float]
	text_context: str
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	warnings: List[str] = field(default_factory=list)


_SEMANTIC_MODEL = None


def _run_llm(
	llm_generate: Callable[..., Any],
	prompt: str,
	temperature: float,
	max_tokens: int,
	max_retries: int,
	system_prompt: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
	usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

	for _ in range(max_retries):
		try:
			if system_prompt is not None:
				try:
					result, usage = llm_generate(
						prompt,
						temperature=temperature,
						max_tokens=max_tokens,
						system_prompt=system_prompt,
					)
				except TypeError:
					result, usage = llm_generate(
						prompt,
						temperature=temperature,
						max_tokens=max_tokens,
					)
			else:
				result, usage = llm_generate(
					prompt,
					temperature=temperature,
					max_tokens=max_tokens,
				)
			if result:
				return result, usage
		except Exception:
			continue

	return None, usage


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
	payload = (text or "").strip()
	if not payload:
		return None

	candidates = [payload]
	candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", payload, flags=re.IGNORECASE))
	obj_match = re.search(r"\{[\s\S]*\}", payload)
	if obj_match:
		candidates.append(obj_match.group(0))

	for candidate in candidates:
		candidate = candidate.strip()
		if not candidate:
			continue
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, dict):
				return parsed
		except Exception:
			pass
		try:
			parsed = ast.literal_eval(candidate)
			if isinstance(parsed, dict):
				return parsed
		except Exception:
			pass

	return None


def _parse_entity_name_list(text: str) -> List[str]:
	payload = (text or "").strip()
	if not payload:
		return []

	candidates: List[str] = [payload]
	candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", payload, flags=re.IGNORECASE))
	arr_match = re.search(r"\[[\s\S]*\]", payload)
	if arr_match:
		candidates.append(arr_match.group(0))

	for candidate in candidates:
		candidate = candidate.strip()
		if not candidate:
			continue
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, list):
				return [str(item).strip() for item in parsed if str(item).strip()]
		except Exception:
			pass
		try:
			parsed = ast.literal_eval(candidate)
			if isinstance(parsed, list):
				return [str(item).strip() for item in parsed if str(item).strip()]
		except Exception:
			pass

	return []


def _normalize_group_rel_name(group: Dict[str, Any]) -> str:
	relation = group.get("relation")
	if isinstance(relation, Relation):
		return relation.id
	if isinstance(relation, str):
		return relation
	return ""


def _normalize_group_head(group: Dict[str, Any]) -> bool:
	head = group.get("head")
	if isinstance(head, bool):
		return head
	if isinstance(head, str):
		return head.strip().lower() in {"true", "1", "yes", "head"}
	return bool(head)


def _normalize_group_entities(group: Dict[str, Any]) -> List[Entity]:
	raw_entities = group.get("entities", [])
	entities: List[Entity] = []
	for item in raw_entities:
		if isinstance(item, Entity):
			entities.append(item)
		elif isinstance(item, dict):
			ent_id = item.get("id") or item.get("mid")
			ent_name = item.get("name", "")
			if ent_id:
				entities.append(Entity(id=str(ent_id), name=str(ent_name)))
	return entities


def _normalize_group_topic_name(group: Dict[str, Any], fallback: str) -> str:
	topic_entity = group.get("topic_entity")
	if isinstance(topic_entity, Entity):
		name = str(topic_entity.name or "").strip()
		if name:
			return name
	if isinstance(topic_entity, dict):
		name = str(topic_entity.get("name", "")).strip()
		if name:
			return name
	name = str(group.get("topic_entity_name", "")).strip()
	if name:
		return name
	return str(fallback or "").strip()


def _build_candidates_text(topic_entity_name: str, candidate_groups: Dict[str, Dict[str, Any]]) -> str:
	lines: List[str] = []
	for group_id, group in candidate_groups.items():
		relation_name = _normalize_group_rel_name(group)
		head = _normalize_group_head(group)
		direction = "tail entities" if head else "head entities"
		entities = _normalize_group_entities(group)

		lines.append(f"Group {group_id}:")
		if relation_name:
			lines.append(f"  Relation: {relation_name}")
		lines.append(f"  Direction: {topic_entity_name} -> {direction}")
		if not entities:
			lines.append("  Candidates: []")
			continue

		candidate_text = ", ".join([ent.name or ent.id for ent in entities])
		lines.append(f"  Candidates: [{candidate_text}]")

	return "\n".join(lines).strip()


def _is_mid_like(entity_id: str) -> bool:
	text = str(entity_id or "").strip().lower()
	return text.startswith("m.") or text.startswith("g.")


def _is_all_digit_entity_ids(entities: List[Entity]) -> bool:
	if not entities:
		return False
	for ent in entities:
		if not str(ent.id or "").isdigit():
			return False
	return True


def _needs_no_prune(relation_name: str, entities: List[Entity]) -> bool:
	rela = str(relation_name or "").strip().lower()
	if len(entities) <= 1:
		return True
	if _is_all_digit_entity_ids(entities):
		return True
	if rela in {"time", "number", "date"}:
		return True
	return False


def _get_semantic_model():
	global _SEMANTIC_MODEL
	if _SEMANTIC_MODEL is not None:
		return _SEMANTIC_MODEL

	try:
		from sentence_transformers import SentenceTransformer
	except Exception as e:
		err_msg = (
			"Failed to import sentence-transformers while initializing PoG semantic model. "
			f"original_error={type(e).__name__}: {e}"
		)
		logging.error(err_msg)
		raise RuntimeError(err_msg) from None

	configured_model = os.getenv("POG_SENTENCE_MODEL", "").strip()
	if configured_model:
		model_name_or_path = configured_model
	else:
		repo_root = Path(__file__).resolve().parents[5]
		local_model_path = repo_root / "models" / "sentence-transformers" / "msmarco-distilbert-base-tas-b"
		if local_model_path.exists():
			model_name_or_path = str(local_model_path)
		else:
			model_name_or_path = "sentence-transformers/msmarco-distilbert-base-tas-b"

	logging.info("[PoG] Initializing SentenceTransformer from: %s", model_name_or_path)
	try:
		_SEMANTIC_MODEL = SentenceTransformer(model_name_or_path)
	except Exception as e:
		err_msg = (
			"Failed to initialize SentenceTransformer at startup. "
			"Set POG_SENTENCE_MODEL to a local model directory or ensure network access to HuggingFace. "
			f"model_name_or_path={model_name_or_path}. "
			f"original_error={type(e).__name__}: {e}"
		)
		logging.error(err_msg)
		raise RuntimeError(err_msg) from None
	return _SEMANTIC_MODEL


def initialize_semantic_model():
	"""Eagerly initialize semantic model for PoG at startup."""
	return _get_semantic_model()


def _semantic_topn(question: str, entities: List[Entity], topn: int) -> List[Entity]:
	if len(entities) <= topn:
		return entities

	from sentence_transformers import util

	model = _get_semantic_model()
	entity_names = [str(entity.name or entity.id) for entity in entities]
	query_emb = model.encode(question)
	doc_emb = model.encode(entity_names)
	scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
	scored_entities = sorted(zip(entities, scores), key=lambda item: float(item[1]), reverse=True)
	return [item[0] for item in scored_entities[:topn]]


def _collect_entity_name_map(candidate_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Entity]:
	name_map: Dict[str, Entity] = {}
	for group in candidate_groups.values():
		entities = _normalize_group_entities(group)
		for ent in entities:
			name_key = (ent.name or "").strip().lower()
			if name_key and name_key not in name_map:
				name_map[name_key] = ent
	return name_map


def _fallback_pick_entities(text_context: str, name_map: Dict[str, Entity]) -> Tuple[List[Entity], List[float]]:
	selected: List[Entity] = []
	for name_key, ent in name_map.items():
		if name_key and name_key in text_context.lower():
			selected.append(ent)
	if not selected:
		selected = list(name_map.values())[:3]
	if not selected:
		return [], []
	uniform = 1.0 / float(len(selected))
	return selected, [uniform] * len(selected)


def _parse_entity_scores(decision: Dict[str, Any], name_map: Dict[str, Entity]) -> Tuple[List[Entity], List[float]]:
	entity_names_raw = decision.get("entity_by_relevance", [])
	scores_raw = decision.get("entity_score", [])

	if not isinstance(entity_names_raw, list):
		entity_names_raw = []
	if not isinstance(scores_raw, list):
		scores_raw = []

	selected_entities: List[Entity] = []
	selected_scores: List[float] = []
	for idx, name in enumerate(entity_names_raw):
		if not isinstance(name, str):
			continue
		name_key = name.strip().lower()
		if not name_key or name_key not in name_map:
			continue
		ent = name_map[name_key]
		if any(existing.id == ent.id for existing in selected_entities):
			continue
		selected_entities.append(ent)
		try:
			score = float(scores_raw[idx]) if idx < len(scores_raw) else 1.0
		except Exception:
			score = 1.0
		selected_scores.append(max(0.0, score))

	if not selected_entities:
		return [], []

	total = sum(selected_scores)
	if total <= 0:
		selected_scores = [1.0 / len(selected_scores)] * len(selected_scores)
	else:
		selected_scores = [score / total for score in selected_scores]
	return selected_entities, selected_scores


def build_entity_condition_prompt(
	question: str,
	triples_text: str,
	prompt_template: str,
) -> str:
	return (
		prompt_template
		.replace("{{question}}", question)
		.replace("{{triples}}", triples_text)
	)


def entity_condition_prune(
	inp: EntityConditionPruneInput,
	llm_generate: Callable[..., Any],
) -> EntityConditionPruneOutput:
	warnings: List[str] = []
	all_pruned_entities: List[Entity] = []
	all_pruned_scores: List[float] = []
	all_text_contexts: List[str] = []
	prompt_blocks: List[str] = []
	response_blocks: List[str] = []
	total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

	for group_id, group in sorted(inp.candidate_groups.items(), key=lambda item: str(item[0])):
		relation_name = _normalize_group_rel_name(group)
		head = _normalize_group_head(group)
		entities = _normalize_group_entities(group)
		topic_name = _normalize_group_topic_name(group, inp.topic_entity_name)

		if not entities:
			continue

		if _needs_no_prune(relation_name, entities):
			sorted_entities = sorted(entities, key=lambda item: str(item.name or item.id))
			for ent in sorted_entities:
				all_pruned_entities.append(ent)
				all_pruned_scores.append(1.0)
			continue

		if all(_is_mid_like(ent.id) for ent in entities) and len(entities) > 10:
			entities = random.sample(entities, 10)
			warnings.append(f"Group {group_id}: sampled 10 MID entities before pruning.")

		if len(entities) > 70:
			entities = _semantic_topn(inp.question, entities, 70)
			warnings.append(f"Group {group_id}: reduced candidates to semantic top-70 before pruning.")

		sorted_entities = sorted(entities, key=lambda item: str(item.name or item.id))
		sorted_entity_names = [str(ent.name or ent.id) for ent in sorted_entities]
		triples_text = f"{topic_name} {relation_name} {sorted_entity_names}"
		text_context = triples_text
		all_text_contexts.append(text_context)

		prompt = build_entity_condition_prompt(
			question=inp.question,
			triples_text=triples_text,
			prompt_template=inp.user_prompt,
		)
		prompt_blocks.append(prompt)

		response, usage = _run_llm(
			llm_generate=llm_generate,
			prompt=prompt,
			temperature=inp.temperature,
			max_tokens=inp.max_tokens,
			max_retries=inp.max_retries,
			system_prompt=inp.system_prompt,
		)
		if response:
			response_blocks.append(response)

		for key in total_usage.keys():
			total_usage[key] += int(usage.get(key, 0) or 0)

		name_map: Dict[str, Entity] = {}
		for ent in sorted_entities:
			name_key = str(ent.name or ent.id).strip().lower()
			if name_key and name_key not in name_map:
				name_map[name_key] = ent

		if not response:
			warnings.append(f"Group {group_id}: empty response; skip this relation group.")
			continue

		selected_names = _parse_entity_name_list(response)
		if not selected_names:
			warnings.append(f"Group {group_id}: failed to parse selected entities list; skip this relation group.")
			continue

		selected_names = sorted(selected_names)
		for name in selected_names:
			name_key = str(name).strip().lower()
			if not name_key:
				continue
			if name_key not in name_map:
				continue
			all_pruned_entities.append(name_map[name_key])
			all_pruned_scores.append(1.0)

	if not all_pruned_entities:
		warnings.append("No candidate entities available after pruning.")

	merged_text_context = "\n\n".join(all_text_contexts).strip()
	merged_prompt = "\n\n---\n\n".join(prompt_blocks)
	merged_response = "\n\n---\n\n".join(response_blocks) if response_blocks else None

	return EntityConditionPruneOutput(
		success=bool(all_pruned_entities),
		pruned_entities=all_pruned_entities,
		pruned_scores=all_pruned_scores,
		text_context=merged_text_context,
		prompt=merged_prompt,
		response=merged_response,
		usage=total_usage,
		warnings=warnings,
	)
