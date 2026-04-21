#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   relation_prune.py
@Time    :   2026/03/06 17:42:10
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

from chain_of_relations.schema import Entity, Relation


@dataclass
class RelationPruneInput:
	question: str
	topic_entity: Entity
	relation_chain: List[Relation]
	head_relations: List[str]
	tail_relations: List[str]
	user_prompt: str
	top_k: int = 3
	sample_relation_threshold: int = 500
	relation_display_names: Optional[Dict[str, str]] = None
	render_mode: str = "directional"  # directional | unified
	temperature: float = 0.3
	max_tokens: int = 512
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class RelationPruneOutput:
	success: bool
	selected_relations: List[Relation]
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	candidate_size: int
	pruned_candidate_size: int
	warnings: List[str] = field(default_factory=list)


def build_relation_prune_prompt(
	question: str,
	topic_entity: Entity,
	relation_chain: List[Relation],
	head_relations: List[str],
	tail_relations: List[str],
	relation_display_names: Optional[Dict[str, str]],
	prompt_template: str,
	render_mode: str = "directional",
) -> str:
	prompt = prompt_template.replace("{{question}}", question)
	mode = (render_mode or "directional").strip().lower()
	if mode not in ("directional", "unified"):
		mode = "directional"

	def _display_name(relation_id: str) -> str:
		if relation_display_names is None:
			return relation_id
		name = str(relation_display_names.get(relation_id, "")).strip()
		return name or relation_id

	if mode == "unified":
		all_relations: List[str] = []
		seen = set()
		for relation in head_relations + tail_relations:
			if relation in seen:
				continue
			seen.add(relation)
			all_relations.append(_display_name(relation))
		relations_str = "Relations: " + "; ".join(all_relations)
		if all_relations:
			relations_str += ";"
	else:
		forward_snippet, backward_snippet = _build_relation_snippets(topic_entity, relation_chain, relation_display_names)
		relations_str = ""
		if head_relations:
			relations_str += (
				f"# {forward_snippet}\n"
				f"{'; '.join([_display_name(relation) for relation in head_relations])};\n"
			)
		if tail_relations:
			relations_str += (
				f"# {backward_snippet}\n"
				f"{'; '.join([_display_name(relation) for relation in tail_relations])};\n"
			)
	return prompt.replace("{{relations}}", relations_str)


def _build_relation_snippets(
	topic_entity: Entity,
	relation_chain: List[Relation],
	relation_display_names: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
	def _display_name(relation_id: str) -> str:
		if relation_display_names is None:
			return relation_id
		name = str(relation_display_names.get(relation_id, "")).strip()
		return name or relation_id

	base = f"TopicEntity({topic_entity.name})"
	if not relation_chain:
		return f"{base} ?relation ?x", f"?x ?relation {base}"

	triple_parts: List[str] = []
	last_entity = base
	step = 0
	for relation in relation_chain:
		step += 1
		next_entity = f"e{step}"
		relation_name = _display_name(relation.id)
		if relation.left:
			triple_parts.append(f"{last_entity} {relation_name} {next_entity} .")
		else:
			triple_parts.append(f"{next_entity} {relation_name} {last_entity} .")
		last_entity = next_entity

	prefix = " ".join(triple_parts)
	forward_snippet = f"{prefix} {last_entity} ?relation ?x"
	backward_snippet = f"{prefix} ?x ?relation {last_entity}"
	return forward_snippet, backward_snippet


def _clean_relations(
	text: str,
	head_relations: List[str],
	tail_relations: List[str],
	relation_display_names: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Union[List[Relation], str]]:
	payload = (text or "").strip()
	if not payload:
		return False, "empty output"

	if not payload.startswith("[") and not payload.startswith("{"):
		obj_start = payload.find("{")
		obj_end = payload.rfind("}")
		arr_start = payload.find("[")
		arr_end = payload.rfind("]")
		if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
			payload = payload[obj_start : obj_end + 1]
		elif arr_start != -1 and arr_end != -1 and arr_end > arr_start:
			payload = payload[arr_start : arr_end + 1]

	try:
		items = json.loads(payload)
	except Exception:
		return False, "invalid json output"

	parsed_relation_ids: List[Tuple[Optional[int], int, str, Optional[float]]] = []
	counter = 0

	def _normalize_rank(rank_raw: Any) -> Optional[int]:
		if rank_raw is None:
			return None
		try:
			rank = int(rank_raw)
		except Exception:
			return None
		if rank <= 0:
			return None
		return rank

	def _normalize_score(score_raw: Any) -> Optional[float]:
		if score_raw is None:
			return None
		try:
			score = float(score_raw)
		except Exception:
			return None
		if score < 0:
			return 0.0
		if score > 1:
			return 1.0
		return score

	def _append_relation_id(
		relation_id_raw: Any,
		rank_raw: Any = None,
		score_raw: Any = None,
	) -> None:
		nonlocal counter
		relation_id = str(relation_id_raw).strip()
		if not relation_id:
			return
		rank = _normalize_rank(rank_raw)
		score = _normalize_score(score_raw)
		parsed_relation_ids.append((rank, counter, relation_id, score))
		counter += 1

	if isinstance(items, dict):
		relation_by_relevance = items.get("relation_by_relevance")
		relevance_score = items.get("relevance_score")

		score_by_index: Dict[int, Optional[float]] = {}
		if isinstance(relevance_score, list):
			for index, score in enumerate(relevance_score):
				score_by_index[index] = _normalize_score(score)

		if isinstance(relation_by_relevance, list):
			for index, relation_id in enumerate(relation_by_relevance):
				if isinstance(relation_id, (list, tuple)):
					if len(relation_id) >= 1:
						_append_relation_id(relation_id[0], score_raw=score_by_index.get(index))
					continue
				if isinstance(relation_id, dict):
					_append_relation_id(
						relation_id.get("relation") or relation_id.get("relation_id") or relation_id.get("id"),
						score_raw=score_by_index.get(index),
					)
					continue
				_append_relation_id(relation_id, score_raw=score_by_index.get(index))
		elif isinstance(relation_by_relevance, str):
			_append_relation_id(relation_by_relevance, score_raw=score_by_index.get(0))
		if not parsed_relation_ids:
			for relation_id in items.keys():
				_append_relation_id(relation_id)
	elif isinstance(items, list):
		for item in items:
			if not isinstance(item, dict):
				continue

			rank_raw = item.get("rank") if "rank" in item else None

			score_raw = item.get("score") if "score" in item else item.get("relevance_score")
			if "relation_id" in item:
				_append_relation_id(item.get("relation_id"), rank_raw, score_raw)
				continue

			if "relation" in item:
				_append_relation_id(item.get("relation"), rank_raw, score_raw)
				continue

			if len(item) == 1:
				relation_id, _rationale = next(iter(item.items()))
				_append_relation_id(relation_id, rank_raw, score_raw)
				continue

			if "rank" in item and len(item) == 2:
				for key, _value in item.items():
					if key == "rank":
						continue
					_append_relation_id(key, item.get("rank"), score_raw)
				continue
	else:
		return False, "json output is not an object or list"

	if not parsed_relation_ids:
		return False, "no relations found"

	parsed_relation_ids.sort(key=lambda x: (x[0] if x[0] is not None else 10**9, x[1]))

	relations: List[Relation] = []
	seen = set()
	display_to_ids: Dict[str, List[str]] = {}
	if relation_display_names:
		for relation_id, display_name in relation_display_names.items():
			name = str(display_name or "").strip()
			if not name:
				continue
			display_to_ids.setdefault(name, []).append(relation_id)

	def _resolve_relation_id(candidate_id: str) -> str:
		if candidate_id in head_relations or candidate_id in tail_relations:
			return candidate_id
		matched_ids = display_to_ids.get(candidate_id, [])
		if len(matched_ids) == 1:
			return matched_ids[0]
		return candidate_id

	for _rank, _idx, relation_id, parsed_score in parsed_relation_ids:
		relation_id = _resolve_relation_id(relation_id)
		if relation_id in seen:
			continue
		score = parsed_score if parsed_score is not None else 0.0
		in_head = relation_id in head_relations
		in_tail = relation_id in tail_relations
		if in_head:
			relations.append(Relation(id=relation_id, score=score, left=True))
			seen.add(relation_id)
		elif in_tail:
			relations.append(Relation(id=relation_id, score=score, left=False))
			seen.add(relation_id)

	if not relations:
		return False, "no relations found"
	return True, relations


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


def _cap_relation_candidates(
	head_relations: List[str],
	tail_relations: List[str],
	limit: int,
) -> Tuple[List[str], List[str], bool]:
	limit = max(1, int(limit))
	head_list = list(head_relations)
	tail_list = list(tail_relations)
	if len(head_list) + len(tail_list) <= limit:
		return head_list, tail_list, False

	combined: List[Tuple[str, bool]] = [(relation, True) for relation in head_list]
	combined.extend((relation, False) for relation in tail_list)
	truncated = combined[:limit]

	new_head: List[str] = []
	new_tail: List[str] = []
	for relation, is_head in truncated:
		if is_head:
			new_head.append(relation)
		else:
			new_tail.append(relation)
	return new_head, new_tail, True


def relation_prune(
	inp: RelationPruneInput,
	llm_generate: Callable[..., Any],
) -> RelationPruneOutput:
	candidate_size = len(inp.head_relations) + len(inp.tail_relations)
	head_relations, tail_relations, was_capped = _cap_relation_candidates(
		head_relations=inp.head_relations,
		tail_relations=inp.tail_relations,
		limit=inp.sample_relation_threshold,
	)
	pruned_candidate_size = len(head_relations) + len(tail_relations)
	warnings: List[str] = []
	if was_capped:
		warnings.append(
			f"Relation candidates capped from {candidate_size} to {pruned_candidate_size} by sample_relation_threshold={max(1, int(inp.sample_relation_threshold))}."
		)
	prompt_template = inp.user_prompt

	prompt = build_relation_prune_prompt(
		question=inp.question,
		topic_entity=inp.topic_entity,
		relation_chain=inp.relation_chain,
		head_relations=head_relations,
		tail_relations=tail_relations,
		relation_display_names=inp.relation_display_names,
		prompt_template=prompt_template,
		render_mode=inp.render_mode,
	)

	response, usage = _run_llm(
		llm_generate=llm_generate,
		prompt=prompt,
		temperature=inp.temperature,
		max_tokens=inp.max_tokens,
		max_retries=inp.max_retries,
		system_prompt=inp.system_prompt,
	)

	if not response:
		warnings.append("LLM returned empty response.")
		return RelationPruneOutput(
			success=False,
			selected_relations=[],
			prompt=prompt,
			response=response,
			usage=usage,
			candidate_size=candidate_size,
			pruned_candidate_size=pruned_candidate_size,
			warnings=warnings,
		)

	flag, parsed = _clean_relations(
		response,
		head_relations,
		tail_relations,
		inp.relation_display_names,
	)
	if not flag or not isinstance(parsed, list):
		warnings.append(f"Failed to parse relation prune result: {parsed}")
		return RelationPruneOutput(
			success=False,
			selected_relations=[],
			prompt=prompt,
			response=response,
			usage=usage,
			candidate_size=candidate_size,
			pruned_candidate_size=pruned_candidate_size,
			warnings=warnings,
		)

	valid_relations: List[Relation] = []
	for relation in parsed:
		if relation.left and relation.id in head_relations:
			valid_relations.append(relation)
		elif (not relation.left) and relation.id in tail_relations:
			valid_relations.append(relation)

	if inp.relation_chain:
		previous_relation = inp.relation_chain[-1]
		acyclic_relations: List[Relation] = []
		for relation in valid_relations:
			if previous_relation.left and (not relation.left) and relation.id == previous_relation.id:
				continue
			if (not previous_relation.left) and relation.left and relation.id == previous_relation.id:
				continue
			acyclic_relations.append(relation)
		valid_relations = acyclic_relations

	top_k = max(1, int(inp.top_k))
	valid_relations = valid_relations[:top_k]
	pruned_candidate_size = len(valid_relations)

	if not valid_relations:
		warnings.append("No valid relations after validation/acyclic filtering.")
		return RelationPruneOutput(
			success=False,
			selected_relations=[],
			prompt=prompt,
			response=response,
			usage=usage,
			candidate_size=candidate_size,
			pruned_candidate_size=pruned_candidate_size,
			warnings=warnings,
		)

	return RelationPruneOutput(
		success=True,
		selected_relations=valid_relations,
		prompt=prompt,
		response=response,
		usage=usage,
		candidate_size=candidate_size,
		pruned_candidate_size=pruned_candidate_size,
		warnings=warnings,
	)


