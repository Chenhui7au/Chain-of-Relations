#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   entity_prune.py
@Time    :   2026/03/10
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import random
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parent.parent

from chain_of_relations.kg_backend import KGBackend, get_default_backend
from chain_of_relations.schema import Entity, Relation


@dataclass
class EntityPruneBranchInput:
	source_entity: Entity
	relation: Relation
	candidate_entity_ids: List[str]
	candidate_entity_names: List[str] = field(default_factory=list)


@dataclass
class EntityPruneInput:
	question: str
	branches: List[EntityPruneBranchInput]
	user_prompt: str
	entity_width: int = 3
	sample_entity_threshold: int = 500
	drop_unnamed_entity: bool = True
	temperature: float = 0.3
	max_tokens: int = 512
	max_retries: int = 3
	system_prompt: Optional[str] = None
	random_seed: Optional[int] = None


@dataclass
class EntityPruneOutput:
	success: bool
	chain_of_entities: List[List[Tuple[str, str, str]]]
	selected_entities: List[Entity]
	selected_entity_ids: List[str]
	selected_relations: List[Relation]
	selected_heads: List[bool]
	prompt_records: List[Dict[str, Any]]
	usage: Dict[str, Any]
	candidate_size: int
	pruned_candidate_size: int
	warnings: List[str] = field(default_factory=list)


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


def _safe_usage_value(usage: Dict[str, Any], key: str) -> int:
	try:
		return int(usage.get(key, 0) or 0)
	except Exception:
		return 0


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
	payload = (text or "").strip()
	if not payload:
		return None

	candidates: List[str] = [payload]
	fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", payload, flags=re.IGNORECASE)
	candidates.extend(block.strip() for block in fenced if block.strip())
	candidates.extend(match.group(0) for match in re.finditer(r"\{[\s\S]*?\}", payload))

	for candidate in candidates:
		try:
			parsed = json.loads(candidate)
		except Exception:
			continue
		if isinstance(parsed, dict):
			return parsed

	return None


def _extract_entity_scores(text: str, ordered_names: Sequence[str]) -> Tuple[List[float], str]:
	parsed_json = _try_parse_json_object(text)
	if not isinstance(parsed_json, dict):
		return [], "json_not_found"

	score_raw = parsed_json.get("entity_score")
	entity_raw = parsed_json.get("entity_by_relevance")

	if not isinstance(score_raw, list):
		return [], "json_missing_entity_score"

	scores: List[float] = []
	for item in score_raw:
		try:
			scores.append(float(item))
		except Exception:
			return [], "json_invalid_entity_score"

	if (
		isinstance(entity_raw, list)
		and len(entity_raw) == len(scores)
		and len(scores) == len(ordered_names)
	):
		index_map: Dict[str, int] = {}
		for idx, name in enumerate(entity_raw):
			name_str = str(name)
			if name_str not in index_map:
				index_map[name_str] = idx

		if all(name in index_map for name in ordered_names):
			aligned_scores = [scores[index_map[name]] for name in ordered_names]
			return aligned_scores, "json_entity_by_relevance"

	return scores, "json_entity_score"


def _normalize_scores(scores: List[float], size: int) -> List[float]:
	if size <= 0:
		return []
	if len(scores) != size:
		return [1.0 / size] * size
	total = sum(scores)
	if total <= 0:
		return [1.0 / size] * size
	return [float(score) / float(total) for score in scores]


def _build_entity_score_prompt(
	question: str,
	relation: str,
	entity_candidates: Sequence[str],
	prompt_template: str,
) -> str:
	return (
		prompt_template
		.replace("{{question}}", question)
		.replace("{{relation}}", relation)
		.replace("{{entities}}", "; ".join(entity_candidates))
	)


def _score_branch_candidates(
	inp: EntityPruneInput,
	branch: EntityPruneBranchInput,
	llm_generate: Callable[..., Any],
	name_resolver: Callable[[str], str],
	rng: random.Random,
) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any], List[str], Dict[str, int]]:
	branch_warnings: List[str] = []
	record_usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
	stats = {"before_sample": len(branch.candidate_entity_ids), "after_sample": 0}

	base_score = float(branch.relation.score)
	if base_score <= 0:
		base_score = 1.0

	candidate_ids = list(branch.candidate_entity_ids)
	candidate_names = list(branch.candidate_entity_names or [])
	use_provided_names = bool(candidate_names) and len(candidate_names) == len(candidate_ids)
	if candidate_names and not use_provided_names:
		branch_warnings.append(
			f"candidate_entity_names length mismatch for relation '{branch.relation.id}', fallback to name resolver."
		)
	if len(candidate_ids) >= max(1, int(inp.sample_entity_threshold)):
		keep_num = min(max(1, int(inp.sample_entity_threshold)), len(candidate_ids))
		indices = rng.sample(range(len(candidate_ids)), keep_num)
		candidate_ids = [candidate_ids[idx] for idx in indices]
		if use_provided_names:
			candidate_names = [candidate_names[idx] for idx in indices]
	stats["after_sample"] = len(candidate_ids)

	if not candidate_ids:
		record = {
			"type": "entity_score",
			"relation": branch.relation.id,
			"head": branch.relation.left,
			"source_entity": branch.source_entity.id,
			"prompt": None,
			"response": None,
			"candidate_size": 0,
			"filtered_candidate_size": 0,
			"input_tokens": 0,
			"output_tokens": 0,
			"parsed_result": [],
			"status": "no entity candidates",
		}
		return [], record, branch_warnings, stats

	if not (use_provided_names and len(candidate_names) == len(candidate_ids)):
		candidate_names = [name_resolver(entity_id) for entity_id in candidate_ids]
	name_id_pairs = list(zip(candidate_names, candidate_ids))

	if inp.drop_unnamed_entity:
		filtered_pairs = [pair for pair in name_id_pairs if pair[0] != "UnName_Entity"]
		if filtered_pairs:
			name_id_pairs = filtered_pairs

	if not name_id_pairs:
		uniform = [base_score / float(len(candidate_ids))] * len(candidate_ids)
		scored = [(entity_id, "UnName_Entity", score) for entity_id, score in zip(candidate_ids, uniform)]
		record = {
			"type": "entity_score",
			"relation": branch.relation.id,
			"head": branch.relation.left,
			"source_entity": branch.source_entity.id,
			"prompt": None,
			"response": None,
			"candidate_size": len(candidate_ids),
			"filtered_candidate_size": len(candidate_ids),
			"input_tokens": 0,
			"output_tokens": 0,
			"parsed_result": uniform,
			"status": "all entities unknown",
		}
		return scored, record, branch_warnings, stats

	name_id_pairs = sorted(name_id_pairs, key=lambda item: item[0])
	ordered_names = [pair[0] for pair in name_id_pairs]
	ordered_ids = [pair[1] for pair in name_id_pairs]

	if len(ordered_ids) == 1:
		scored = [(ordered_ids[0], ordered_names[0], base_score)]
		record = {
			"type": "entity_score",
			"relation": branch.relation.id,
			"head": branch.relation.left,
			"source_entity": branch.source_entity.id,
			"prompt": None,
			"response": None,
			"candidate_size": 1,
			"filtered_candidate_size": 1,
			"input_tokens": 0,
			"output_tokens": 0,
			"parsed_result": [base_score],
			"status": "single entity",
		}
		return scored, record, branch_warnings, stats

	prompt = _build_entity_score_prompt(
		question=inp.question,
		relation=branch.relation.id,
		entity_candidates=ordered_names,
		prompt_template=inp.user_prompt,
	)

	response, usage = _run_llm(
		llm_generate=llm_generate,
		prompt=prompt,
		temperature=inp.temperature,
		max_tokens=inp.max_tokens,
		max_retries=inp.max_retries,
		system_prompt=inp.system_prompt,
	)

	record_usage = usage
	if not response:
		branch_warnings.append(
			f"LLM returned empty response for relation '{branch.relation.id}', falling back to uniform scores."
		)
		normalized = [1.0 / float(len(ordered_ids))] * len(ordered_ids)
	else:
		parsed_scores, score_source = _extract_entity_scores(response, ordered_names)
		normalized = _normalize_scores(parsed_scores, len(ordered_ids))
		if len(parsed_scores) != len(ordered_ids):
			branch_warnings.append(
				f"Score count mismatch for relation '{branch.relation.id}' ({score_source}), fallback to uniform scores."
			)

	weighted = [float(score) * base_score for score in normalized]
	scored = [
		(entity_id, entity_name, entity_score)
		for entity_id, entity_name, entity_score in zip(ordered_ids, ordered_names, weighted)
	]

	record = {
		"type": "entity_score",
		"relation": branch.relation.id,
		"head": branch.relation.left,
		"source_entity": branch.source_entity.id,
		"prompt": prompt,
		"response": response,
		"candidate_size": len(candidate_ids),
		"filtered_candidate_size": len(ordered_ids),
		"input_tokens": _safe_usage_value(usage, "input_tokens"),
		"output_tokens": _safe_usage_value(usage, "output_tokens"),
		"parsed_result": weighted,
		"status": "ok",
	}
	return scored, record, branch_warnings, stats


def entity_prune(
	inp: EntityPruneInput,
	llm_generate: Callable[..., Any],
	name_resolver: Optional[Callable[[str], str]] = None,
	backend: Optional[KGBackend] = None,
) -> EntityPruneOutput:
	warnings: List[str] = []
	resolved_backend = backend or get_default_backend()
	resolved_name_resolver = name_resolver or resolved_backend.id2entity_name_or_type
	prompt_records: List[Dict[str, Any]] = []
	agg_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
	all_candidates: List[Tuple[str, str, Relation, Entity, float]] = []

	rng = random.Random(inp.random_seed)

	for branch in inp.branches:
		scored_candidates, record, branch_warnings, stats = _score_branch_candidates(
			inp=inp,
			branch=branch,
			llm_generate=llm_generate,
			name_resolver=resolved_name_resolver,
			rng=rng,
		)
		record["before_sample"] = stats.get("before_sample", 0)
		record["after_sample"] = stats.get("after_sample", 0)
		prompt_records.append(record)
		warnings.extend(branch_warnings)

		agg_usage["input_tokens"] += _safe_usage_value(record, "input_tokens")
		agg_usage["output_tokens"] += _safe_usage_value(record, "output_tokens")

		for entity_id, entity_name, entity_score in scored_candidates:
			all_candidates.append((entity_id, entity_name, branch.relation, branch.source_entity, entity_score))

	agg_usage["total_tokens"] = agg_usage["input_tokens"] + agg_usage["output_tokens"]
	candidate_size = len(all_candidates)

	if not all_candidates:
		warnings.append("No candidate entities available after scoring.")
		return EntityPruneOutput(
			success=False,
			chain_of_entities=[],
			selected_entities=[],
			selected_entity_ids=[],
			selected_relations=[],
			selected_heads=[],
			prompt_records=prompt_records,
			usage=agg_usage,
			candidate_size=0,
			pruned_candidate_size=0,
			warnings=warnings,
		)

	sorted_candidates = sorted(all_candidates, key=lambda item: item[4], reverse=True)
	entity_width = max(1, int(inp.entity_width))
	truncated = sorted_candidates[:entity_width]
	filtered = [item for item in truncated if item[4] != 0]

	if not filtered:
		warnings.append("No non-zero candidates after global top-k pruning.")
		return EntityPruneOutput(
			success=False,
			chain_of_entities=[],
			selected_entities=[],
			selected_entity_ids=[],
			selected_relations=[],
			selected_heads=[],
			prompt_records=prompt_records,
			usage=agg_usage,
			candidate_size=candidate_size,
			pruned_candidate_size=0,
			warnings=warnings,
		)

	selected_entity_ids: List[str] = []
	selected_entities: List[Entity] = []
	selected_relations: List[Relation] = []
	selected_heads: List[bool] = []
	chain: List[Tuple[str, str, str]] = []

	for entity_id, entity_name, relation, source_entity, entity_score in filtered:
		selected_entity_ids.append(entity_id)
		selected_entities.append(Entity(id=entity_id, name=entity_name, score=float(entity_score)))
		selected_relations.append(Relation(id=relation.id, score=float(entity_score), left=relation.left))
		selected_heads.append(bool(relation.left))
		chain.append((source_entity.name, relation.id, entity_name))

	return EntityPruneOutput(
		success=True,
		chain_of_entities=[chain],
		selected_entities=selected_entities,
		selected_entity_ids=selected_entity_ids,
		selected_relations=selected_relations,
		selected_heads=selected_heads,
		prompt_records=prompt_records,
		usage=agg_usage,
		candidate_size=candidate_size,
		pruned_candidate_size=len(filtered),
		warnings=warnings,
	)

