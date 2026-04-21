#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   reasoning.py
@Time    :   2026/03/07 13:03:48
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


PROJECT_DIR = Path(__file__).resolve().parent.parent

from chain_of_relations.schema import Entity, Relation


@dataclass
class ReasoningInput:
	question: str
	topic_entity: Entity
	relation_chain: List[Relation]
	target_entity_candidates: List[str]
	user_prompt: str
	relation_display_names: Dict[str, str] = field(default_factory=dict)
	candidate_maxsize: Optional[int] = None
	temperature: float = 0.1
	max_tokens: int = 512
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class ReasoningOutput:
	success: bool
	action: str
	decision: Optional[str]
	answers: List[str]
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	candidate_size: int
	filtered_candidate_size: int
	warnings: List[str] = field(default_factory=list)


def _reasoning_path_to_str(
	topic_entity: Entity,
	relation_chain: List[Relation],
	target_entity_name_str: str,
	relation_display_names: Optional[Dict[str, str]] = None,
) -> str:
	lines: List[str] = []
	current_node = f"TopicEntity({topic_entity.name})"
	relation_display_names = relation_display_names or {}

	for index, relation in enumerate(relation_chain, start=1):
		next_node = f"e{index}"
		relation_text = relation_display_names.get(str(relation.id), str(relation.id))
		if relation.left:
			lines.append(f"{current_node} {relation_text} {next_node} .")
		else:
			lines.append(f"{next_node} {relation_text} {current_node} .")
		current_node = next_node

	if target_entity_name_str:
		lines.append(f"{current_node} candidate.target TargetEntity({target_entity_name_str}) .")

	return "\n".join(lines)


def _try_parse_reasoning_json(text: str) -> Optional[Dict[str, Any]]:
	if not text:
		return None

	candidates: List[str] = [text.strip()]
	fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
	candidates.extend(block.strip() for block in fenced_blocks if block.strip())
	candidates.extend(match.group(0) for match in re.finditer(r"\{[\s\S]*?\}", text))

	for candidate in candidates:
		try:
			parsed = json.loads(candidate)
		except Exception:
			continue
		if isinstance(parsed, dict):
			return parsed
	return None


def build_reasoning_prompt(
	question: str,
	topic_entity: Entity,
	relation_chain: List[Relation],
	target_entity_candidates: List[str],
	prompt_template: str,
	relation_display_names: Optional[Dict[str, str]] = None,
	candidate_maxsize: Optional[int] = None,
) -> Tuple[str, int, int]:
	candidate_size = len(target_entity_candidates)
	if candidate_maxsize is not None and candidate_size > candidate_maxsize:
		filtered_candidate_size = candidate_maxsize
		target_entity_name_str = ", ".join(target_entity_candidates[:candidate_maxsize]) + ", ..."
	else:
		filtered_candidate_size = candidate_size
		target_entity_name_str = ", ".join(target_entity_candidates)

	reasoning_path_str = _reasoning_path_to_str(
		topic_entity=topic_entity,
		relation_chain=relation_chain,
		target_entity_name_str=target_entity_name_str,
		relation_display_names=relation_display_names,
	)
	prompt = (
		prompt_template
		.replace("{{question}}", question)
		.replace("{{reasoning_path}}", reasoning_path_str)
	)
	return prompt, candidate_size, filtered_candidate_size


def _normalize_decision(decision_raw: Any) -> Optional[str]:
	if decision_raw is None:
		return None
	decision = str(decision_raw).strip().lower()
	if decision in {"forward", "backtrack", "constraint", "filter", "stop", "yes", "no"}:
		return decision
	return None


def _extract_reasoning_answer(text: str) -> Tuple[Optional[str], List[str]]:
	parsed_json = _try_parse_reasoning_json(text)
	if parsed_json is not None:
		raw_decision = parsed_json.get("decision")
		decision = _normalize_decision(raw_decision)

		raw_answers = parsed_json.get("answer", [])
		answers: List[str] = []
		if isinstance(raw_answers, list):
			answers = [str(item).strip() for item in raw_answers if str(item).strip()]
		return decision, answers

	return None, []


def _decision_to_action(decision: Optional[str]) -> str:
	if decision == "stop":
		return "Stop"
	if decision == "forward":
		return "Forward"
	if decision == "constraint":
		return "Constraint"
	if decision == "filter":
		return "Constraint"
	if decision == "backtrack":
		return "Backtrack"
	if decision == "yes":
		return "Stop"
	if decision == "no":
		return "Forward"
	return "UnknownAction"


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


def reasoning(inp: ReasoningInput, llm_generate: Callable[..., Any]) -> ReasoningOutput:
	warnings: List[str] = []
	prompt, candidate_size, filtered_candidate_size = build_reasoning_prompt(
		question=inp.question,
		topic_entity=inp.topic_entity,
		relation_chain=inp.relation_chain,
		target_entity_candidates=inp.target_entity_candidates,
		prompt_template=inp.user_prompt,
		relation_display_names=inp.relation_display_names,
		candidate_maxsize=inp.candidate_maxsize,
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
		return ReasoningOutput(
			success=False,
			action="UnknownAction",
			decision=None,
			answers=[],
			prompt=prompt,
			response=response,
			usage=usage,
			candidate_size=candidate_size,
			filtered_candidate_size=filtered_candidate_size,
			warnings=warnings,
		)

	decision, answers = _extract_reasoning_answer(response)
	if decision is None:
		warnings.append("Failed to parse decision from reasoning response JSON.")

	action = _decision_to_action(decision)
	if action == "UnknownAction":
		warnings.append("Failed to map reasoning output to valid action.")

	return ReasoningOutput(
		success=action != "UnknownAction",
		action=action,
		decision=decision,
		answers=answers,
		prompt=prompt,
		response=response,
		usage=usage,
		candidate_size=candidate_size,
		filtered_candidate_size=filtered_candidate_size,
		warnings=warnings,
	)
