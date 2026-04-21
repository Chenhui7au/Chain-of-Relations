#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   filter.py
@Time    :   2026/03/07 23:30:00
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from chain_of_relations.schema import Entity, Relation


@dataclass
class FilterInput:
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
class FilterOutput:
	success: bool
	result: str
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


def build_filter_prompt(
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


def _extract_answer(text: str) -> Tuple[bool, str]:
	if not text:
		return False, ""

	# Preferred format: strict JSON object with `answer: list[str]`
	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			answers = parsed.get("answer", [])
			if isinstance(answers, list):
				clean_answers = [str(item).strip() for item in answers if str(item).strip()]
				return True, ", ".join(clean_answers)
	except Exception:
		pass

	# Backward compatibility: legacy "Therefore, the answer is {a, b}" style
	start_index = text.find("{")
	end_index = text.find("}")
	if start_index != -1 and end_index != -1 and end_index > start_index:
		return True, text[start_index + 1 : end_index].strip()

	return False, ""


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


def filter(inp: FilterInput, llm_generate: Callable[..., Any]) -> FilterOutput:
	warnings: List[str] = []
	prompt, candidate_size, filtered_candidate_size = build_filter_prompt(
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
		return FilterOutput(
			success=False,
			result="",
			prompt=prompt,
			response=response,
			usage=usage,
			candidate_size=candidate_size,
			filtered_candidate_size=filtered_candidate_size,
			warnings=warnings,
		)

	has_parseable_answer, result = _extract_answer(response)
	if not has_parseable_answer:
		warnings.append("Failed to parse answer from response (expected JSON answer list or legacy braced answer).")

	return FilterOutput(
		success=has_parseable_answer,
		result=result,
		prompt=prompt,
		response=response,
		usage=usage,
		candidate_size=candidate_size,
		filtered_candidate_size=filtered_candidate_size,
		warnings=warnings,
	)