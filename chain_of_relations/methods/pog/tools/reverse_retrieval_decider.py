#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   reverse_retrieval_decider.py
@Time    :   2026/03/10 23:58:54
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ReverseRetrievalDeciderInput:
	question: str
	relation: str
	head_entities_text: str
	tail_entities_text: str
	user_prompt: str
	current_entities_text: str = "[]"
	candidate_entities_text: str = "[]"
	memory: str = "{}"
	knowledge_triplets: str = ""
	temperature: float = 0.3
	max_tokens: int = 512
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class ReverseRetrievalDeciderOutput:
	success: bool
	need_reverse: bool
	reverse_entity_names: List[str]
	reason: str
	raw_decision: Dict[str, Any]
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	warnings: List[str] = field(default_factory=list)


def build_reverse_retrieval_prompt(
	question: str,
	relation: str,
	head_entities_text: str,
	tail_entities_text: str,
	current_entities_text: str,
	candidate_entities_text: str,
	memory: str,
	knowledge_triplets: str,
	prompt_template: str,
) -> str:
	prompt = (
		prompt_template
		.replace("{{question}}", question)
		.replace("{{relation}}", relation)
		.replace("{{head_entities}}", head_entities_text)
		.replace("{{tail_entities}}", tail_entities_text)
		.replace("{{current_entities}}", current_entities_text)
		.replace("{{candidate_entities}}", candidate_entities_text)
		.replace("{{memory}}", memory)
		.replace("{{knowledge_triplets}}", knowledge_triplets)
	)
	return prompt


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


def _to_bool(value: Any) -> Optional[bool]:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		v = value.strip().lower()
		if v in {"true", "yes", "1"}:
			return True
		if v in {"false", "no", "0"}:
			return False
	return None


def _to_name_list(value: Any) -> List[str]:
	if isinstance(value, list):
		return [str(item).strip() for item in value if str(item).strip()]
	if isinstance(value, str):
		text = value.strip()
		if not text:
			return []
		parts = [p.strip() for p in re.split(r"[,;\n]", text) if p.strip()]
		return parts
	return []


def _fallback_parse(response: str) -> Dict[str, Any]:
	decision: Dict[str, Any] = {}
	if not response:
		return decision

	for line in response.splitlines():
		line = line.strip()
		if not line:
			continue
		if line.lower().startswith("need_reverse:"):
			decision["need_reverse"] = line.split(":", 1)[1].strip()
		elif line.lower().startswith("reverse_entity_names:"):
			decision["reverse_entity_names"] = line.split(":", 1)[1].strip()
		elif line.lower().startswith("reason:"):
			decision["reason"] = line.split(":", 1)[1].strip()

	return decision


def reverse_retrieval_decider(
	inp: ReverseRetrievalDeciderInput,
	llm_generate: Callable[..., Any],
) -> ReverseRetrievalDeciderOutput:
	warnings: List[str] = []
	prompt = build_reverse_retrieval_prompt(
		question=inp.question,
		relation=inp.relation,
		head_entities_text=inp.head_entities_text,
		tail_entities_text=inp.tail_entities_text,
		current_entities_text=inp.current_entities_text,
		candidate_entities_text=inp.candidate_entities_text,
		memory=inp.memory,
		knowledge_triplets=inp.knowledge_triplets,
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

	if not response:
		warnings.append("LLM returned empty response.")
		return ReverseRetrievalDeciderOutput(
			success=False,
			need_reverse=False,
			reverse_entity_names=[],
			reason="",
			raw_decision={},
			prompt=prompt,
			response=response,
			usage=usage,
			warnings=warnings,
		)

	decision = _extract_json_object(response)
	if decision is None:
		decision = _fallback_parse(response)
		warnings.append("Failed to parse strict JSON; fallback line parser used.")

	need_reverse = _to_bool(decision.get("need_reverse"))
	if need_reverse is None:
		need_reverse = False
		reason_missing = True
	else:
		reason_missing = False

	reverse_names = _to_name_list(decision.get("reverse_entity_names"))
	reason = str(decision.get("reason", "")).strip()
	if reason_missing:
		warnings.append("Decision missing valid 'need_reverse'; defaulted to False.")

	return ReverseRetrievalDeciderOutput(
		success=bool(decision),
		need_reverse=need_reverse,
		reverse_entity_names=reverse_names,
		reason=reason,
		raw_decision=decision,
		prompt=prompt,
		response=response,
		usage=usage,
		warnings=warnings,
	)
