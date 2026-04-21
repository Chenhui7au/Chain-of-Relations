#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   reasoning.py
@Time    :   2026/03/10 23:59:01
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class PoGReasoningInput:
	question: str
	entity_candidates: str
	memory: str
	knowledge_triplets: str
	user_prompt: str
	temperature: float = 0.3
	max_tokens: int = 1024
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class PoGReasoningOutput:
	success: bool
	action: str
	retrieve_entity: Optional[str]
	sufficient: Optional[bool]
	answer: Optional[str]
	reason: str
	raw_decision: Dict[str, Any]
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	warnings: list = field(default_factory=list)


def build_pog_reasoning_prompt(
	question: str,
	entity_candidates: str,
	memory: str,
	knowledge_triplets: str,
	prompt_template: str,
) -> str:
	prompt = (
		prompt_template
		.replace("{{question}}", question)
		.replace("{{entity_candidates}}", entity_candidates)
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


def _normalize_action(value: Any) -> str:
	if not isinstance(value, str):
		return "retrieve"
	v = value.strip().lower()
	if v in {"answer", "a"}:
		return "answer"
	if v in {"retrieve", "r", "search"}:
		return "retrieve"
	return "retrieve"


def _parse_fallback_lines(text: str) -> Dict[str, Any]:
	decision: Dict[str, Any] = {}
	if not text:
		return decision

	for line in text.splitlines():
		line = line.strip()
		if not line:
			continue
		if line.startswith("A:"):
			decision["A"] = line.split(":", 1)[1].strip()
		elif line.startswith("R:"):
			decision["R"] = line.split(":", 1)[1].strip()
		elif line.lower().startswith("sufficient:"):
			decision["Sufficient"] = line.split(":", 1)[1].strip()
		elif line.lower().startswith("answer:"):
			decision["Answer"] = line.split(":", 1)[1].strip()
		elif line.lower().startswith("reason:"):
			decision["Reason"] = line.split(":", 1)[1].strip()
	return decision


def _extract_answer_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	if isinstance(value, list):
		parts = [str(item).strip() for item in value if str(item).strip()]
		if not parts:
			return None
		return ", ".join(parts)
	text = str(value).strip()
	return text or None


def _looks_like_reason(text: str) -> bool:
	words = [token for token in text.strip().split() if token]
	return len(words) >= 4


def pog_reasoning(
	inp: PoGReasoningInput,
	llm_generate: Callable[..., Any],
) -> PoGReasoningOutput:
	warnings = []
	prompt = build_pog_reasoning_prompt(
		question=inp.question,
		entity_candidates=inp.entity_candidates,
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
		return PoGReasoningOutput(
			success=False,
			action="retrieve",
			retrieve_entity=None,
			sufficient=None,
			answer=None,
			reason="",
			raw_decision={},
			prompt=prompt,
			response=response,
			usage=usage,
			warnings=warnings,
		)

	decision = _extract_json_object(response)
	if decision is None:
		decision = _parse_fallback_lines(response)
		warnings.append("Failed to parse strict JSON; fallback line parser used.")

	action = _normalize_action(decision.get("A"))
	retrieve_entity: Optional[str] = None
	sufficient = _to_bool(decision.get("Sufficient"))
	answer: Optional[str] = None
	reason = ""

	if isinstance(decision.get("A"), dict):
		a_obj = decision.get("A") or {}
		sufficient = _to_bool(a_obj.get("Sufficient"))
		answer = _extract_answer_text(a_obj.get("Answer"))
		reason = str(decision.get("R", "")).strip()
		if sufficient is True:
			action = "stop"
		else:
			action = "retrieve"
	else:
		answer = _extract_answer_text(decision.get("Answer"))
		reason_candidate = decision.get("Reason")
		if isinstance(reason_candidate, str):
			reason = reason_candidate.strip()

		r_field = decision.get("R")
		if isinstance(r_field, str):
			r_value = r_field.strip()
			if r_value:
				if not reason and _looks_like_reason(r_value):
					reason = r_value
				else:
					retrieve_entity = r_value

	if action == "answer" and sufficient is None:
		sufficient = True
	if action == "retrieve" and sufficient is None:
		sufficient = False

	if answer is not None:
		answer_lower = answer.lower()
		if answer_lower in {"null", "none", ""}:
			action = "retrieve"
			sufficient = False
		if answer.startswith("m.") or answer.startswith("['m.") or answer.startswith('[\"m.'):
			action = "retrieve"
			sufficient = False

	success = bool(decision)
	return PoGReasoningOutput(
		success=success,
		action=action,
		retrieve_entity=retrieve_entity,
		sufficient=sufficient,
		answer=answer,
		reason=reason,
		raw_decision=decision,
		prompt=prompt,
		response=response,
		usage=usage,
		warnings=warnings,
	)
