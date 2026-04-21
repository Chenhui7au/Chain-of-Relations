#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   memory_update.py
@Time    :   2026/03/10 23:58:58
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
class MemoryUpdateInput:
	question: str
	subquestions: List[str]
	memory: str
	knowledge_triplets: str
	user_prompt: str
	temperature: float = 0.3
	max_tokens: int = 1024
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class MemoryUpdateOutput:
	success: bool
	updated_memory: str
	memory_obj: Dict[str, Any]
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	warnings: List[str] = field(default_factory=list)


def build_memory_update_prompt(
	question: str,
	subquestions: List[str],
	memory: str,
	knowledge_triplets: str,
	prompt_template: str,
) -> str:
	prompt = (
		prompt_template
		.replace("{{question}}", question)
		.replace("{{subquestions}}", str(subquestions))
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


def _try_parse_memory_obj(text: str) -> Dict[str, Any]:
	payload = (text or "").strip()
	if not payload:
		return {}

	candidates: List[str] = [payload]
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

	return {}


def memory_update(
	inp: MemoryUpdateInput,
	llm_generate: Callable[..., Any],
) -> MemoryUpdateOutput:
	warnings: List[str] = []
	prompt = build_memory_update_prompt(
		question=inp.question,
		subquestions=inp.subquestions,
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
		return MemoryUpdateOutput(
			success=False,
			updated_memory=inp.memory,
			memory_obj={},
			prompt=prompt,
			response=response,
			usage=usage,
			warnings=warnings,
		)

	memory_obj = _try_parse_memory_obj(response)
	if not memory_obj:
		warnings.append("Failed to parse updated memory JSON object from response.")
		return MemoryUpdateOutput(
			success=False,
			updated_memory=inp.memory,
			memory_obj={},
			prompt=prompt,
			response=response,
			usage=usage,
			warnings=warnings,
		)

	updated_memory = json.dumps(memory_obj, ensure_ascii=False)
	return MemoryUpdateOutput(
		success=True,
		updated_memory=updated_memory,
		memory_obj=memory_obj,
		prompt=prompt,
		response=response,
		usage=usage,
		warnings=warnings,
	)
