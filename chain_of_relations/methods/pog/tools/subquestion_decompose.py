#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   subquestion_decompose.py
@Time    :   2026/03/10 23:59:07
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
class SubquestionDecomposeInput:
	question: str
	user_prompt: str
	temperature: float = 0.3
	max_tokens: int = 512
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class SubquestionDecomposeOutput:
	success: bool
	subquestions: List[str]
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	warnings: List[str] = field(default_factory=list)


def build_subquestion_decompose_prompt(question: str, prompt_template: str) -> str:
	return prompt_template.replace("{{question}}", question)


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


def _parse_subquestions(text: str) -> List[str]:
	payload = (text or "").strip()
	if not payload:
		return []

	for candidate in [payload] + re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", payload, flags=re.IGNORECASE):
		candidate = candidate.strip()
		if not candidate:
			continue
		try:
			obj = json.loads(candidate)
			if isinstance(obj, list):
				return [str(item).strip() for item in obj if str(item).strip()]
		except Exception:
			pass
		try:
			obj = ast.literal_eval(candidate)
			if isinstance(obj, list):
				return [str(item).strip() for item in obj if str(item).strip()]
		except Exception:
			pass

	arr_match = re.search(r"\[[\s\S]*\]", payload)
	if arr_match:
		candidate = arr_match.group(0)
		try:
			obj = ast.literal_eval(candidate)
			if isinstance(obj, list):
				return [str(item).strip() for item in obj if str(item).strip()]
		except Exception:
			pass

	lines = [line.strip("-• \t") for line in payload.splitlines()]
	return [line for line in lines if line]


def subquestion_decompose(
	inp: SubquestionDecomposeInput,
	llm_generate: Callable[..., Any],
) -> SubquestionDecomposeOutput:
	warnings: List[str] = []
	prompt = build_subquestion_decompose_prompt(inp.question, inp.user_prompt)

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
		return SubquestionDecomposeOutput(
			success=False,
			subquestions=[],
			prompt=prompt,
			response=response,
			usage=usage,
			warnings=warnings,
		)

	subquestions = _parse_subquestions(response)
	if not subquestions:
		warnings.append("Failed to parse subquestions from response.")

	return SubquestionDecomposeOutput(
		success=bool(subquestions),
		subquestions=subquestions,
		prompt=prompt,
		response=response,
		usage=usage,
		warnings=warnings,
	)
