#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   generate_directly.py
@Time    :   2026/03/07 14:21:05
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class GenerateDirectlyInput:
	question: str
	user_prompt: str
	temperature: float = 0.1
	max_tokens: int = 512
	max_retries: int = 3
	system_prompt: Optional[str] = None


@dataclass
class GenerateDirectlyOutput:
	success: bool
	result: str
	prompt: str
	response: Optional[str]
	usage: Dict[str, Any]
	warnings: List[str] = field(default_factory=list)


def build_generate_directly_prompt(question: str, prompt_template: str) -> str:
	return prompt_template.replace("{{question}}", question)


def _extract_answer(text: str) -> str:
	if not text:
		return ""

	# Preferred format: strict JSON object with `answer: list[str]`
	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			answers = parsed.get("answer", [])
			if isinstance(answers, list):
				clean_answers = [str(item).strip() for item in answers if str(item).strip()]
				return ", ".join(clean_answers)
	except Exception:
		pass

	# Backward compatibility: legacy "Therefore, the answer is {...}" format
	start_index = text.find("{")
	end_index = text.find("}")
	if start_index != -1 and end_index != -1 and end_index > start_index:
		return text[start_index + 1 : end_index].strip()

	return ""


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


def generate_directly(
	inp: GenerateDirectlyInput,
	llm_generate: Callable[..., Any],
) -> GenerateDirectlyOutput:
	warnings: List[str] = []
	prompt = build_generate_directly_prompt(inp.question, inp.user_prompt)

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
		return GenerateDirectlyOutput(
			success=False,
			result="",
			prompt=prompt,
			response=response,
			usage=usage,
			warnings=warnings,
		)

	result = _extract_answer(response)
	if not result:
		warnings.append("Failed to parse answer from response (expected JSON answer list or legacy braced answer).")

	return GenerateDirectlyOutput(
		success=bool(result),
		result=result,
		prompt=prompt,
		response=response,
		usage=usage,
		warnings=warnings,
	)

