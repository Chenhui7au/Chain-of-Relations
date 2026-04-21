#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

from chain_of_relations.llm_api import LLMAPI
from chain_of_relations.kg_backend import KGBackend, get_default_backend
from chain_of_relations.schema import Entity
from chain_of_relations.tools.generate_directly import GenerateDirectlyInput, generate_directly


class IOPromptAgent:
	def __init__(
		self,
		model_name: str = "gpt-4.1-mini",
		temperature_reasoning: float = 0.1,
		max_token: int = 512,
		backend: Optional[KGBackend] = None,
		**kwargs,
	):
		self.model_name = model_name
		self.temperature_reasoning = temperature_reasoning
		self.max_token = max_token
		self.kg_backend = backend or get_default_backend()

		self.llm_api = LLMAPI(model_name=self.model_name)
		self.prompt_file = str(Path(__file__).resolve().parent / "prompt.yml")
		self.prompts = self._load_prompts(self.prompt_file)
		self._trace_seq = 0

	def _reset_trace(self) -> None:
		self._trace_seq = 0

	def _with_trace(self, payload: Dict[str, Any]) -> Dict[str, Any]:
		self._trace_seq += 1
		payload["trace_id"] = self._trace_seq
		return payload

	def _load_prompts(self, prompt_file: str) -> Dict[str, Dict[str, str]]:
		path = Path(prompt_file)
		if not path.exists():
			raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
		with path.open("r", encoding="utf-8") as f:
			content = yaml.safe_load(f) or {}
		if not isinstance(content, dict):
			raise ValueError("Prompt YAML must be a mapping at top-level")
		return content

	def _get_user_prompt(self, key: str) -> str:
		block = self.prompts.get(key)
		if not isinstance(block, dict):
			raise KeyError(f"Prompt key '{key}' not found in {self.prompt_file}")
		user_prompt = block.get("user_prompt")
		if not user_prompt:
			raise KeyError(f"Prompt key '{key}' has no user_prompt in {self.prompt_file}")
		return user_prompt

	def _get_system_prompt(self, key: str) -> Optional[str]:
		block = self.prompts.get(key)
		if not isinstance(block, dict):
			raise KeyError(f"Prompt key '{key}' not found in {self.prompt_file}")
		return block.get("system_prompt")

	def _llm_generate(
		self,
		user_prompt: str,
		temperature: float,
		max_tokens: int,
		system_prompt: Optional[str] = None,
	):
		return self.llm_api.generate(
			user_prompt=user_prompt,
			temperature=temperature,
			max_tokens=max_tokens,
			system_prompt=system_prompt,
		)

	@staticmethod
	def _normalize_topic_entities(topic_entities: List[Any]) -> List[Entity]:
		normalized_entities: List[Entity] = []
		for item in topic_entities:
			if isinstance(item, Entity):
				normalized_entities.append(item)
				continue
			if isinstance(item, dict):
				entity_id = str(item.get("id", "")).strip()
				entity_name = str(item.get("name", "")).strip()
				if entity_id and entity_name:
					normalized_entities.append(Entity(id=entity_id, name=entity_name))
		return normalized_entities

	def answer(self, question: str, topic_entities: List[Entity], **kwargs) -> Dict[str, Any]:
		_ = self._normalize_topic_entities(topic_entities)

		prompt_history = kwargs.get("prompt_history", [])
		sparql_history = kwargs.get("sparql_history", [])
		step_history = kwargs.get("step_history", [])
		self._reset_trace()

		out = generate_directly(
			GenerateDirectlyInput(
				question=question,
				user_prompt=self._get_user_prompt("generate_directly"),
				system_prompt=self._get_system_prompt("generate_directly"),
				temperature=self.temperature_reasoning,
				max_tokens=self.max_token,
			),
			llm_generate=self._llm_generate,
		)

		prompt_history.append(
			self._with_trace(
				{
					"type": "generate_directly",
					"prompt": out.prompt,
					"response": out.response,
					"parsed_result": out.result,
					"input_tokens": out.usage.get("input_tokens", -1),
					"output_tokens": out.usage.get("output_tokens", -1),
					"warnings": out.warnings,
				}
			)
		)

		results = [out.result] if out.result else []
		return {
			"action": "generate_directly",
			"question": question,
			"results": results,
			"reasoning_chains": [],
			"prompt_history": prompt_history,
			"sparql_history": sparql_history,
			"step_history": step_history,
		}
