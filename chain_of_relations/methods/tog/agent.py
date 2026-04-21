#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   agent.py
@Time    :   2026/03/10 19:25:45
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


from pathlib import Path
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

from chain_of_relations.llm_api import LLMAPI
from chain_of_relations.kg_backend import KGBackend, get_default_backend
from chain_of_relations.schema import Entity, Relation
from chain_of_relations.tools.entity_prune import EntityPruneBranchInput, EntityPruneInput, entity_prune
from chain_of_relations.tools.entity_search import EntitySearchInput, entity_search
from chain_of_relations.tools.generate_directly import GenerateDirectlyInput, generate_directly
from chain_of_relations.tools.relation_prune import RelationPruneInput, relation_prune
from chain_of_relations.tools.relation_search import RelationSearchInput, relation_search


class ToGAgent:
	def __init__(
		self,
		model_name: str = "gpt-4.1-mini",
		relation_width: int = 3,
		entity_width: int = 3,
		depth: int = 3,
		sample_relation_threshold: int = 500,
		sample_entity_threshold: int = 500,
		remove_unnecessary_rel: bool = True,
		temperature_exploration: float = 0.3,
		temperature_reasoning: float = 0.1,
		max_token: int = 512,
		backend: Optional[KGBackend] = None,
	):
		self.model_name = model_name
		self.relation_width = max(1, int(relation_width))
		self.entity_width = max(1, int(entity_width))
		self.width = self.relation_width
		self.depth = depth
		self.sample_relation_threshold = max(1, int(sample_relation_threshold))
		self.sample_entity_threshold = max(1, int(sample_entity_threshold))
		self.remove_unnecessary_rel = remove_unnecessary_rel
		self.temperature_exploration = temperature_exploration
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
		prompt_path = Path(prompt_file)
		if not prompt_path.exists():
			raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
		with prompt_path.open("r", encoding="utf-8") as file:
			content = yaml.safe_load(file) or {}
		if not isinstance(content, dict):
			raise ValueError("Prompt YAML must be a mapping at top-level")
		return content

	def _get_prompt_block(self, key: str) -> Dict[str, str]:
		aliases = {
			"relation_pruning": ["relation_pruning", "extract_relation"],
			"entity_pruning": ["entity_pruning", "entity_score"],
			"reasoning": ["reasoning"],
			"generate_directly": ["generate_directly"],
		}
		for candidate in aliases.get(key, [key]):
			block = self.prompts.get(candidate)
			if isinstance(block, dict):
				return block
		raise KeyError(f"Prompt key '{key}' not found in {self.prompt_file}")

	def _get_user_prompt(self, key: str) -> str:
		block = self._get_prompt_block(key)
		user_prompt = block.get("user_prompt")
		if not user_prompt:
			raise KeyError(f"Prompt key '{key}' has no user_prompt in {self.prompt_file}")
		return user_prompt

	def _get_system_prompt(self, key: str) -> Optional[str]:
		block = self._get_prompt_block(key)
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
				entity_id = item.get("id", "")
				entity_name = item.get("name", "")
				if entity_id and entity_name:
					normalized_entities.append(Entity(entity_id, entity_name, 0.0))
		return normalized_entities

	@staticmethod
	def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
		payload = (text or "").strip()
		if not payload:
			return None

		candidates: List[str] = [payload]
		fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", payload, flags=re.IGNORECASE)
		candidates.extend(block.strip() for block in fenced_blocks if block.strip())
		candidates.extend(match.group(0) for match in re.finditer(r"\{[\s\S]*?\}", payload))

		for candidate in candidates:
			try:
				parsed = json.loads(candidate)
			except Exception:
				continue
			if isinstance(parsed, dict):
				return parsed

		return None

	@staticmethod
	def _build_reasoning_chain_text(cluster_chain_of_entities: List[List[Tuple[str, str, str]]]) -> str:
		lines: List[str] = []
		for chain in cluster_chain_of_entities:
			for head_name, relation_id, tail_name in chain:
				lines.append(f"{head_name}, {relation_id}, {tail_name}")
		return "\n".join(lines)

	def _run_reasoning_yes_no(
		self,
		question: str,
		cluster_chain_of_entities: List[List[Tuple[str, str, str]]],
		prompt_history: List[Dict[str, Any]],
	) -> Tuple[Optional[str], List[str], Optional[str]]:
		chain_text = self._build_reasoning_chain_text(cluster_chain_of_entities)
		user_prompt = (
			self._get_user_prompt("reasoning")
			.replace("{{question}}", question)
			.replace("{{reasoning_path}}", chain_text)
		)

		response, usage = self._llm_generate(
			user_prompt=user_prompt,
			temperature=self.temperature_reasoning,
			max_tokens=self.max_token,
			system_prompt=self._get_system_prompt("reasoning"),
		)

		decision: Optional[str] = None
		answers: List[str] = []
		warnings: List[str] = []

		parsed = self._try_parse_json_object(response or "") if response else None
		if isinstance(parsed, dict):
			raw_decision = str(parsed.get("decision", "")).strip().lower()
			if raw_decision in {"yes", "no"}:
				decision = raw_decision
			else:
				warnings.append("Invalid or missing decision in reasoning JSON.")

			raw_answers = parsed.get("answer", [])
			if isinstance(raw_answers, list):
				answers = [str(item).strip() for item in raw_answers if str(item).strip()]
		else:
			warnings.append("Failed to parse reasoning JSON response.")

		prompt_history.append(
			self._with_trace(
				{
					"type": "reasoning",
					"prompt": user_prompt,
					"response": response,
					"decision": decision,
					"answers": answers,
					"parsed_result": decision,
					"candidate_size": len(cluster_chain_of_entities),
					"filtered_candidate_size": len(cluster_chain_of_entities),
					"input_tokens": usage.get("input_tokens", -1),
					"output_tokens": usage.get("output_tokens", -1),
					"warnings": warnings,
				}
			)
		)

		return decision, answers, response

	def _generate_directly(self, question: str, prompt_history: List[Dict[str, Any]]) -> str:
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

		return out.result

	def answer(self, question: str, topic_entities: List[Entity], **kwargs) -> Dict[str, Any]:
		normalized_entities = self._normalize_topic_entities(topic_entities)

		prompt_history = kwargs.get("prompt_history", [])
		sparql_history = kwargs.get("sparql_history", [])
		step_history = kwargs.get("step_history", [])
		self._reset_trace()

		if not normalized_entities:
			direct_answer = self._generate_directly(question, prompt_history)
			results = [direct_answer] if direct_answer else []
			return {
				"action": "generate_directly",
				"question": question,
				"results": results,
				"reasoning_chains": [],
				"prompt_history": prompt_history,
				"sparql_history": sparql_history,
				"step_history": step_history,
			}

		frontier_entities: List[Entity] = normalized_entities
		cluster_chain_of_entities: List[List[Tuple[str, str, str]]] = []
		pre_relations: List[str] = []
		pre_heads: List[Optional[bool]] = [None] * len(frontier_entities)

		for depth in range(1, self.depth + 1):
			logging.info("[ToG] depth=%s | frontier_size=%s", depth, len(frontier_entities))

			branches: List[EntityPruneBranchInput] = []

			for index, frontier_entity in enumerate(frontier_entities):
				current_pre_head = pre_heads[index] if index < len(pre_heads) else None
				relation_out = relation_search(
					RelationSearchInput(
						topic_entity=frontier_entity,
						relation_chain=[],
						remove_unnecessary_rel=self.remove_unnecessary_rel,
						prune_inverse_of_last_hop=False,
					),
					backend=self.kg_backend,
				)

				head_relations = list(relation_out.head_relations)
				tail_relations = list(relation_out.tail_relations)
				if pre_relations:
					pre_relation_set = set(pre_relations)
					if current_pre_head is True:
						tail_relations = [relation for relation in tail_relations if relation not in pre_relation_set]
					elif current_pre_head is False:
						head_relations = [relation for relation in head_relations if relation not in pre_relation_set]

				relation_display_names = self.kg_backend.relation_ids2labels(head_relations + tail_relations)

				sparql_history.append(
					self._with_trace(
						{
							"search_type": "relation_search_head",
							"sparql": relation_out.head_query,
							"result_count": relation_out.raw_head_count,
							"status": relation_out.head_status,
							"timed_out": relation_out.head_timed_out,
						}
					)
				)
				sparql_history.append(
					self._with_trace(
						{
							"search_type": "relation_search_tail",
							"sparql": relation_out.tail_query,
							"result_count": relation_out.raw_tail_count,
							"status": relation_out.tail_status,
							"timed_out": relation_out.tail_timed_out,
						}
					)
				)

				prune_out = relation_prune(
					RelationPruneInput(
						question=question,
						topic_entity=frontier_entity,
						relation_chain=[],
						head_relations=head_relations,
						tail_relations=tail_relations,
						relation_display_names=relation_display_names,
						user_prompt=self._get_user_prompt("relation_pruning"),
						top_k=self.relation_width,
						sample_relation_threshold=self.sample_relation_threshold,
						render_mode="unified",
						temperature=self.temperature_exploration,
						max_tokens=self.max_token,
						system_prompt=self._get_system_prompt("relation_pruning"),
					),
					llm_generate=self._llm_generate,
				)

				prompt_history.append(
					self._with_trace(
						{
							"type": "relation_prune",
							"prompt": prune_out.prompt,
							"response": prune_out.response,
							"parsed_result": [
								{"id": rel.id, "left": rel.left, "score": rel.score}
								for rel in prune_out.selected_relations
							],
							"candidate_size": prune_out.candidate_size,
							"filtered_candidate_size": prune_out.pruned_candidate_size,
							"input_tokens": prune_out.usage.get("input_tokens", -1),
							"output_tokens": prune_out.usage.get("output_tokens", -1),
							"warnings": prune_out.warnings,
						}
					)
				)

				if not prune_out.success:
					continue

				for selected_relation in prune_out.selected_relations[: self.relation_width]:
					entity_out = entity_search(
						EntitySearchInput(
							topic_entity=frontier_entity,
							relation_chain=[selected_relation],
							filter_entity_prefix=self.kg_backend.default_entity_prefix,
							drop_unnamed_entity=True,
						),
						backend=self.kg_backend,
					)

					sparql_history.append(
						self._with_trace(
							{
								"search_type": "entity_search",
								"sparql": entity_out.query,
								"result_count": entity_out.entity_count,
								"status": entity_out.query_status,
								"timed_out": entity_out.timed_out,
								"filter_caused_empty": entity_out.filter_caused_empty,
							}
						)
					)

					if not entity_out.entity_ids:
						continue

					branches.append(
						EntityPruneBranchInput(
							source_entity=frontier_entity,
							relation=selected_relation,
							candidate_entity_ids=entity_out.entity_ids,
							candidate_entity_names=entity_out.entity_names,
						)
					)

			if not branches:
				direct_answer = self._generate_directly(question, prompt_history)
				results = [direct_answer] if direct_answer else []
				return {
					"action": "generate_directly",
					"question": question,
					"results": results,
					"reasoning_chains": [],
					"prompt_history": prompt_history,
					"sparql_history": sparql_history,
					"step_history": step_history,
				}

			entity_prune_out = entity_prune(
				EntityPruneInput(
					question=question,
					branches=branches,
					user_prompt=self._get_user_prompt("entity_pruning"),
					entity_width=self.entity_width,
					sample_entity_threshold=self.sample_entity_threshold,
					temperature=self.temperature_exploration,
					max_tokens=self.max_token,
					system_prompt=self._get_system_prompt("entity_pruning"),
				),
				llm_generate=self._llm_generate,
			)

			for record in entity_prune_out.prompt_records:
				prompt_history.append(
					self._with_trace(
						{
							"type": "entity_pruning",
							"prompt": record.get("prompt"),
							"response": record.get("response"),
							"parsed_result": record.get("parsed_result", []),
							"candidate_size": record.get("candidate_size", 0),
							"filtered_candidate_size": record.get("filtered_candidate_size", 0),
							"input_tokens": record.get("input_tokens", -1),
							"output_tokens": record.get("output_tokens", -1),
							"warnings": entity_prune_out.warnings,
						}
					)
				)

			if entity_prune_out.chain_of_entities:
				cluster_chain_of_entities.extend(entity_prune_out.chain_of_entities)

			if not entity_prune_out.success:
				direct_answer = self._generate_directly(question, prompt_history)
				results = [direct_answer] if direct_answer else []
				return {
					"action": "generate_directly",
					"question": question,
					"results": results,
					"reasoning_chains": [],
					"prompt_history": prompt_history,
					"sparql_history": sparql_history,
					"step_history": step_history,
				}

			decision, answers, _response = self._run_reasoning_yes_no(
				question=question,
				cluster_chain_of_entities=cluster_chain_of_entities,
				prompt_history=prompt_history,
			)

			if decision == "yes":
				result_answers = answers
				if not result_answers:
					result_answers = [entity.name for entity in entity_prune_out.selected_entities if entity.name]
				return {
					"action": "stop",
					"question": question,
					"results": result_answers,
					"reasoning_chains": str(cluster_chain_of_entities),
					"prompt_history": prompt_history,
					"sparql_history": sparql_history,
					"step_history": step_history,
				}

			frontier_entities = entity_prune_out.selected_entities
			pre_relations = [relation.id for relation in entity_prune_out.selected_relations]
			pre_heads = [bool(head_flag) for head_flag in entity_prune_out.selected_heads]
			if not frontier_entities:
				direct_answer = self._generate_directly(question, prompt_history)
				results = [direct_answer] if direct_answer else []
				return {
					"action": "generate_directly",
					"question": question,
					"results": results,
					"reasoning_chains": [],
					"prompt_history": prompt_history,
					"sparql_history": sparql_history,
					"step_history": step_history,
				}

		direct_answer = self._generate_directly(question, prompt_history)
		results = [direct_answer] if direct_answer else []
		return {
			"action": "generate_directly",
			"question": question,
			"results": results,
			"reasoning_chains": [],
			"prompt_history": prompt_history,
			"sparql_history": sparql_history,
			"step_history": step_history,
		}
