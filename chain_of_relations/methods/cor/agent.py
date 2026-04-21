#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   agent.py
@Time    :   2026/03/07 18:04:05
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""

from pathlib import Path
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

from chain_of_relations.llm_api import LLMAPI
from chain_of_relations.kg_backend import KGBackend, get_default_backend
from chain_of_relations.schema import Entity, Relation
from chain_of_relations.methods.cor.dfs_stack import (
	DFSStackInitInput,
	PushRelationChildrenInput,
	create_dfs_stack,
	push_relation_children,
)
from chain_of_relations.tools.entity_search import EntitySearchInput, entity_search
from chain_of_relations.tools.generate_directly import GenerateDirectlyInput, generate_directly
from chain_of_relations.tools.reasoning import ReasoningInput, reasoning
from chain_of_relations.tools.relation_prune import RelationPruneInput, relation_prune
from chain_of_relations.tools.relation_search import RelationSearchInput, relation_search
from chain_of_relations.methods.cor.tools.filter import FilterInput, filter as run_filter


class CoRAgent:
	def __init__(
		self,
		model_name: str = "gpt-4.1-mini",
		relation_width: int = 3,
		depth: int = 3,
		sample_relation_threshold: int = 500,
		remove_unnecessary_rel: bool = True,
		temperature_exploration: float = 0.3,
		temperature_reasoning: float = 0.1,
		max_token: int = 512,
		backend: Optional[KGBackend] = None,
	):
		self.model_name = model_name
		self.relation_width = max(1, int(relation_width))
		self.depth = depth
		self.sample_relation_threshold = max(1, int(sample_relation_threshold))
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

	@staticmethod
	def _sparql_event_to_step(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		if "search_type" in item and "sparql" in item:
			return {
				"step_type": "sparql",
				"search_type": item.get("search_type", ""),
				"sparql": item.get("sparql", ""),
				"result_count": item.get("result_count", -1),
				"status": item.get("status", "ok"),
				"timed_out": item.get("timed_out", False),
				"filter_caused_empty": item.get("filter_caused_empty", False),
			}

		for operation, query in item.items():
			if operation in ("result_count", "trace_id"):
				continue
			return {
				"step_type": "sparql",
				"search_type": operation,
				"sparql": query,
				"result_count": item.get("result_count", -1),
				"status": item.get("status", "ok"),
				"timed_out": item.get("timed_out", False),
				"filter_caused_empty": item.get("filter_caused_empty", False),
			}
		return None

	@staticmethod
	def _llm_event_to_step(item: Dict[str, Any]) -> Dict[str, Any]:
		return {
			"step_type": "llm",
			"operation_type": item.get("type", ""),
			"prompt": item.get("prompt", ""),
			"input_tokens": item.get("input_tokens", -1),
			"output_tokens": item.get("output_tokens", -1),
			"candidate_size": item.get("candidate_size", item.get("candidates_size", -1)),
			"filtered_candidate_size": item.get("filtered_candidate_size", -1),
			"response": item.get("response", ""),
			"parsed_result": item.get("parsed_result", ""),
		}

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
			"relation_pruning": ["relation_pruning", "relation pruning", "extract_relation"],
			"constrained_relation_pruning": [
				"constrained_relation_pruning",
				"constraint_relation_pruning",
			],
			"reasoning": ["reasoning"],
			"filter": ["filter"],
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
	def _reasoning_path_to_str(
		topic_entity: Entity,
		relation_chain: List[Relation],
		target_entity_name_str: Optional[str] = None,
	) -> str:
		lines: List[str] = []
		current_node = f"TopicEntity({topic_entity.name})"

		for index, relation in enumerate(relation_chain, start=1):
			next_node = f"e{index}"
			if relation.left:
				lines.append(f"{current_node} {relation.id} {next_node} .")
			else:
				lines.append(f"{next_node} {relation.id} {current_node} .")
			current_node = next_node

		if target_entity_name_str:
			lines.append(f"{current_node} candidate.target TargetEntity({target_entity_name_str}) .")

		return "\n".join(lines)

	@staticmethod
	def _split_answers(text: str) -> List[str]:
		if not text:
			return []
		return [part.strip() for part in re.split(r",|;", text) if part.strip()]

	@staticmethod
	def _build_result_items(
		entity_ids: List[str],
		entity_names: List[str],
		literal_candidates: List[str],
	) -> List[Dict[str, str]]:
		results: List[Dict[str, str]] = []
		seen: set = set()

		for entity_id, entity_name in zip(entity_ids, entity_names):
			name = str(entity_name).strip()
			if not name:
				continue
			key = (str(entity_id).strip(), name)
			if key in seen:
				continue
			seen.add(key)
			results.append({"id": str(entity_id).strip(), "name": name})

		for literal in literal_candidates:
			name = str(literal).strip()
			if not name:
				continue
			key = ("", name)
			if key in seen:
				continue
			seen.add(key)
			results.append({"id": "", "name": name})

		return results

	@staticmethod
	def _map_answer_names_to_results(
		answer_names: List[str],
		entity_ids: List[str],
		entity_names: List[str],
	) -> List[Dict[str, str]]:
		name_to_id: Dict[str, str] = {}
		for entity_id, entity_name in zip(entity_ids, entity_names):
			name = str(entity_name).strip()
			if name and name not in name_to_id:
				name_to_id[name] = str(entity_id).strip()

		results: List[Dict[str, str]] = []
		seen: set = set()
		for answer in answer_names:
			name = str(answer).strip()
			if not name:
				continue
			item_id = name_to_id.get(name, "")
			key = (item_id, name)
			if key in seen:
				continue
			seen.add(key)
			results.append({"id": item_id, "name": name})

		return results

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
	def _expand_bidirectional_relations(
		selected_relations: List[Relation],
		head_relations: List[str],
		tail_relations: List[str],
	) -> List[Relation]:
		head_set = set(head_relations)
		tail_set = set(tail_relations)
		expanded: List[Relation] = []
		seen: set = set()

		for relation in selected_relations:
			relation_id = relation.id
			in_head = relation_id in head_set
			in_tail = relation_id in tail_set

			if in_head and in_tail:
				primary_left = bool(relation.left)
				ordered_directions = [primary_left, not primary_left]
				for direction_left in ordered_directions:
					key = (relation_id, direction_left)
					if key in seen:
						continue
					expanded.append(Relation(id=relation_id, score=relation.score, left=direction_left))
					seen.add(key)
				continue

			key = (relation_id, bool(relation.left))
			if key in seen:
				continue
			expanded.append(relation)
			seen.add(key)

		return expanded

	@staticmethod
	def _cap_relations_by_threshold(
		head_relations: List[str],
		tail_relations: List[str],
		threshold: int,
	) -> Tuple[List[str], List[str]]:
		limit = max(1, int(threshold))
		if len(head_relations) + len(tail_relations) <= limit:
			return list(head_relations), list(tail_relations)
		combined: List[Tuple[str, bool]] = [(relation, True) for relation in head_relations]
		combined.extend((relation, False) for relation in tail_relations)
		truncated = combined[:limit]
		new_head: List[str] = []
		new_tail: List[str] = []
		for relation, is_head in truncated:
			if is_head:
				new_head.append(relation)
			else:
				new_tail.append(relation)
		return new_head, new_tail

	@staticmethod
	def _relation_chain_signature(relation_chain: List[Relation]) -> str:
		if not relation_chain:
			return "<root>"
		parts: List[str] = []
		for relation in relation_chain:
			direction = "->" if relation.left else "<-"
			parts.append(f"{direction}{relation.id}")
		return " ".join(parts)

	def relation_search(
		self,
		topic_entity: Entity,
		relation_chain: List[Relation],
		sparql_history: List[Dict[str, Any]],
	) -> Tuple[List[str], List[str]]:
		out = relation_search(
			RelationSearchInput(
				topic_entity=topic_entity,
				relation_chain=relation_chain,
				remove_unnecessary_rel=self.remove_unnecessary_rel,
				prune_inverse_of_last_hop=True,
			),
			backend=self.kg_backend,
		)
		sparql_history.append(
			self._with_trace(
				{
					"search_type": "relation_search_head",
					"sparql": out.head_query,
					"result_count": out.raw_head_count,
					"status": out.head_status,
					"timed_out": out.head_timed_out,
				}
			)
		)
		sparql_history.append(
			self._with_trace(
				{
					"search_type": "relation_search_tail",
					"sparql": out.tail_query,
					"result_count": out.raw_tail_count,
					"status": out.tail_status,
					"timed_out": out.tail_timed_out,
				}
			)
		)
		return out.head_relations, out.tail_relations

	def relation_prune(
		self,
		question: str,
		topic_entity: Entity,
		relation_chain: List[Relation],
		head_relations: List[str],
		tail_relations: List[str],
		prompt_history: List[Dict[str, Any]],
	) -> List[Relation]:
		head_relations, tail_relations = self._cap_relations_by_threshold(
			head_relations=head_relations,
			tail_relations=tail_relations,
			threshold=self.sample_relation_threshold,
		)
		relation_display_names = self.kg_backend.relation_ids2labels(head_relations + tail_relations)
		out = relation_prune(
			RelationPruneInput(
				question=question,
				topic_entity=topic_entity,
				relation_chain=relation_chain,
				head_relations=head_relations,
				tail_relations=tail_relations,
				relation_display_names=relation_display_names,
				user_prompt=self._get_user_prompt("relation_pruning"),
				top_k=self.relation_width,
				sample_relation_threshold=self.sample_relation_threshold,
				system_prompt=self._get_system_prompt("relation_pruning"),
				temperature=self.temperature_exploration,
				max_tokens=self.max_token,
			),
			llm_generate=self._llm_generate,
		)
		prompt_history.append(
			self._with_trace(
				{
					"type": "relation_prune",
					"prompt": out.prompt,
					"input_tokens": out.usage.get("input_tokens", -1),
					"output_tokens": out.usage.get("output_tokens", -1),
					"candidates_size": out.candidate_size,
					"filtered_candidate_size": out.pruned_candidate_size,
					"response": out.response,
					"parsed_result": [item.id for item in out.selected_relations],
					"warnings": out.warnings,
				}
			)
		)
		return out.selected_relations if out.success else []

	def entity_search(
		self,
		topic_entity: Entity,
		relation_chain: List[Relation],
		sparql_history: List[Dict[str, Any]],
	) -> Tuple[List[str], List[str], List[str]]:
		out = entity_search(
			EntitySearchInput(
				topic_entity=topic_entity,
				relation_chain=relation_chain,
				filter_entity_prefix=self.kg_backend.default_entity_prefix,
				drop_unnamed_entity=True,
			),
			backend=self.kg_backend,
		)
		sparql_history.append(
			self._with_trace(
				{
					"search_type": "entity_search",
					"sparql": out.query,
					"result_count": out.entity_count,
					"status": out.query_status,
					"timed_out": out.timed_out,
					"filter_caused_empty": out.filter_caused_empty,
				}
			)
		)
		return out.entity_ids, out.entity_names, out.literal_entities

	def reasoning(
		self,
		question: str,
		topic_entity: Entity,
		relation_chain: List[Relation],
		target_entity_candidates: List[str],
		prompt_history: List[Dict[str, Any]],
		candidate_maxsize: Optional[int] = None,
	) -> Tuple[Optional[str], Optional[str], List[str]]:
		relation_display_names = self.kg_backend.relation_ids2labels([relation.id for relation in relation_chain])
		out = reasoning(
			ReasoningInput(
				question=question,
				topic_entity=topic_entity,
				relation_chain=relation_chain,
				target_entity_candidates=target_entity_candidates,
				user_prompt=self._get_user_prompt("reasoning"),
				relation_display_names=relation_display_names,
				system_prompt=self._get_system_prompt("reasoning"),
				candidate_maxsize=candidate_maxsize,
				temperature=self.temperature_reasoning,
				max_tokens=self.max_token,
			),
			llm_generate=self._llm_generate,
		)
		prompt_history.append(
			self._with_trace(
				{
					"type": "reasoning",
					"prompt": out.prompt,
					"response": out.response,
					"decision": out.decision,
					"answers": out.answers,
					"parsed_result": out.action,
					"candidate_size": out.candidate_size,
					"filtered_candidate_size": out.filtered_candidate_size,
					"input_tokens": out.usage.get("input_tokens", -1),
					"output_tokens": out.usage.get("output_tokens", -1),
					"warnings": out.warnings,
				}
			)
		)
		if not out.response:
			return None, None, []
		return out.action, out.response, out.answers

	def validate(
		self,
		question: str,
		topic_entity: Entity,
		relation_chain: List[Relation],
		target_entity_names: List[str],
		prompt_history: List[Dict[str, Any]],
	) -> Optional[str]:
		try:
			user_prompt = self._get_user_prompt("filter")
			system_prompt = self._get_system_prompt("filter")
		except KeyError:
			prompt_history.append(
				self._with_trace(
					{
						"type": "filter_answer",
						"prompt": None,
						"response": None,
						"parsed_result": None,
						"warnings": ["filter prompt is missing"],
					}
				)
			)
			return None

		out = run_filter(
			FilterInput(
				question=question,
				topic_entity=topic_entity,
				relation_chain=relation_chain,
				target_entity_candidates=target_entity_names,
				user_prompt=user_prompt,
				relation_display_names=self.kg_backend.relation_ids2labels([relation.id for relation in relation_chain]),
				system_prompt=system_prompt,
				temperature=self.temperature_reasoning,
				max_tokens=self.max_token,
			),
			llm_generate=self._llm_generate,
		)

		prompt_history.append(
			self._with_trace(
				{
					"type": "filter_answer",
					"prompt": out.prompt,
					"input_tokens": out.usage.get("input_tokens", -1),
					"output_tokens": out.usage.get("output_tokens", -1),
					"candidate_size": out.candidate_size,
					"filtered_candidate_size": out.filtered_candidate_size,
					"response": out.response,
					"parsed_result": out.result,
					"warnings": out.warnings,
				}
			)
		)
		return out.result if out.result else None

	def generate_directly(self, question: str, prompt_history: List[Dict[str, Any]]) -> str:
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
					"type": "generate_without_explored_paths",
					"prompt": out.prompt,
					"input_tokens": out.usage.get("input_tokens", -1),
					"output_tokens": out.usage.get("output_tokens", -1),
					"response": out.response,
					"parsed_result": out.result,
					"warnings": out.warnings,
				}
			)
		)
		return out.result

	def answer(self, question: str, topic_entities: List[Entity], **kwargs) -> Dict[str, Any]:
		if not topic_entities:
			return {
				"action": "ERROR",
				"question": question,
				"results": [],
				"reasoning_chains": "No topic_entities provided",
			}

		normalized_entities = self._normalize_topic_entities(topic_entities)
		if not normalized_entities:
			return {
				"action": "ERROR",
				"question": question,
				"results": [],
				"reasoning_chains": "Invalid topic_entities format",
			}

		prompt_history = kwargs.get("prompt_history", [])
		sparql_history = kwargs.get("sparql_history", [])
		step_history = kwargs.get("step_history", [])
		self._reset_trace()

		root_entity = normalized_entities[0]
		logging.info(
			"[CoR] Start question | root=%s(%s) | width=%s depth=%s",
			root_entity.name,
			root_entity.id,
			self.relation_width,
			self.depth,
		)
		stack = create_dfs_stack(DFSStackInitInput(root_entity=root_entity)).stack

		while not stack.is_empty():
			current_state = stack.pop()
			if current_state is None:
				break

			node = current_state.node
			current_path = current_state.path
			current_depth = current_state.depth

			if isinstance(node, Entity):
				relation_chain: List[Relation] = [item for item in current_path[1:] if isinstance(item, Relation)]
				chain_sig = self._relation_chain_signature(relation_chain)
				logging.info("[CoR][depth=%s] Entity node | chain=%s", current_depth, chain_sig)
				sparql_before = len(sparql_history)
				head_relations, tail_relations = self.relation_search(
					topic_entity=root_entity,
					relation_chain=relation_chain,
					sparql_history=sparql_history,
				)
				logging.info(
					"[CoR][depth=%s] relation_search done | head=%s tail=%s | chain=%s",
					current_depth,
					len(head_relations),
					len(tail_relations),
					chain_sig,
				)
				for event in sparql_history[sparql_before:]:
					step = self._sparql_event_to_step(event)
					if step is not None:
						step_history.append(step)
				if not head_relations and not tail_relations:
					logging.info("[CoR][depth=%s] no relations, skip | chain=%s", current_depth, chain_sig)
					continue

				prompt_before = len(prompt_history)
				selected_relations = self.relation_prune(
					question=question,
					topic_entity=root_entity,
					relation_chain=relation_chain,
					head_relations=head_relations,
					tail_relations=tail_relations,
					prompt_history=prompt_history,
				)
				for event in prompt_history[prompt_before:]:
					step_history.append(self._llm_event_to_step(event))
				selected_relations = selected_relations[: self.relation_width]
				selected_relations = self._expand_bidirectional_relations(
					selected_relations=selected_relations,
					head_relations=head_relations,
					tail_relations=tail_relations,
				)
				logging.info(
					"[CoR][depth=%s] relation_prune selected=%s | chain=%s",
					current_depth,
					[f"{item.id}({'head' if item.left else 'tail'})" for item in selected_relations],
					chain_sig,
				)
				push_relation_children(
					PushRelationChildrenInput(
						stack=stack,
						current_state=current_state,
						selected_relations=selected_relations,
						max_depth=self.depth,
					)
				)
				continue

			if isinstance(node, Relation):
				relation_chain: List[Relation] = [item for item in current_path[1:] if isinstance(item, Relation)]
				chain_sig = self._relation_chain_signature(relation_chain)
				logging.info("[CoR][depth=%s] Relation node | chain=%s", current_depth, chain_sig)
				sparql_before = len(sparql_history)
				reasoning_answers: List[str] = []
				entity_ids, entity_names, literal_candidates = self.entity_search(
					topic_entity=root_entity,
					relation_chain=relation_chain,
					sparql_history=sparql_history,
				)
				logging.info(
					"[CoR][depth=%s] entity_search done | entities=%s literals=%s | chain=%s",
					current_depth,
					len(entity_ids),
					len(literal_candidates),
					chain_sig,
				)
				for event in sparql_history[sparql_before:]:
					step = self._sparql_event_to_step(event)
					if step is not None:
						step_history.append(step)

				if len(entity_ids) + len(literal_candidates) == 0:
					action = "Backtrack" if current_depth >= self.depth else "Forward"
					response = None
					logging.info(
						"[CoR][depth=%s] empty candidates => action=%s | chain=%s",
						current_depth,
						action,
						chain_sig,
					)
				else:
					joint_entity_candidates = entity_names + literal_candidates
					prompt_before = len(prompt_history)
					action, response, reasoning_answers = self.reasoning(
						question=question,
						topic_entity=root_entity,
						relation_chain=relation_chain,
						target_entity_candidates=joint_entity_candidates,
						prompt_history=prompt_history,
					)
					for event in prompt_history[prompt_before:]:
						step_history.append(self._llm_event_to_step(event))
					logging.info(
						"[CoR][depth=%s] reasoning action=%s | candidates=%s | chain=%s",
						current_depth,
						action,
						len(joint_entity_candidates),
						chain_sig,
					)

				if action is None:
					return {
						"action": "ERROR",
						"question": question,
						"results": [],
						"reasoning_chains": "LLM API call failed",
						"prompt_history": prompt_history,
						"sparql_history": sparql_history,
						"step_history": step_history,
					}

				if action.lower() == "stop":
					if reasoning_answers:
						answer = reasoning_answers
						result_items = self._map_answer_names_to_results(
							answer_names=reasoning_answers,
							entity_ids=entity_ids,
							entity_names=entity_names,
						)
					else:
						answer = entity_names + literal_candidates
						result_items = self._build_result_items(
							entity_ids=entity_ids,
							entity_names=entity_names,
							literal_candidates=literal_candidates,
						)
					reasoning_chain_str = self._reasoning_path_to_str(
						topic_entity=root_entity,
						relation_chain=relation_chain,
						target_entity_name_str=", ".join(answer),
					)
					return {
						"action": action,
						"question": question,
						"results": result_items,
						"reasoning_chains": reasoning_chain_str,
						"prompt_history": prompt_history,
						"sparql_history": sparql_history,
						"step_history": step_history,
					}

				if action.lower() == "constraint":
					logging.info("[CoR][depth=%s] enter constraint filter | chain=%s", current_depth, chain_sig)
					joint_entity_candidates = entity_names + literal_candidates
					filter_response = self.validate(
						# filter step is appended immediately after call
						question=question,
						topic_entity=root_entity,
						relation_chain=relation_chain,
						target_entity_names=joint_entity_candidates,
						prompt_history=prompt_history,
					)
					if len(prompt_history) > 0:
						last_event = prompt_history[-1]
						if last_event.get("type") == "filter_answer":
							step_history.append(self._llm_event_to_step(last_event))
					result = self._split_answers(filter_response or "")
					if result:
						logging.info("[CoR][depth=%s] filter accepted answers=%s | chain=%s", current_depth, len(result), chain_sig)
						mapped_results = self._map_answer_names_to_results(
							answer_names=result,
							entity_ids=entity_ids,
							entity_names=entity_names,
						)
						reasoning_chain_str = self._reasoning_path_to_str(
							topic_entity=root_entity,
							relation_chain=relation_chain,
							target_entity_name_str=", ".join(joint_entity_candidates),
						)
						return {
							"action": action,
							"question": question,
							"results": mapped_results,
							"reasoning_chains": reasoning_chain_str,
							"prompt_history": prompt_history,
							"sparql_history": sparql_history,
							"step_history": step_history,
						}
					logging.info("[CoR][depth=%s] filter returned empty, break search | chain=%s", current_depth, chain_sig)
					break

				if action.lower() == "forward":
					logging.info("[CoR][depth=%s] continue forward | chain=%s", current_depth, chain_sig)
					if current_depth >= self.depth:
						logging.info("[CoR][depth=%s] reached max depth, skip forward | chain=%s", current_depth, chain_sig)
						continue
					sparql_before_forward = len(sparql_history)
					head_relations, tail_relations = self.relation_search(
						# relation_search steps are appended immediately after call
						topic_entity=root_entity,
						relation_chain=relation_chain,
						sparql_history=sparql_history,
					)
					for event in sparql_history[sparql_before_forward:]:
						step = self._sparql_event_to_step(event)
						if step is not None:
							step_history.append(step)
					if not head_relations and not tail_relations:
						logging.info("[CoR][depth=%s] forward relation_search empty | chain=%s", current_depth, chain_sig)
						continue

					prompt_before = len(prompt_history)
					selected_relations = self.relation_prune(
						question=question,
						topic_entity=root_entity,
						relation_chain=relation_chain,
						head_relations=head_relations,
						tail_relations=tail_relations,
						prompt_history=prompt_history,
					)
					for event in prompt_history[prompt_before:]:
						step_history.append(self._llm_event_to_step(event))
					selected_relations = selected_relations[: self.relation_width]
					selected_relations = self._expand_bidirectional_relations(
						selected_relations=selected_relations,
						head_relations=head_relations,
						tail_relations=tail_relations,
					)
					push_relation_children(
						PushRelationChildrenInput(
							stack=stack,
							current_state=current_state,
							selected_relations=selected_relations,
							max_depth=self.depth,
						)
					)
					continue

				if action.lower() == "backtrack":
					logging.info("[CoR][depth=%s] backtrack | chain=%s", current_depth, chain_sig)
					continue

				logging.warning(f"Unknown action from reasoning: {action}, raw={response}")
				continue

		prompt_before = len(prompt_history)
		logging.info("[CoR] DFS ended without final answer, fallback to generate_directly")
		generated = self.generate_directly(question=question, prompt_history=prompt_history)
		for event in prompt_history[prompt_before:]:
			step_history.append(self._llm_event_to_step(event))
		results = self._split_answers(generated)
		return {
			"action": "generate_directly",
			"question": question,
			"results": results,
			"reasoning_chains": [],
			"prompt_history": prompt_history,
			"sparql_history": sparql_history,
			"step_history": step_history,
		}


