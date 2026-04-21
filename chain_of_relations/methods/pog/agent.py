#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   agent.py
@Time    :   2026/03/10 23:41:42
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""
  

from pathlib import Path
import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

from chain_of_relations.llm_api import LLMAPI
from chain_of_relations.kg_backend import KGBackend, get_default_backend
from chain_of_relations.schema import Entity, Relation
from chain_of_relations.tools.entity_search import EntitySearchInput, entity_search
from chain_of_relations.tools.generate_directly import GenerateDirectlyInput, generate_directly
from chain_of_relations.tools.relation_prune import RelationPruneInput, relation_prune
from chain_of_relations.tools.relation_search import RelationSearchInput, relation_search
from chain_of_relations.methods.pog.tools.subquestion_decompose import (
	SubquestionDecomposeInput,
	subquestion_decompose,
)
from chain_of_relations.methods.pog.tools.memory_update import MemoryUpdateInput, memory_update
from chain_of_relations.methods.pog.tools.reasoning import PoGReasoningInput, pog_reasoning
from chain_of_relations.methods.pog.tools.entity_condition_prune import (
	EntityConditionPruneInput,
	entity_condition_prune,
	initialize_semantic_model,
)
from chain_of_relations.methods.pog.tools.reverse_retrieval_decider import (
	ReverseRetrievalDeciderInput,
	reverse_retrieval_decider,
)


class PoGAgent:
	def __init__(
		self,
		model_name: str = "gpt-4.1-mini",
		relation_width: int = 3,
		entity_width: int = 3,
		depth: int = 4,
		sample_relation_threshold: int = 500,
		sample_entity_threshold: int = 500,
		remove_unnecessary_rel: bool = True,
		temperature_exploration: float = 0.3,
		temperature_reasoning: float = 0.3,
		max_token: int = 512,
		max_reverse_rounds: int = 5,
		backend: Optional[KGBackend] = None,
	):
		self.model_name = model_name
		self.relation_width = max(1, int(relation_width))
		self.entity_width = max(1, int(entity_width))
		self.width = self.relation_width
		self.depth = max(1, int(depth))
		self.sample_relation_threshold = max(1, int(sample_relation_threshold))
		self.sample_entity_threshold = max(1, int(sample_entity_threshold))
		self.remove_unnecessary_rel = remove_unnecessary_rel
		self.temperature_exploration = temperature_exploration
		self.temperature_reasoning = temperature_reasoning
		self.max_token = max_token
		self.max_reverse_rounds = max(0, int(max_reverse_rounds))
		self.kg_backend = backend or get_default_backend()

		self.llm_api = LLMAPI(model_name=self.model_name)
		self.prompt_file = str(Path(__file__).resolve().parent / "prompt.yml")
		self.prompts = self._load_prompts(self.prompt_file)
		self._trace_seq = 0

		# Initialize sentence-transformer at startup to avoid first-hit lazy-load latency.
		initialize_semantic_model()
		logging.info("[PoG] SentenceTransformer initialized at startup.")

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

	def _get_prompt_block(self, key: str) -> Dict[str, str]:
		aliases = {
			"subquestion_decompose": ["subquestion_decompose"],
			"memory_update": ["memory_update"],
			"pog_reasoning": ["reasoning", "pog_reasoning"],
			"entity_condition_prune": ["entity_condition_prune"],
			"reverse_retrieval_decider": ["reverse_retrieval_decider"],
			"relation_pruning": ["relation_pruning", "extract_relation"],
			"generate_directly": ["generate_directly"],
		}

		for candidate in aliases.get(key, [key]):
			block = self.prompts.get(candidate)
			if isinstance(block, dict):
				return block
		raise KeyError(f"Prompt key '{key}' not found in {self.prompt_file}")

	def _get_user_prompt(self, key: str) -> str:
		block = self._get_prompt_block(key)
		prompt = block.get("user_prompt")
		if not prompt:
			raise KeyError(f"Prompt key '{key}' has no user_prompt")
		return prompt

	def _get_system_prompt(self, key: str) -> Optional[str]:
		return self._get_prompt_block(key).get("system_prompt")

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
		normalized: List[Entity] = []
		for item in topic_entities:
			if isinstance(item, Entity):
				normalized.append(item)
			elif isinstance(item, dict):
				entity_id = str(item.get("id", "")).strip()
				entity_name = str(item.get("name", "")).strip()
				if entity_id and entity_name:
					normalized.append(Entity(id=entity_id, name=entity_name))
		return normalized

	@staticmethod
	def _normalize_name(name: str) -> str:
		return str(name or "").strip().lower()

	@staticmethod
	def _triples_to_text(triples: List[Tuple[str, str, str]]) -> str:
		return "\n".join([f"{h}, {r}, {t}" for h, r, t in triples])

	@staticmethod
	def _dedup_entities(entities: List[Entity]) -> List[Entity]:
		result: List[Entity] = []
		seen = set()
		for ent in entities:
			key = (str(ent.id), str(ent.name))
			if key in seen:
				continue
			seen.add(key)
			result.append(ent)
		return result

	@staticmethod
	def _split_answer_text(answer: str) -> List[str]:
		text = str(answer or "").strip()
		if not text:
			return []
		try:
			parsed = json.loads(text)
			if isinstance(parsed, list):
				return [str(item).strip() for item in parsed if str(item).strip()]
		except Exception:
			pass
		if text.startswith("[") and text.endswith("]"):
			inside = text[1:-1].strip()
			if not inside:
				return []
			items = [item.strip().strip("\"").strip("'") for item in inside.split(",")]
			return [item for item in items if item]
		return [text]

	def _answer_text_to_results(self, answer: str, name_to_entity: Dict[str, Entity]) -> List[Dict[str, str]]:
		results: List[Dict[str, str]] = []
		for item in self._split_answer_text(answer):
			name_key = self._normalize_name(item)
			mapped = name_to_entity.get(name_key)
			if mapped is not None and self._is_mid_like(mapped.id):
				results.append({"id": mapped.id, "name": mapped.name})
			else:
				results.append({"id": "", "name": item})
		return results

	@staticmethod
	def _parse_name_list(text: str) -> List[str]:
		payload = str(text or "").strip()
		if not payload:
			return []
		candidates = [payload]
		candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", payload, flags=re.IGNORECASE))
		for candidate in candidates:
			candidate = candidate.strip()
			if not candidate:
				continue
			try:
				parsed = json.loads(candidate)
				if isinstance(parsed, list):
					return [str(item).strip() for item in parsed if str(item).strip()]
			except Exception:
				pass
			try:
				parsed = ast.literal_eval(candidate)
				if isinstance(parsed, list):
					return [str(item).strip() for item in parsed if str(item).strip()]
			except Exception:
				pass
		if payload.startswith("[") and payload.endswith("]"):
			items = [item.strip().strip("\"").strip("'") for item in payload[1:-1].split(",")]
			return [item for item in items if item]
		return []

	@staticmethod
	def _build_knowledge_triplets_text(
		ent_rel_ent_dict: Dict[str, Dict[str, Dict[str, List[str]]]],
		id_to_entity: Dict[str, Entity],
	) -> str:
		lines: List[str] = []
		for topic_id, ht_dict in sorted(ent_rel_ent_dict.items()):
			topic_name = id_to_entity.get(topic_id).name if topic_id in id_to_entity else topic_id
			for _ht_key, rel_dict in sorted(ht_dict.items()):
				for rel_id, ent_ids in sorted(rel_dict.items()):
					ent_names = []
					for ent_id in ent_ids:
						if ent_id in id_to_entity:
							ent_names.append(id_to_entity[ent_id].name)
						else:
							ent_names.append(ent_id)
					lines.append(f"{topic_name}, {rel_id}, {ent_names}")
		return "\n".join(lines).strip()

	def _is_mid_like(self, value: str) -> bool:
		return self.kg_backend.is_entity_id(value)

	@staticmethod
	def _is_numeric_literal(value: str) -> bool:
		text = str(value or "").strip()
		if not text:
			return False
		return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", text))

	def _record_prompt_history(
		self,
		prompt_history: List[Dict[str, Any]],
		type_name: str,
		prompt: str,
		response: Optional[str],
		parsed_result: Any,
		usage: Dict[str, Any],
		candidate_size: int = -1,
		filtered_candidate_size: int = -1,
		warnings: Optional[List[str]] = None,
	) -> None:
		prompt_history.append(
			self._with_trace(
				{
					"type": type_name,
					"prompt": prompt,
					"response": response,
					"parsed_result": parsed_result,
					"candidate_size": candidate_size,
					"filtered_candidate_size": filtered_candidate_size,
					"input_tokens": usage.get("input_tokens", -1),
					"output_tokens": usage.get("output_tokens", -1),
					"warnings": warnings or [],
				}
			)
		)

	def _record_sparql_relation_search(self, sparql_history: List[Dict[str, Any]], out: Any) -> None:
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

	def _record_sparql_entity_search(self, sparql_history: List[Dict[str, Any]], out: Any) -> None:
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

	@staticmethod
	def _find_pre_info_for_entity(
		entity_id: str,
		depth_ent_rel_ent_dict: Dict[int, Dict[str, Dict[str, Dict[str, List[str]]]]],
	) -> Tuple[str, Optional[bool]]:
		for _depth, ent_rel_ent_dict in sorted(depth_ent_rel_ent_dict.items()):
			for _topic_id, ht_dict in sorted(ent_rel_ent_dict.items()):
				for ht_key, rel_dict in sorted(ht_dict.items()):
					head = ht_key == "head"
					for rel_id, ent_ids in sorted(rel_dict.items()):
						if entity_id in ent_ids:
							return str(rel_id), head
		return "", None

	def _generate_directly(self, question: str, prompt_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
		self._record_prompt_history(
			prompt_history=prompt_history,
			type_name="generate_directly",
			prompt=out.prompt,
			response=out.response,
			parsed_result=out.result,
			usage=out.usage,
			warnings=out.warnings,
		)
		answer = str(out.result or "").strip()
		if not answer:
			return []
		return [{"id": "", "name": answer}]

	def answer(self, question: str, topic_entities: List[Entity], **kwargs) -> Dict[str, Any]:
		normalized_topic_entities = self._normalize_topic_entities(topic_entities)

		prompt_history: List[Dict[str, Any]] = kwargs.get("prompt_history", [])
		sparql_history: List[Dict[str, Any]] = kwargs.get("sparql_history", [])
		step_history: List[Dict[str, Any]] = kwargs.get("step_history", [])
		self._reset_trace()

		if not normalized_topic_entities:
			logging.info("[PoG] no topic entities, fallback to generate_directly")
			results = self._generate_directly(question, prompt_history)
			return {
				"action": "generate_directly",
				"question": question,
				"results": results,
				"reasoning_chains": "",
				"prompt_history": prompt_history,
				"sparql_history": sparql_history,
				"step_history": step_history,
			}

		subq_out = subquestion_decompose(
			SubquestionDecomposeInput(
				question=question,
				user_prompt=self._get_user_prompt("subquestion_decompose"),
				system_prompt=self._get_system_prompt("subquestion_decompose"),
				temperature=self.temperature_reasoning,
				max_tokens=self.max_token,
			),
			llm_generate=self._llm_generate,
		)
		self._record_prompt_history(
			prompt_history=prompt_history,
			type_name="subquestion_decompose",
			prompt=subq_out.prompt,
			response=subq_out.response,
			parsed_result=subq_out.subquestions,
			usage=subq_out.usage,
			warnings=subq_out.warnings,
		)
		subquestions = subq_out.subquestions if subq_out.subquestions else [question]

		name_to_entity: Dict[str, Entity] = {}
		id_to_entity: Dict[str, Entity] = {}
		for ent in normalized_topic_entities:
			name_to_entity[self._normalize_name(ent.name)] = ent
			id_to_entity[ent.id] = ent

		frontier_entities = list(normalized_topic_entities)
		memory_text = "{}"
		reverse_round = 0
		all_reasoning_triples: List[Tuple[str, str, str]] = []
		cluster_chain_of_entities: List[List[Tuple[str, str, str]]] = []
		depth_ent_rel_ent_dict: Dict[int, Dict[str, Dict[str, Dict[str, List[str]]]]] = {}
		all_candidate_entity_names = set(ent.name for ent in frontier_entities)
		pre_relations: List[str] = []
		pre_heads: List[Optional[bool]] = [None] * len(frontier_entities)

		for current_depth in range(1, self.depth + 1):
			logging.info("[PoG][depth=%s] start | frontier_size=%s", current_depth, len(frontier_entities))
			candidate_groups: Dict[str, Dict[str, Any]] = {}
			candidate_meta_by_entity_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
			current_depth_ent_rel_ent_dict: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
			group_index = 0

			for idx, topic_entity in enumerate(frontier_entities):
				pre_head = pre_heads[idx] if idx < len(pre_heads) else None
				relation_out = relation_search(
					RelationSearchInput(
						topic_entity=topic_entity,
						relation_chain=[],
						remove_unnecessary_rel=self.remove_unnecessary_rel,
						prune_inverse_of_last_hop=False,
					),
					backend=self.kg_backend,
				)
				self._record_sparql_relation_search(sparql_history, relation_out)

				head_relations = list(relation_out.head_relations)
				tail_relations = list(relation_out.tail_relations)
				if pre_relations:
					pre_relation_set = set(pre_relations)
					if pre_head is True:
						tail_relations = [r for r in tail_relations if r not in pre_relation_set]
					elif pre_head is False:
						head_relations = [r for r in head_relations if r not in pre_relation_set]

				relation_display_names = self.kg_backend.relation_ids2labels(head_relations + tail_relations)
				relation_prune_question = question if not subquestions else f"{question}\nSubobjectives: {subquestions}"

				prune_out = relation_prune(
					RelationPruneInput(
						question=relation_prune_question,
						topic_entity=topic_entity,
						relation_chain=[],
						head_relations=head_relations,
						tail_relations=tail_relations,
						relation_display_names=relation_display_names,
						user_prompt=self._get_user_prompt("relation_pruning"),
						top_k=self.relation_width,
						sample_relation_threshold=self.sample_relation_threshold,
						system_prompt=self._get_system_prompt("relation_pruning"),
						render_mode="unified",
						temperature=self.temperature_exploration,
						max_tokens=self.max_token,
					),
					llm_generate=self._llm_generate,
				)
				self._record_prompt_history(
					prompt_history=prompt_history,
					type_name="relation_prune",
					prompt=prune_out.prompt,
					response=prune_out.response,
					parsed_result=[
						{"id": rel.id, "left": rel.left, "score": rel.score}
						for rel in prune_out.selected_relations
					],
					usage=prune_out.usage,
					candidate_size=prune_out.candidate_size,
					filtered_candidate_size=prune_out.pruned_candidate_size,
					warnings=prune_out.warnings,
				)
				if not prune_out.success:
					continue

				for selected_relation in prune_out.selected_relations[: self.relation_width]:
					entity_out = entity_search(
						EntitySearchInput(
							topic_entity=topic_entity,
							relation_chain=[selected_relation],
							filter_entity_prefix=self.kg_backend.default_entity_prefix,
							drop_unnamed_entity=True,
						),
						backend=self.kg_backend,
					)
					self._record_sparql_entity_search(sparql_history, entity_out)

					entities_for_group: List[Entity] = []
					for ent_id, ent_name in zip(entity_out.entity_ids, entity_out.entity_names):
						ent = Entity(id=ent_id, name=ent_name)
						entities_for_group.append(ent)
						name_to_entity[self._normalize_name(ent_name)] = ent
						id_to_entity[ent_id] = ent
					for literal in entity_out.literal_entities:
						literal_text = str(literal).strip()
						if not literal_text:
							continue
						literal_ent = Entity(id=literal_text, name=literal_text, type="literal")
						entities_for_group.append(literal_ent)
						name_to_entity[self._normalize_name(literal_text)] = literal_ent
						id_to_entity[literal_text] = literal_ent

					if not entities_for_group:
						continue

					topic_bucket = current_depth_ent_rel_ent_dict.setdefault(topic_entity.id, {})
					ht_key = "head" if bool(selected_relation.left) else "tail"
					rel_bucket = topic_bucket.setdefault(ht_key, {})
					ent_bucket = rel_bucket.setdefault(selected_relation.id, [])
					for ent in entities_for_group:
						if ent.id not in ent_bucket:
							ent_bucket.append(ent.id)
						all_candidate_entity_names.add(ent.name)

					group_index += 1
					group_id = str(group_index)
					candidate_groups[group_id] = {
						"relation": selected_relation.id,
						"head": bool(selected_relation.left),
						"entities": entities_for_group,
						"topic_entity": topic_entity,
					}

					for ent in entities_for_group:
						key = (ent.id, ent.name)
						if key in candidate_meta_by_entity_key:
							continue
						candidate_meta_by_entity_key[key] = {
							"topic_entity": topic_entity,
							"relation": selected_relation,
							"head": bool(selected_relation.left),
						}

			candidate_entity_size = sum(len(group.get("entities", [])) for group in candidate_groups.values())
			logging.info(
				"[PoG][depth=%s] retrieve_done | candidate_groups=%s | candidate_entities=%s",
				current_depth,
				len(candidate_groups),
				candidate_entity_size,
			)
			if not candidate_groups:
				logging.info("[PoG][depth=%s] no new knowledge, fallback to generate_directly", current_depth)
				fallback_results = self._generate_directly(question, prompt_history)
				return {
					"action": "generate_directly",
					"question": question,
					"results": fallback_results,
					"reasoning_chains": "",
					"prompt_history": prompt_history,
					"sparql_history": sparql_history,
					"step_history": step_history,
				}

			depth_ent_rel_ent_dict[current_depth] = current_depth_ent_rel_ent_dict

			current_subquestion = subquestions[min(current_depth - 1, len(subquestions) - 1)]

			selected_entities: List[Entity] = []
			selected_scores: List[float] = []
			selected_pre_relations: List[str] = []
			selected_pre_heads: List[Optional[bool]] = []
			new_ent_rel_ent_dict: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
			depth_chain: List[Tuple[str, str, str]] = []

			prune_out = entity_condition_prune(
				EntityConditionPruneInput(
					question=question,
					current_question=current_subquestion,
					topic_entity_name="",
					condition=current_subquestion,
					candidate_groups=candidate_groups,
					user_prompt=self._get_user_prompt("entity_condition_prune"),
					entity_width=self.entity_width,
					sample_entity_threshold=self.sample_entity_threshold,
					system_prompt=self._get_system_prompt("entity_condition_prune"),
					temperature=self.temperature_reasoning,
					max_tokens=self.max_token,
				),
				llm_generate=self._llm_generate,
			)
			self._record_prompt_history(
				prompt_history=prompt_history,
				type_name="entity_condition_prune",
				prompt=prune_out.prompt,
				response=prune_out.response,
				parsed_result=[
					{"id": ent.id, "name": ent.name, "score": score}
					for ent, score in zip(prune_out.pruned_entities, prune_out.pruned_scores)
				],
				usage=prune_out.usage,
				candidate_size=sum(len(group.get("entities", [])) for group in candidate_groups.values()),
				filtered_candidate_size=len(prune_out.pruned_entities),
				warnings=prune_out.warnings,
			)

			for ent, score in zip(prune_out.pruned_entities, prune_out.pruned_scores):
				selected_entities.append(Entity(id=ent.id, name=ent.name, score=score, type=ent.type))
				selected_scores.append(float(score))

				meta = candidate_meta_by_entity_key.get((ent.id, ent.name))
				if meta is None:
					continue
				topic_ent = meta["topic_entity"]
				relation = meta["relation"]
				head = bool(meta["head"])
				selected_pre_relations.append(str(relation.id))
				selected_pre_heads.append(head)
				depth_chain.append((topic_ent.name, relation.id, ent.name))

				topic_bucket = new_ent_rel_ent_dict.setdefault(topic_ent.id, {})
				ht_key = "head" if head else "tail"
				rel_bucket = topic_bucket.setdefault(ht_key, {})
				ent_list = rel_bucket.setdefault(relation.id, [])
				if ent.id not in ent_list:
					ent_list.append(ent.id)

			logging.info(
				"[PoG][depth=%s] prune_done | selected_entities=%s",
				current_depth,
				len(selected_entities),
			)
			if not selected_entities:
				logging.info("[PoG][depth=%s] prune empty, fallback to generate_directly", current_depth)
				fallback_results = self._generate_directly(question, prompt_history)
				return {
					"action": "generate_directly",
					"question": question,
					"results": fallback_results,
					"reasoning_chains": "",
					"prompt_history": prompt_history,
					"sparql_history": sparql_history,
					"step_history": step_history,
				}

			selected_entities = self._dedup_entities(selected_entities)
			if depth_chain:
				cluster_chain_of_entities.append(depth_chain)
				all_reasoning_triples.extend(depth_chain)

			knowledge_triplets_text = ""
			for topic_id, ht_dict in sorted(new_ent_rel_ent_dict.items()):
				topic_name = id_to_entity.get(topic_id).name if topic_id in id_to_entity else topic_id
				for ht_key, rel_dict in sorted(ht_dict.items()):
					for rel_id, ent_ids in sorted(rel_dict.items()):
						ent_names = []
						for ent_id in ent_ids:
							if ent_id in id_to_entity:
								ent_names.append(id_to_entity[ent_id].name)
							else:
								ent_names.append(ent_id)
						knowledge_triplets_text += f"{topic_name}, {rel_id}, {ent_names}\n"

			mem_out = memory_update(
				MemoryUpdateInput(
					question=question,
					subquestions=subquestions,
					memory=memory_text,
					knowledge_triplets=knowledge_triplets_text.strip(),
					user_prompt=self._get_user_prompt("memory_update"),
					system_prompt=self._get_system_prompt("memory_update"),
					temperature=self.temperature_reasoning,
					max_tokens=max(self.max_token, 1024),
				),
				llm_generate=self._llm_generate,
			)
			self._record_prompt_history(
				prompt_history=prompt_history,
				type_name="memory_update",
				prompt=mem_out.prompt,
				response=mem_out.response,
				parsed_result=mem_out.memory_obj,
				usage=mem_out.usage,
				warnings=mem_out.warnings,
			)
			if mem_out.success and mem_out.updated_memory:
				memory_text = mem_out.updated_memory

			candidate_name_text = str([ent.name for ent in selected_entities])
			reasoning_out = pog_reasoning(
				PoGReasoningInput(
					question=question,
					entity_candidates=candidate_name_text,
					memory=memory_text,
					knowledge_triplets=knowledge_triplets_text.strip(),
					user_prompt=self._get_user_prompt("pog_reasoning"),
					system_prompt=self._get_system_prompt("pog_reasoning"),
					temperature=self.temperature_reasoning,
					max_tokens=max(self.max_token, 1024),
				),
				llm_generate=self._llm_generate,
			)
			self._record_prompt_history(
				prompt_history=prompt_history,
				type_name="pog_reasoning",
				prompt=reasoning_out.prompt,
				response=reasoning_out.response,
				parsed_result=reasoning_out.raw_decision,
				usage=reasoning_out.usage,
				candidate_size=len(selected_entities),
				filtered_candidate_size=1 if reasoning_out.sufficient else 0,
				warnings=reasoning_out.warnings,
			)

			answer_text = str(reasoning_out.answer or "").strip()
			answer_lower = answer_text.lower()
			stop = bool(reasoning_out.sufficient)
			if not answer_text:
				stop = False
			if answer_lower in {"null", "none"}:
				stop = False
			if self.kg_backend.is_entity_id(answer_text):
				stop = False
			logging.info(
				"[PoG][depth=%s] reasoning | sufficient=%s | stop=%s | answer=%s",
				current_depth,
				reasoning_out.sufficient,
				stop,
				(answer_text[:120] + "...") if len(answer_text) > 120 else answer_text,
			)

			if stop:
				logging.info("[PoG][depth=%s] stop with answer", current_depth)
				results = self._answer_text_to_results(answer_text, name_to_entity)
				if not results:
					results = [{"id": ent.id if self._is_mid_like(ent.id) else "", "name": ent.name} for ent in selected_entities]
				if not results:
					results = self._generate_directly(question, prompt_history)
				return {
					"action": "stop",
					"question": question,
					"results": results,
					"reasoning_chains": str(cluster_chain_of_entities),
					"prompt_history": prompt_history,
					"sparql_history": sparql_history,
					"step_history": step_history,
				}

			head_entities_text = str(sorted({group["topic_entity"].name for group in candidate_groups.values()}))
			tail_entities_text = str(sorted({ent.name for group in candidate_groups.values() for ent in group.get("entities", [])}))
			relation_text = "; ".join(sorted({str(group.get("relation", "")) for group in candidate_groups.values()}))
			current_entity_name_set = {self._normalize_name(ent.name) for ent in selected_entities if ent.name}
			filtered_candidate_entity_names = sorted(
				name
				for name in all_candidate_entity_names
				if self._normalize_name(name) not in current_entity_name_set
			)
			reverse_out = reverse_retrieval_decider(
				ReverseRetrievalDeciderInput(
					question=question,
					relation=relation_text,
					head_entities_text=head_entities_text,
					tail_entities_text=tail_entities_text,
					current_entities_text=str(sorted({ent.name for ent in selected_entities})),
					candidate_entities_text=str(filtered_candidate_entity_names),
					memory=memory_text,
					knowledge_triplets=self._triples_to_text(all_reasoning_triples),
					user_prompt=self._get_user_prompt("reverse_retrieval_decider"),
					system_prompt=self._get_system_prompt("reverse_retrieval_decider"),
					temperature=self.temperature_reasoning,
					max_tokens=self.max_token,
				),
				llm_generate=self._llm_generate,
			)
			logging.info(
				"[PoG][depth=%s] reverse_decision | need_reverse=%s | reverse_entities=%s",
				current_depth,
				reverse_out.need_reverse,
				len(reverse_out.reverse_entity_names),
			)
			self._record_prompt_history(
				prompt_history=prompt_history,
				type_name="reverse_retrieval_decider",
				prompt=reverse_out.prompt,
				response=reverse_out.response,
				parsed_result=reverse_out.raw_decision,
				usage=reverse_out.usage,
				warnings=reverse_out.warnings,
			)

			next_frontier: List[Entity] = list(selected_entities)
			next_pre_relations: List[str] = list(selected_pre_relations)
			next_pre_heads: List[Optional[bool]] = list(selected_pre_heads)

			if reverse_out.need_reverse and reverse_round < self.max_reverse_rounds:
				reverse_round += 1
				selector_prompt_template = self._get_user_prompt("reverse_entity_selector")
				selector_system_prompt = self._get_system_prompt("reverse_entity_selector")
				selector_prompt = (
					selector_prompt_template
					.replace("{{question}}", question)
					.replace("{{reason}}", reverse_out.reason or "")
					.replace("{{candidate_entities}}", str(filtered_candidate_entity_names))
					.replace("{{memory}}", memory_text)
				)
				selector_response, selector_usage = self._llm_generate(
					user_prompt=selector_prompt,
					temperature=self.temperature_reasoning,
					max_tokens=self.max_token,
					system_prompt=selector_system_prompt,
				)
				selector_names = self._parse_name_list(selector_response)
				self._record_prompt_history(
					prompt_history=prompt_history,
					type_name="reverse_entity_selector",
					prompt=selector_prompt,
					response=selector_response,
					parsed_result=selector_names,
					usage=selector_usage,
					warnings=[],
				)

				for name in selector_names:
					ent = name_to_entity.get(self._normalize_name(name))
					if ent is None:
						continue
					if any(existing.id == ent.id and existing.name == ent.name for existing in next_frontier):
						continue
					next_frontier.append(ent)
					add_rel, add_head = self._find_pre_info_for_entity(ent.id, depth_ent_rel_ent_dict)
					next_pre_relations.append(add_rel)
					next_pre_heads.append(add_head)

			if reasoning_out.retrieve_entity:
				retrieve_ent = name_to_entity.get(self._normalize_name(reasoning_out.retrieve_entity))
				if retrieve_ent is not None and not any(existing.id == retrieve_ent.id and existing.name == retrieve_ent.name for existing in next_frontier):
					next_frontier.append(retrieve_ent)
					next_pre_relations.append("")
					next_pre_heads.append(None)

			if not next_frontier:
				logging.info("[PoG][depth=%s] empty next frontier, break", current_depth)
				break

			frontier_entities = self._dedup_entities(next_frontier)
			pre_relations = list(next_pre_relations[: len(frontier_entities)])
			pre_heads = list(next_pre_heads[: len(frontier_entities)])
			if len(pre_heads) < len(frontier_entities):
				pre_heads.extend([None] * (len(frontier_entities) - len(pre_heads)))
			logging.info(
				"[PoG][depth=%s] end | next_frontier_size=%s | reverse_round=%s",
				current_depth,
				len(frontier_entities),
				reverse_round,
			)

		logging.info("[PoG] reached max depth or no stop, fallback to generate_directly")
		fallback_results = self._generate_directly(question, prompt_history)
		if not fallback_results:
			fallback_results = [{"id": ent.id if self._is_mid_like(ent.id) else "", "name": ent.name} for ent in frontier_entities if ent.name]
		return {
			"action": "generate_directly",
			"question": question,
			"results": fallback_results,
			"reasoning_chains": "",
			"prompt_history": prompt_history,
			"sparql_history": sparql_history,
			"step_history": step_history,
		}
