#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from chain_of_relations.kg_backend.wikidata.db_func import (
	abandon_rels,
	build_sparql_constraint_pairs,
	build_sparql_constraints,
	build_sparql_entities,
	build_sparql_relations,
	execurte_sparql,
	execurte_sparql_with_meta,
	id2entity_name_or_type,
	id2entity_names,
	relation_ids2labels,
	replace_entities_prefix,
	replace_relation_prefix,
)


class WikidataBackend:
	name = "wikidata"
	default_entity_prefix: Optional[str] = "Q"

	def __init__(self) -> None:
		self._relation_label_cache: Dict[str, str] = {}

	def build_sparql_relations(self, query: str) -> str:
		return build_sparql_relations(query)

	def build_sparql_entities(self, query: str) -> str:
		return build_sparql_entities(query)

	def build_sparql_constraints(self, query: str) -> str:
		return build_sparql_constraints(query)

	def build_sparql_constraint_pairs(self, query: str) -> str:
		return build_sparql_constraint_pairs(query)

	def execute_sparql(self, sparql_txt: str) -> List[Dict[str, Any]]:
		return execurte_sparql(sparql_txt)

	def execute_sparql_with_meta(self, sparql_txt: str) -> Dict[str, Any]:
		return execurte_sparql_with_meta(sparql_txt)

	def parse_relations(self, relations: List[Dict[str, Any]]) -> List[str]:
		for row in relations:
			raw_relation = str(row.get("relation", {}).get("value", "")).strip()
			relation_id = self.normalize_uri(raw_relation)
			if not self._is_property_id(relation_id):
				continue
			label = str(row.get("relationLabel", {}).get("value", "")).strip()
			if label:
				self._relation_label_cache[relation_id] = label
		parsed = replace_relation_prefix(relations)
		return [relation for relation in parsed if self._is_property_id(relation)]

	def parse_entities(self, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
		return replace_entities_prefix(entities)

	def id2entity_name_or_type(self, entity_id: str) -> str:
		return id2entity_name_or_type(entity_id)

	def id2entity_names(self, entity_ids: List[str]) -> Dict[str, str]:
		return id2entity_names(entity_ids)

	def is_unnecessary_relation(self, relation: str) -> bool:
		return abandon_rels(relation)

	def normalize_uri(self, value: str) -> str:
		text = str(value or "").strip()
		if text.startswith("http://www.wikidata.org/entity/"):
			return text.replace("http://www.wikidata.org/entity/", "")
		if text.startswith("http://www.wikidata.org/prop/direct/"):
			return text.replace("http://www.wikidata.org/prop/direct/", "")
		return text

	def is_entity_id(self, value: str) -> bool:
		text = str(value or "").strip()
		return text.startswith("Q") and len(text) > 1 and text[1:].isdigit()

	@staticmethod
	def _is_property_id(value: str) -> bool:
		return bool(re.fullmatch(r"P\d+", str(value or "").strip()))

	@staticmethod
	def _fallback_relation_name(relation_id: str) -> str:
		text = str(relation_id or "").strip()
		if not text:
			return text
		if text.startswith("http://") or text.startswith("https://"):
			if "#" in text:
				return text.split("#")[-1]
			return text.rstrip("/").split("/")[-1]
		return text

	def relation_id2label(self, relation_id: str) -> str:
		text = str(relation_id or "").strip()
		if not text:
			return text
		if text in self._relation_label_cache:
			return self._relation_label_cache[text]
		mapping = self.relation_ids2labels([text])
		return mapping.get(text, self._fallback_relation_name(text))

	def relation_ids2labels(self, relation_ids: List[str]) -> Dict[str, str]:
		result: Dict[str, str] = {}
		to_query: List[str] = []

		for relation_id in relation_ids:
			text = str(relation_id or "").strip()
			if not text:
				continue
			if text in self._relation_label_cache:
				result[text] = self._relation_label_cache[text]
				continue
			if self._is_property_id(text):
				to_query.append(text)
			else:
				fallback = self._fallback_relation_name(text)
				self._relation_label_cache[text] = fallback
				result[text] = fallback

		if to_query:
			chunk_size = 100
			for start in range(0, len(to_query), chunk_size):
				chunk = to_query[start : start + chunk_size]
				label_map = relation_ids2labels(chunk)
				for relation_id in chunk:
					label = str(label_map.get(relation_id, "")).strip()
					if not label:
						label = self._fallback_relation_name(relation_id)
					self._relation_label_cache[relation_id] = label
					result[relation_id] = label

		for relation_id in relation_ids:
			text = str(relation_id or "").strip()
			if not text:
				continue
			if text not in result:
				result[text] = self._relation_label_cache.get(text, self._fallback_relation_name(text))

		return result

	def format_entity_node(self, entity_id: str) -> str:
		if entity_id.startswith("wd:"):
			return entity_id
		return f"wd:{entity_id}"

	def format_relation_node(self, relation_id: str) -> str:
		if relation_id.startswith("wdt:"):
			return relation_id
		if relation_id.startswith("P"):
			return f"wdt:{relation_id}"
		return relation_id
