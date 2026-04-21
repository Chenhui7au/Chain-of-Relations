#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from chain_of_relations.kg_backend.freebase.db_func import (
	abandon_rels,
	build_sparql_constraint_pairs,
	build_sparql_constraints,
	build_sparql_entities,
	build_sparql_relations,
	execurte_sparql,
	execurte_sparql_with_meta,
	id2entity_names as freebase_id2entity_names,
	id2entity_name_or_type,
	replace_entities_prefix,
	replace_relation_prefix,
)


class FreebaseBackend:
	name = "freebase"
	default_entity_prefix: Optional[str] = "m."

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
		return replace_relation_prefix(relations)

	def parse_entities(self, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
		return replace_entities_prefix(entities)

	def id2entity_name_or_type(self, entity_id: str) -> str:
		return id2entity_name_or_type(entity_id)

	def id2entity_names(self, entity_ids: List[str]) -> Dict[str, str]:
		return freebase_id2entity_names(entity_ids)

	def is_unnecessary_relation(self, relation: str) -> bool:
		return abandon_rels(relation)

	def normalize_uri(self, value: str) -> str:
		return str(value).replace("http://rdf.freebase.com/ns/", "").strip()

	def is_entity_id(self, value: str) -> bool:
		text = str(value or "").strip()
		return text.startswith("m.") or text.startswith("g.")

	def relation_id2label(self, relation_id: str) -> str:
		return str(relation_id or "")

	def relation_ids2labels(self, relation_ids: List[str]) -> Dict[str, str]:
		return {str(relation_id): str(relation_id) for relation_id in relation_ids}

	def format_entity_node(self, entity_id: str) -> str:
		return f"ns:{entity_id}"

	def format_relation_node(self, relation_id: str) -> str:
		return f"ns:{relation_id}"
