#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   entity_search.py
@Time    :   2026/03/06 18:34:25
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import logging


PROJECT_DIR = Path(__file__).resolve().parent.parent

from chain_of_relations.kg_backend import KGBackend, get_default_backend
from chain_of_relations.schema import Entity, Relation


@dataclass
class EntitySearchInput:
	topic_entity: Entity
	relation_chain: List[Relation] = field(default_factory=list)
	filter_entity_prefix: Optional[str] = "__AUTO__"
	drop_unnamed_entity: bool = True


@dataclass
class EntitySearchOutput:
	entity_ids: List[str]
	entity_names: List[str]
	literal_entities: List[str]
	query: str
	query_status: str
	timed_out: bool
	filter_caused_empty: bool
	entity_count: int
	filtered_entity_count: int
	warnings: List[str] = field(default_factory=list)


def _extract_inline_entity_labels(rows: List[Dict], backend: KGBackend) -> Dict[str, str]:
	label_map: Dict[str, str] = {}
	for row in rows:
		entity_binding = row.get("targetEntity", {})
		if entity_binding.get("type", "") != "uri":
			continue
		entity_uri = str(entity_binding.get("value", "")).strip()
		entity_id = backend.normalize_uri(entity_uri)
		if not entity_id or entity_id in label_map:
			continue
		label = str(row.get("targetEntityLabel", {}).get("value", "")).strip()
		if label:
			label_map[entity_id] = label
	return label_map


def _build_entities_extract(topic_entity: Entity, relation_chain: List[Relation], backend: KGBackend) -> Tuple[str, str]:
	if not relation_chain:
		return "", "relation_chain is empty"

	sparql_entities_extract = ""
	last_entity = backend.format_entity_node(topic_entity.id)

	for index, relation in enumerate(relation_chain, start=1):
		is_last = index == len(relation_chain)
		if is_last:
			next_entity = "?targetEntity"
		else:
			next_entity = f"?e_{index}"

		relation_node = backend.format_relation_node(relation.id)

		if relation.left:
			sparql_entities_extract += f"{last_entity} {relation_node} {next_entity} .\n"
		else:
			sparql_entities_extract += f"{next_entity} {relation_node} {last_entity} .\n"

		if not is_last:
			last_entity = next_entity

	return sparql_entities_extract.strip(), ""


def entity_search(
	inp: EntitySearchInput,
	query_executor: Optional[Callable[[str], List[Dict]]] = None,
	name_resolver: Optional[Callable[[str], str]] = None,
	backend: Optional[KGBackend] = None,
) -> EntitySearchOutput:
	warnings: List[str] = []
	resolved_backend = backend or get_default_backend()
	resolved_name_resolver = name_resolver or resolved_backend.id2entity_name_or_type

	sparql_entities_extract, err = _build_entities_extract(
		topic_entity=inp.topic_entity,
		relation_chain=inp.relation_chain,
		backend=resolved_backend,
	)
	if err:
		warnings.append(err)
		return EntitySearchOutput(
			entity_ids=[],
			entity_names=[],
			literal_entities=[],
			query="",
			query_status="error",
			timed_out=False,
			filter_caused_empty=False,
			entity_count=0,
			filtered_entity_count=0,
			warnings=warnings,
		)

	query = resolved_backend.build_sparql_entities(sparql_entities_extract)
	if query_executor is None:
		query_meta = resolved_backend.execute_sparql_with_meta(query)
		entities = query_meta.get("rows", []) or []
		query_status = str(query_meta.get("status", "ok"))
		timed_out = bool(query_meta.get("timed_out", False))
	else:
		entities = query_executor(query) or []
		query_status = "ok"
		timed_out = False
	raw_result_count = len(entities)

	if entities:
		literal_entity, entity_ids = resolved_backend.parse_entities(entities)
	else:
		literal_entity, entity_ids = [], []

	inline_label_map = _extract_inline_entity_labels(entities, resolved_backend) if entities else {}

	effective_prefix = inp.filter_entity_prefix
	if effective_prefix == "__AUTO__":
		effective_prefix = resolved_backend.default_entity_prefix
	if effective_prefix is not None:
		entity_ids = [entity for entity in entity_ids if entity.startswith(effective_prefix)]

	unique_entity_ids = list(dict.fromkeys(entity_ids))

	batch_name_map: Dict[str, str] = {}
	missing_name_ids = [entity_id for entity_id in unique_entity_ids if entity_id not in inline_label_map]
	if missing_name_ids:
		logging.info(
			"[EntitySearch] need resolve names | total_entities=%s",
			len(missing_name_ids),
		)
		try:
			batch_name_map = resolved_backend.id2entity_names(missing_name_ids)
		except Exception:
			batch_name_map = {}

	entity_names = [
		str(inline_label_map.get(entity_id, "")).strip()
		or str(batch_name_map.get(entity_id, "")).strip()
		or resolved_name_resolver(entity_id)
		for entity_id in entity_ids
	]

	if inp.drop_unnamed_entity:
		filtered_entity_ids: List[str] = []
		filtered_entity_names: List[str] = []
		for entity_id, entity_name in zip(entity_ids, entity_names):
			if entity_name != "UnName_Entity":
				filtered_entity_ids.append(entity_id)
				filtered_entity_names.append(entity_name)
		entity_ids, entity_names = filtered_entity_ids, filtered_entity_names

	final_candidate_count = len(entity_ids) + len(literal_entity)
	filter_caused_empty = raw_result_count > 0 and final_candidate_count == 0

	if not entity_ids and not literal_entity:
		if timed_out:
			warnings.append("No entities found due to SPARQL timeout.")
		elif filter_caused_empty:
			warnings.append("No entities remained after filtering.")
		else:
			warnings.append("No entities found from current relation_chain.")

	return EntitySearchOutput(
		entity_ids=entity_ids,
		entity_names=entity_names,
		literal_entities=literal_entity,
		query=query,
		query_status=query_status,
		timed_out=timed_out,
		filter_caused_empty=filter_caused_empty,
		entity_count=raw_result_count,
		filtered_entity_count=len(entity_ids),
		warnings=warnings,
	)
