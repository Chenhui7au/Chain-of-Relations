#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   relation_search.py
@Time    :   2026/04/03 08:09:32
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import re

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

from chain_of_relations.kg_backend import KGBackend, get_default_backend
from chain_of_relations.schema import Entity, Relation


_RELATION_LABEL_NOISE_PATTERN = re.compile(
	r"(^|\b)(id|identifier|code|url|username|handle|catalog|index|registry|authority|vocabulary|thesaurus|api endpoint|image|logo|flag image|coat of arms|official website)(\b|$)",
	re.IGNORECASE,
)


@dataclass
class RelationSearchInput:
	topic_entity: Entity
	relation_chain: List[Relation] = field(default_factory=list)
	remove_unnecessary_rel: bool = True
	prune_inverse_of_last_hop: bool = True


@dataclass
class RelationSearchOutput:
	head_relations: List[str]
	tail_relations: List[str]
	head_query: str
	tail_query: str
	head_status: str
	tail_status: str
	head_timed_out: bool
	tail_timed_out: bool
	raw_head_count: int
	raw_tail_count: int
	final_head_count: int
	final_tail_count: int
	warnings: List[str] = field(default_factory=list)


def _build_relation_extract(topic_entity_id: str, relation_chain: List[Relation], backend: KGBackend) -> str:
	if not relation_chain:
		return "", backend.format_entity_node(topic_entity_id)

	relation_extract = ""
	last_entity = backend.format_entity_node(topic_entity_id)
	for index, hop in enumerate(relation_chain, start=1):
		relation_node = backend.format_relation_node(hop.id)
		if hop.left:
			relation_extract += f"{last_entity} {relation_node} ?e_{index} .\n"
		else:
			relation_extract += f"?e_{index} {relation_node} {last_entity} .\n"
		last_entity = f"?e_{index}"
	return relation_extract, last_entity


def _uniq_sorted(values: List[str]) -> List[str]:
	return sorted(list(set(values)))


def _is_noisy_relation_label(label: str) -> bool:
	text = str(label or "").strip().lower()
	if not text:
		return False
	return bool(_RELATION_LABEL_NOISE_PATTERN.search(text))


def _filter_relations_by_label(relations: List[str], backend: KGBackend) -> List[str]:
	if not relations:
		return relations
	if getattr(backend, "name", "") != "wikidata":
		return relations

	label_map = backend.relation_ids2labels(relations)
	filtered = [relation for relation in relations if not _is_noisy_relation_label(label_map.get(relation, relation))]
	return filtered or relations


def relation_search(
	inp: RelationSearchInput,
	query_executor: Optional[Callable[[str], List[Dict]]] = None,
	backend: Optional[KGBackend] = None,
) -> RelationSearchOutput:
	warnings: List[str] = []
	resolved_backend = backend or get_default_backend()

	relation_extract, last_entity = _build_relation_extract(
		topic_entity_id=inp.topic_entity.id,
		relation_chain=inp.relation_chain,
		backend=resolved_backend,
	)
	head_extract = relation_extract + f"{last_entity} ?relation ?x ."
	tail_extract = relation_extract + f"?x ?relation {last_entity} ."

	head_query = resolved_backend.build_sparql_relations(head_extract)
	tail_query = resolved_backend.build_sparql_relations(tail_extract)

	if query_executor is None:
		head_meta = resolved_backend.execute_sparql_with_meta(head_query)
		tail_meta = resolved_backend.execute_sparql_with_meta(tail_query)
		raw_head_rows = head_meta.get("rows", []) or []
		raw_tail_rows = tail_meta.get("rows", []) or []
		head_status = str(head_meta.get("status", "ok"))
		tail_status = str(tail_meta.get("status", "ok"))
		head_timed_out = bool(head_meta.get("timed_out", False))
		tail_timed_out = bool(tail_meta.get("timed_out", False))
	else:
		raw_head_rows = query_executor(head_query) or []
		raw_tail_rows = query_executor(tail_query) or []
		head_status = "ok"
		tail_status = "ok"
		head_timed_out = False
		tail_timed_out = False

	head_relations = resolved_backend.parse_relations(raw_head_rows) if raw_head_rows else []
	tail_relations = resolved_backend.parse_relations(raw_tail_rows) if raw_tail_rows else []

	raw_head_count = len(head_relations)
	raw_tail_count = len(tail_relations)

	if inp.remove_unnecessary_rel:
		head_relations = [relation for relation in head_relations if not resolved_backend.is_unnecessary_relation(relation)]
		tail_relations = [relation for relation in tail_relations if not resolved_backend.is_unnecessary_relation(relation)]

	head_relations = _filter_relations_by_label(head_relations, resolved_backend)
	tail_relations = _filter_relations_by_label(tail_relations, resolved_backend)

	if inp.prune_inverse_of_last_hop and inp.relation_chain:
		previous_relation = inp.relation_chain[-1]
		if previous_relation.left:
			tail_relations = [relation for relation in tail_relations if relation != previous_relation.id]
		else:
			head_relations = [relation for relation in head_relations if relation != previous_relation.id]

	head_relations = _uniq_sorted(head_relations)
	tail_relations = _uniq_sorted(tail_relations)

	if not head_relations and not tail_relations:
		if head_timed_out or tail_timed_out:
			warnings.append("No relation candidates found due to SPARQL timeout.")
		else:
			warnings.append("No relation candidates found.")

	return RelationSearchOutput(
		head_relations=head_relations,
		tail_relations=tail_relations,
		head_query=head_query,
		tail_query=tail_query,
		head_status=head_status,
		tail_status=tail_status,
		head_timed_out=head_timed_out,
		tail_timed_out=tail_timed_out,
		raw_head_count=raw_head_count,
		raw_tail_count=raw_tail_count,
		final_head_count=len(head_relations),
		final_tail_count=len(tail_relations),
		warnings=warnings,
	)
