#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   db_func.py
@Time    :   2026/03/06 16:52:59
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from SPARQLWrapper import SPARQLWrapper, JSON


SPARQLPATH = os.getenv(
	"FREEBASE_SPARQL_ENDPOINT",
	"http://127.0.0.1:8890/sparql",
)  # depend on your own endpoint, can override by FREEBASE_SPARQL_ENDPOINT
SPARQL_TIMEOUT = int(os.getenv("SPARQL_TIMEOUT", "30"))
SPARQL_RELATION_LIMIT = 1000
SPARQL_ENTITY_LIMIT = 1000
SPARQL_ID_LIMIT = 1
SPARQL_LABEL_BATCH_SIZE = int(os.getenv("FREEBASE_LABEL_BATCH_SIZE", "100"))
_FREEBASE_URI_PREFIX = "http://rdf.freebase.com/ns/"
_ID2NAME_CACHE: Dict[str, str] = {}


def _load_query_templates():
	query_template_path = Path(__file__).resolve().parent / "query_template.yml"
	with open(query_template_path, "r", encoding="utf-8") as f:
		templates = yaml.safe_load(f) or {}

	required_keys = {
		"sparql_relations",
		"sparql_entities",
		"sparql_constraints",
		"sparql_constraint_pairs",
		"sparql_id2name",
	}
	missing_keys = [key for key in required_keys if key not in templates]
	if missing_keys:
		raise ValueError(
			f"Missing query templates in {query_template_path}: {missing_keys}"
		)
	return templates


_QUERY_TEMPLATES = _load_query_templates()

sparql_relations = _QUERY_TEMPLATES["sparql_relations"]
sparql_entities = _QUERY_TEMPLATES["sparql_entities"]
sparql_constraints = _QUERY_TEMPLATES["sparql_constraints"]
sparql_constraint_pairs = _QUERY_TEMPLATES["sparql_constraint_pairs"]
sparql_id2name = _QUERY_TEMPLATES["sparql_id2name"]


def build_sparql_relations(query):
	return sparql_relations.replace("{{query}}", query)


def build_sparql_entities(query):
	return sparql_entities.replace("{{query}}", query)


def build_sparql_constraints(query):
	return sparql_constraints.replace("{{query}}", query)


def build_sparql_constraint_pairs(query):
	return sparql_constraint_pairs.replace("{{query}}", query)


def _render_id2name_query(entity_id):
	template = sparql_id2name.replace("{{entity_id}}", entity_id)
	count = template.count("%s")
	if count == 0:
		return template
	if count == 1:
		return template % (entity_id,)
	return template % (entity_id, entity_id)


def _normalize_entity_uri(entity_uri: str) -> str:
	text = str(entity_uri or "").strip()
	if text.startswith(_FREEBASE_URI_PREFIX):
		return text[len(_FREEBASE_URI_PREFIX) :]
	return text


def _label_rank(lang: str) -> int:
	text = str(lang or "").strip().lower()
	if text == "en":
		return 0
	if text == "":
		return 1
	return 2


def _render_ids2names_query(entity_ids: List[str]) -> str:
	values_clause = " ".join(f"ns:{entity_id}" for entity_id in entity_ids)
	return f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity ?targetEntity
WHERE {{
VALUES ?entity {{ {values_clause} }}
OPTIONAL {{
	?entity ns:type.object.name ?targetEntity .
	FILTER (lang(?targetEntity) = '' OR langMatches(lang(?targetEntity), 'en'))
}}
}}"""


def execurte_sparql(sparql_txt):
	meta = execurte_sparql_with_meta(sparql_txt)
	return meta.get("rows", [])


def execurte_sparql_with_meta(sparql_txt):
	last_error = ""
	timed_out = False
	for i in range(3):
		try:
			sparql = SPARQLWrapper(SPARQLPATH)
			sparql.setQuery(sparql_txt)
			sparql.setReturnFormat(JSON)
			sparql.setTimeout(SPARQL_TIMEOUT)
			results = sparql.query().convert()
			return {
				"rows": results["results"]["bindings"],
				"status": "ok",
				"timed_out": False,
				"attempts": i + 1,
				"error": "",
			}
		except BaseException as e:
			last_error = str(e)
			if "timed out" in last_error.lower() or "timeout" in last_error.lower():
				timed_out = True
			logging.error(
				f"Error in executing SPARQL query (attempt {i+1}/3) on endpoint={SPARQLPATH}: {e}"
			)
			if i < 2:
				time.sleep(2)
			continue
	logging.error(
		f"SPARQL query failed after 3 attempts on endpoint={SPARQLPATH}:\n{sparql_txt}"
	)
	return {
		"rows": [],
		"status": "timeout" if timed_out else "error",
		"timed_out": timed_out,
		"attempts": 3,
		"error": last_error,
	}


def abandon_rels(relation):
	if (
		relation == "type.object.type"
		or relation == "type.object.name"
		or relation.startswith("common.")
		or relation.startswith("freebase.")
		or "sameAs" in relation
	):
		return True
	return False


def replace_relation_prefix(relations):
	return [
		relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
		for relation in relations
	]


def replace_entities_prefix(entities):
	literal_entities = []
	uri_entities = []

	for entity in entities:
		binding = entity.get("targetEntity", {})
		binding_type = binding.get("type", "")
		value = str(binding.get("value", "")).strip()
		if not value:
			continue

		if binding_type in ("literal", "typed-literal"):
			literal_entities.append(value)
		elif binding_type == "uri":
			uri_entities.append(value.replace("http://rdf.freebase.com/ns/", ""))

	return literal_entities, uri_entities


def id2entity_name_or_type(entity_id):
	entity_id = str(entity_id or "").strip()
	if not entity_id:
		return "UnName_Entity"
	if entity_id in _ID2NAME_CACHE:
		return _ID2NAME_CACHE[entity_id]

	sparql_str = _render_id2name_query(entity_id)
	for i in range(3):
		try:
			sparql = SPARQLWrapper(SPARQLPATH)
			sparql.setQuery(sparql_str)
			sparql.setReturnFormat(JSON)
			sparql.setTimeout(SPARQL_TIMEOUT)
			results = sparql.query().convert()
			bindings = results.get("results", {}).get("bindings", [])
			if len(bindings) == 0:
				_ID2NAME_CACHE[entity_id] = "UnName_Entity"
				return "UnName_Entity"

			best_label = ""
			best_rank = 10**9
			fallback_label = ""
			for binding in bindings:
				label_binding = binding.get("targetEntity", {})
				label = str(label_binding.get("value", "")).strip()
				if not label:
					continue
				if not fallback_label:
					fallback_label = label
				rank = _label_rank(str(label_binding.get("xml:lang", "")))
				if rank < best_rank:
					best_rank = rank
					best_label = label

			resolved = best_label or fallback_label or "UnName_Entity"
			_ID2NAME_CACHE[entity_id] = resolved
			return resolved
		except BaseException as e:
			logging.error(
				f"Error in id2entity_name_or_type (attempt {i+1}/3) entity={entity_id} "
				f"endpoint={SPARQLPATH}: {e}"
			)
			if i < 2:
				time.sleep(1)
	_ID2NAME_CACHE[entity_id] = "UnName_Entity"
	return "UnName_Entity"


def id2entity_names(entity_ids: List[str]) -> Dict[str, str]:
	clean_ids: List[str] = []
	seen = set()
	for entity_id in entity_ids:
		text = str(entity_id or "").strip()
		if not text or text in seen:
			continue
		seen.add(text)
		clean_ids.append(text)

	if not clean_ids:
		return {}

	result: Dict[str, str] = {}
	pending_ids: List[str] = []
	for entity_id in clean_ids:
		if entity_id in _ID2NAME_CACHE:
			result[entity_id] = _ID2NAME_CACHE[entity_id]
		else:
			pending_ids.append(entity_id)

	if pending_ids:
		batch_size = max(1, SPARQL_LABEL_BATCH_SIZE)
		total_batches = (len(pending_ids) + batch_size - 1) // batch_size
		logging.info(
			"[Freebase][id2entity_names] start batch lookup | total_entities=%s | batch_size=%s | total_batches=%s",
			len(pending_ids),
			batch_size,
			total_batches,
		)

		unresolved: List[str] = []
		for batch_index, start in enumerate(range(0, len(pending_ids), batch_size), start=1):
			chunk = pending_ids[start : start + batch_size]
			logging.info(
				"[Freebase][id2entity_names] processing batch %s/%s | batch_entities=%s",
				batch_index,
				total_batches,
				len(chunk),
			)

			query = _render_ids2names_query(chunk)
			meta = execurte_sparql_with_meta(query)
			rows = meta.get("rows", []) or []

			best: Dict[str, Tuple[int, str]] = {}
			for row in rows:
				entity_uri = str(row.get("entity", {}).get("value", "")).strip()
				entity_id = _normalize_entity_uri(entity_uri)
				if not entity_id:
					continue
				label_binding = row.get("targetEntity", {})
				label = str(label_binding.get("value", "")).strip()
				if not label:
					continue
				rank = _label_rank(str(label_binding.get("xml:lang", "")))
				previous = best.get(entity_id)
				if previous is None or rank < previous[0]:
					best[entity_id] = (rank, label)

			for entity_id in chunk:
				if entity_id in best:
					resolved = best[entity_id][1]
					result[entity_id] = resolved
					_ID2NAME_CACHE[entity_id] = resolved
				else:
					unresolved.append(entity_id)

		for entity_id in unresolved:
			resolved = id2entity_name_or_type(entity_id)
			result[entity_id] = resolved

	return result
