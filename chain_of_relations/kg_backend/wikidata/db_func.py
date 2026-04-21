#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from SPARQLWrapper import JSON, SPARQLWrapper


SPARQLPATH = os.getenv("WIKIDATA_SPARQL_ENDPOINT", "https://query.wikidata.org/sparql")
SPARQL_TIMEOUT = int(os.getenv("WIKIDATA_SPARQL_TIMEOUT", os.getenv("SPARQL_TIMEOUT", "30")))
SPARQL_RELATION_LIMIT = 1000
SPARQL_ENTITY_LIMIT = 1000
SPARQL_ID_LIMIT = 1
WIKIDATA_LABEL_LANGS = os.getenv("WIKIDATA_LABEL_LANGS", "en").strip()
WIKIDATA_LABEL_BATCH_SIZE = int(os.getenv("WIKIDATA_LABEL_BATCH_SIZE", "100"))


def _load_query_templates() -> Dict[str, str]:
	query_template_path = Path(__file__).resolve().parent / "query_template.yml"
	with open(query_template_path, "r", encoding="utf-8") as f:
		templates = yaml.safe_load(f) or {}

	required_keys = {
		"sparql_relations",
		"sparql_entities",
		"sparql_constraints",
		"sparql_constraint_pairs",
		"sparql_id2name",
		"sparql_ids2names",
		"sparql_relation_labels",
	}
	missing_keys = [key for key in required_keys if key not in templates]
	if missing_keys:
		raise ValueError(f"Missing query templates in {query_template_path}: {missing_keys}")
	return templates


_QUERY_TEMPLATES = _load_query_templates()

sparql_relations = _QUERY_TEMPLATES["sparql_relations"]
sparql_entities = _QUERY_TEMPLATES["sparql_entities"]
sparql_constraints = _QUERY_TEMPLATES["sparql_constraints"]
sparql_constraint_pairs = _QUERY_TEMPLATES["sparql_constraint_pairs"]
sparql_id2name = _QUERY_TEMPLATES["sparql_id2name"]
sparql_ids2names = _QUERY_TEMPLATES["sparql_ids2names"]
sparql_relation_labels = _QUERY_TEMPLATES["sparql_relation_labels"]


def build_sparql_relations(query: str) -> str:
	return sparql_relations.replace("{{query}}", query)


def build_sparql_entities(query: str) -> str:
	return sparql_entities.replace("{{query}}", query)


def build_sparql_constraints(query: str) -> str:
	return sparql_constraints.replace("{{query}}", query)


def build_sparql_constraint_pairs(query: str) -> str:
	return sparql_constraint_pairs.replace("{{query}}", query)


def _lang_filter_values() -> str:
	langs = [lang.strip() for lang in WIKIDATA_LABEL_LANGS.split(",") if lang.strip()]
	if not langs:
		langs = ["en"]
	return ", ".join([f'"{lang}"' for lang in langs])


def _lang_order() -> List[str]:
	langs = [lang.strip() for lang in WIKIDATA_LABEL_LANGS.split(",") if lang.strip()]
	return langs or ["en"]


def _render_id2name_query(entity_id: str) -> str:
	return (
		sparql_id2name
		.replace("{{entity_id}}", entity_id)
		.replace("{{langs}}", _lang_filter_values())
	)


def _render_ids2names_query(entity_ids: List[str]) -> str:
	entity_values = " ".join([f"wd:{entity_id}" for entity_id in entity_ids])
	return (
		sparql_ids2names
		.replace("{{entity_values}}", entity_values)
		.replace("{{langs}}", _lang_filter_values())
	)


def _render_relation_labels_query(relation_ids: List[str]) -> str:
	relation_values = " ".join([f"wd:{relation_id}" for relation_id in relation_ids])
	return (
		sparql_relation_labels
		.replace("{{relation_values}}", relation_values)
		.replace("{{langs}}", _lang_filter_values())
	)


def execurte_sparql(sparql_txt: str) -> List[Dict]:
	meta = execurte_sparql_with_meta(sparql_txt)
	return meta.get("rows", [])


def execurte_sparql_with_meta(sparql_txt: str) -> Dict:
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
				"rows": results.get("results", {}).get("bindings", []),
				"status": "ok",
				"timed_out": False,
				"attempts": i + 1,
				"error": "",
			}
		except BaseException as e:
			last_error = str(e)
			is_timeout = "timed out" in last_error.lower() or "timeout" in last_error.lower()
			if is_timeout:
				timed_out = True
				logging.error(
					f"Error in executing SPARQL query (attempt {i + 1}/3) on endpoint={SPARQLPATH}: {e}\n"
					f"SPARQL(timeout):\n{sparql_txt}"
				)
			else:
				logging.error(
					f"Error in executing SPARQL query (attempt {i + 1}/3) on endpoint={SPARQLPATH}: {e}"
				)
			if i < 2:
				time.sleep(2)
			continue

	logging.error(f"SPARQL query failed after 3 attempts on endpoint={SPARQLPATH}:\n{sparql_txt}")
	return {
		"rows": [],
		"status": "timeout" if timed_out else "error",
		"timed_out": timed_out,
		"attempts": 3,
		"error": last_error,
	}


def abandon_rels(relation: str) -> bool:
	# frequent generic/typing properties that usually add little value for QA exploration
	return relation in {"P373", "P910"}


def _normalize_relation_uri(value: str) -> str:
	text = str(value or "").strip()
	if not text:
		return ""
	if text.startswith("http://www.wikidata.org/prop/direct/"):
		return text.replace("http://www.wikidata.org/prop/direct/", "")
	if text.startswith("http://www.wikidata.org/prop/"):
		return text.split("/")[-1]
	if text.startswith("wdt:"):
		return text.replace("wdt:", "")
	return text


def replace_relation_prefix(relations: List[Dict]) -> List[str]:
	values: List[str] = []
	for relation in relations:
		raw = relation.get("relation", {}).get("value", "")
		norm = _normalize_relation_uri(raw)
		if norm:
			values.append(norm)
	return values


def _normalize_entity_uri(value: str) -> str:
	text = str(value or "").strip()
	if not text:
		return ""
	if text.startswith("http://www.wikidata.org/entity/"):
		return text.replace("http://www.wikidata.org/entity/", "")
	if text.startswith("wd:"):
		return text.replace("wd:", "")
	return text


def replace_entities_prefix(entities: List[Dict]) -> Tuple[List[str], List[str]]:
	literal_entities: List[str] = []
	uri_entities: List[str] = []

	for entity in entities:
		binding = entity.get("targetEntity", {})
		binding_type = binding.get("type", "")
		value = str(binding.get("value", "")).strip()
		if not value:
			continue

		if binding_type in ("literal", "typed-literal"):
			literal_entities.append(value)
		elif binding_type == "uri":
			norm = _normalize_entity_uri(value)
			if norm:
				uri_entities.append(norm)

	return literal_entities, uri_entities


def id2entity_name_or_type(entity_id: str) -> str:
	sparql_str = _render_id2name_query(entity_id)
	lang_priority = _lang_order()
	lang_rank = {lang: idx for idx, lang in enumerate(lang_priority)}
	for i in range(3):
		try:
			sparql = SPARQLWrapper(SPARQLPATH)
			sparql.setQuery(sparql_str)
			sparql.setReturnFormat(JSON)
			sparql.setTimeout(SPARQL_TIMEOUT)
			results = sparql.query().convert()
			bindings = results.get("results", {}).get("bindings", [])
			if not bindings:
				return "UnName_Entity"

			best_label = ""
			best_rank = 10**9
			fallback_label = ""
			for binding in bindings:
				label_binding = binding.get("targetEntity", {})
				label = str(label_binding.get("value", "")).strip()
				lang = str(label_binding.get("xml:lang", "")).strip()
				if not label:
					continue
				if not fallback_label:
					fallback_label = label
				rank = lang_rank.get(lang, 10**6)
				if rank < best_rank:
					best_rank = rank
					best_label = label

			return best_label or fallback_label or "UnName_Entity"
		except BaseException as e:
			error_text = str(e)
			is_timeout = "timed out" in error_text.lower() or "timeout" in error_text.lower()
			if is_timeout:
				logging.error(
					f"Error in id2entity_name_or_type (attempt {i + 1}/3) entity={entity_id} endpoint={SPARQLPATH}: {e}\n"
					f"SPARQL(timeout):\n{sparql_str}"
				)
			else:
				logging.error(
					f"Error in id2entity_name_or_type (attempt {i + 1}/3) entity={entity_id} endpoint={SPARQLPATH}: {e}"
				)
			if i < 2:
				time.sleep(1)
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

	lang_priority = _lang_order()
	lang_rank = {lang: idx for idx, lang in enumerate(lang_priority)}
	batch_size = max(1, WIKIDATA_LABEL_BATCH_SIZE)
	result: Dict[str, str] = {entity_id: "UnName_Entity" for entity_id in clean_ids}
	total_entities = len(clean_ids)
	total_batches = (total_entities + batch_size - 1) // batch_size
	logging.info(
		"[Wikidata][id2entity_names] start batch lookup | total_entities=%s | batch_size=%s | total_batches=%s",
		total_entities,
		batch_size,
		total_batches,
	)

	for batch_index, start in enumerate(range(0, len(clean_ids), batch_size), start=1):
		chunk = clean_ids[start : start + batch_size]
		logging.info(
			"[Wikidata][id2entity_names] processing batch %s/%s | batch_entities=%s",
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
			lang = str(label_binding.get("xml:lang", "")).strip()
			if not label:
				continue
			rank = lang_rank.get(lang, 10**6)
			previous = best.get(entity_id)
			if previous is None or rank < previous[0]:
				best[entity_id] = (rank, label)

		for entity_id in chunk:
			if entity_id in best:
				result[entity_id] = best[entity_id][1]
			else:
				result[entity_id] = id2entity_name_or_type(entity_id)

	return result


def relation_ids2labels(relation_ids: List[str]) -> Dict[str, str]:
	clean_ids = []
	seen = set()
	for relation_id in relation_ids:
		text = str(relation_id or "").strip()
		if not text or text in seen:
			continue
		seen.add(text)
		clean_ids.append(text)

	if not clean_ids:
		return {}

	query = _render_relation_labels_query(clean_ids)
	meta = execurte_sparql_with_meta(query)
	rows = meta.get("rows", []) or []
	if not rows:
		return {}

	lang_priority = _lang_order()
	lang_rank = {lang: idx for idx, lang in enumerate(lang_priority)}

	best: Dict[str, Tuple[int, str]] = {}
	for row in rows:
		relation_id = str(row.get("relationId", {}).get("value", "")).strip()
		label_binding = row.get("relationLabel", {})
		label = str(label_binding.get("value", "")).strip()
		lang = str(label_binding.get("xml:lang", "")).strip()
		if not relation_id or not label:
			continue
		rank = lang_rank.get(lang, 10**6)
		previous = best.get(relation_id)
		if previous is None or rank < previous[0]:
			best[relation_id] = (rank, label)

	return {relation_id: label for relation_id, (_rank, label) in best.items()}
