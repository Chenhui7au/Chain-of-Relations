#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   run_freebase_sparql.py
@Time    :   2026/04/03 08:50:00
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""

import os
import json
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQLPATH = os.getenv("FREEBASE_SPARQL_ENDPOINT")
print(f"Using SPARQL endpoint: {SPARQLPATH}")


def _extract_mid(entity_uri: str) -> str:
    prefix = "http://rdf.freebase.com/ns/"
    if entity_uri.startswith(prefix):
        return entity_uri[len(prefix):]
    return entity_uri


def _run_sparql(sparql_txt: str):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def retrieve_entities():
    sparql_txt = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x
WHERE {
FILTER (?x != ns:m.0f8l9c)
FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))
ns:m.0f8l9c ns:location.location.adjoin_s ?y .
?y ns:location.adjoining_relationship.adjoins ?x .
?x ns:location.location.contains ?c .
?c ns:aviation.airport.serves ns:m.05g2b .
}
    """
    results = _run_sparql(sparql_txt)

    entity_ids = []
    for row in results.get("results", {}).get("bindings", []):
        entity_uri = row.get("x", {}).get("value", "")
        entity_ids.append(_extract_mid(entity_uri))
    return entity_ids


def map_literal_names(entity_ids):
    if not entity_ids:
        return []

    values_clause = " ".join(f"ns:{entity_id}" for entity_id in entity_ids)
    sparql_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x ?name
WHERE {{
VALUES ?x {{ {values_clause} }}
OPTIONAL {{
    ?x ns:type.object.name ?name .
    FILTER (lang(?name) = 'en')
}}
}}
    """
    results = _run_sparql(sparql_txt)

    name_map = {}
    for row in results.get("results", {}).get("bindings", []):
        entity_uri = row.get("x", {}).get("value", "")
        entity_id = _extract_mid(entity_uri)
        literal_name = row.get("name", {}).get("value", entity_id)
        name_map[entity_id] = literal_name

    return [{"id": entity_id, "name": name_map.get(entity_id, entity_id)} for entity_id in entity_ids]


def test():
    try:
        retrieved_ids = retrieve_entities()
        print("[Step 1] Retrieval Results (IDs):")
        print(json.dumps(retrieved_ids, ensure_ascii=False, indent=2))

        mapped_results = map_literal_names(retrieved_ids)
        print("[Step 2] Literal Name Mapping:")
        print(json.dumps(mapped_results, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Your database is not installed properly !!! ({exc})")


if __name__ == "__main__":
    test()