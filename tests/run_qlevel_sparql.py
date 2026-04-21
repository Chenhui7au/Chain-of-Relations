#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   run_qlevel_sparql.py
@Time    :   2026/04/09 11:35:59
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


from SPARQLWrapper import JSON, SPARQLWrapper

def query_qlever(sparql_query):
  endpoint = "https://qlever.dev/api/wikidata"
  sparql = SPARQLWrapper(endpoint)
  sparql.setQuery(sparql_query)
  sparql.setReturnFormat(JSON)
  sparql.setTimeout(30)
  return sparql.query().convert()

# 示例：查询所有是“人类”的实体（前10个）
sparql = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?targetEntity
WHERE {
  VALUES ?entity { wd:Q26725927 wd:Q26725925 wd:Q26725931 wd:Q26725929 wd:Q26725933 wd:Q26726651 wd:Q26726652 wd:Q26726653 wd:Q26726674 wd:Q26726675 wd:Q26726673 wd:Q26727622 wd:Q26728465 wd:Q26728558 wd:Q26728559 wd:Q26728581 wd:Q26729319 wd:Q26729322 wd:Q26729323 wd:Q26729321 wd:Q26729326 wd:Q26729327 wd:Q26729325 wd:Q26730149 wd:Q26732262 wd:Q26732260 wd:Q26732685 wd:Q26738168 wd:Q26743570 wd:Q26743569 wd:Q26743635 wd:Q26745345 wd:Q26745392 wd:Q26751879 wd:Q26751877 wd:Q26751882 wd:Q26751883 wd:Q26751880 wd:Q26751881 wd:Q26751887 wd:Q26751884 wd:Q26751891 wd:Q26753551 wd:Q26753554 wd:Q26753555 wd:Q26753558 wd:Q26753559 wd:Q26753562 wd:Q26753563 wd:Q26753561 wd:Q26753566 wd:Q26753567 wd:Q26753565 wd:Q26753568 wd:Q26753569 wd:Q26753578 wd:Q26753579 wd:Q26753576 wd:Q26753577 wd:Q26753582 wd:Q26753583 wd:Q26753580 wd:Q26753581 wd:Q26753586 wd:Q26753584 wd:Q26753588 wd:Q26753589 wd:Q26753598 wd:Q26753599 wd:Q26753597 wd:Q26753602 wd:Q26753603 wd:Q26753600 wd:Q26753601 wd:Q26753607 wd:Q26753604 wd:Q26753605 wd:Q26753608 wd:Q26753612 wd:Q26753659 wd:Q26753666 wd:Q26753667 wd:Q26753664 wd:Q26753665 wd:Q26753674 wd:Q26753681 wd:Q26753688 wd:Q26753695 wd:Q26753692 wd:Q26753699 wd:Q26753707 wd:Q26753708 wd:Q26753709 wd:Q26753720 wd:Q26753738 wd:Q26753752 wd:Q26756409 wd:Q26756587 wd:Q26756665 wd:Q26756917 }
  ?entity rdfs:label ?targetEntity .
}
"""

results = query_qlever(sparql)
print(results)