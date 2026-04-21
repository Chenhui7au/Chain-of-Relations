# Datasets

## Fields

The following fields are required for all datasets in this project.

| Field | Type | Description |
| --- | --- | --- |
| id | string or number | Sample identifier |
| question | string | Natural language question |
| sparql | string | Target SPARQL query |
| meta_data | object | Metadata container |
| meta_data.compositionality_type | string | Question compositionality category |
| answer | array[object] | Gold answer list |
| answer[].id | string | Answer entity or value id |
| answer[].name | string | Answer surface name or value |
| topic_entity | array[object] | Mentioned entities in the question |
| topic_entity[].id | string | Topic entity id |
| topic_entity[].name | string | Topic entity name |
