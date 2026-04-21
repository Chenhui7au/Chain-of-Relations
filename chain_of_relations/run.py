#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   run.py
@Time    :   2026/03/26 14:48:31
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent

from chain_of_relations.schema import Entity
from chain_of_relations.kg_backend import get_backend


def str2bool(v):
	if isinstance(v, bool):
		return v
	v = str(v).lower().strip()
	if v in ("yes", "true", "t", "1", "y"):
		return True
	if v in ("no", "false", "f", "0", "n"):
		return False
	raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def prepare_dataset(dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], str]:
	if dataset_name == "cwq":
		dataset_path = str(CURRENT_DIR / ".." / ".." / "datasets" / "cwq" / "cwq.json")
		indicator = {
			"question_id": "id",
			"question": "question",
			"gold_answer": "answer",
			"topic_entity": "topic_entity",
		}
	elif dataset_name == "webqsp":
		dataset_path = str(CURRENT_DIR / ".." / ".." / "datasets" / "webqsp" / "webqsp.json")
		indicator = {
			"question_id": "id",
			"question": "question",
			"gold_answer": "answer",
			"topic_entity": "topic_entity",
		}
	elif dataset_name == "qald10_en":
		dataset_path = str(CURRENT_DIR / ".." / ".." / "datasets" / "qald10_en" / "qald10_en.json")
		indicator = {
			"question_id": "id",
			"question": "question",
			"gold_answer": "answer",
			"topic_entity": "topic_entity",
		}
	else:
		raise ValueError("dataset not found, choose from {cwq, webqsp, qald10_en}")

	with open(dataset_path, "r", encoding="utf-8") as f:
		datas = json.load(f)
	return datas, indicator, dataset_path


def validate_data_item(data: Dict[str, Any], indicator: Dict[str, str]) -> Tuple[bool, str]:
	for key in ("question_id", "question", "topic_entity"):
		field = indicator[key]
		if field not in data:
			return False, f"Missing field: {field}"

	topic_entity = data[indicator["topic_entity"]]
	if not isinstance(topic_entity, list) or len(topic_entity) == 0:
		return False, "topic_entity must be non-empty list"

	for entity in topic_entity:
		if not isinstance(entity, dict):
			return False, f"topic_entity item must be dict, got {type(entity)}"
		if "id" not in entity or "name" not in entity:
			return False, f"topic_entity item missing id/name: {entity}"

	return True, ""


def normalize_topic_entities(raw_topic_entity: List[Dict[str, Any]]) -> List[Entity]:
	topic_entities: List[Entity] = []
	for entity_info in raw_topic_entity:
		entity_id = entity_info.get("id", "")
		entity_name = entity_info.get("name", "")
		if entity_id and entity_name:
			topic_entities.append(Entity(id=entity_id, name=entity_name))
	return topic_entities


def normalize_results(results: Any) -> List[Dict[str, str]]:
	if isinstance(results, list):
		if len(results) == 0:
			return []
		if isinstance(results[0], dict):
			return results
		if isinstance(results[0], str):
			return [{"id": "", "name": item} for item in results if item]
	if isinstance(results, str):
		return [{"id": "", "name": results}] if results else []
	return []


def model_name_to_dirname(model_name: str, base_url: str = "") -> str:
	name = str(model_name or "").strip()
	if not name:
		return "unknown_model"
	parts = [part for part in name.split("/") if part.strip()]
	if not parts:
		return "unknown_model"
	dirname = parts[-1].strip()

	# Some gateways may append transient model suffixes (e.g. `_1`).
	# Normalize them to stable directory names for these base URLs.
	normalized_base_url = str(base_url or "").rstrip("/")
	normalize_dirname_base_urls = {
		"http://139.9.54.209:19999",  # Ctrl
		"https://api.tokenops.ai/v1",  # Quwan
	}
	if normalized_base_url in normalize_dirname_base_urls:
		ctrl_model_dirname_map = {
			"gpt-4.1-mini-2025-04-14_1": "gpt-4.1-mini",
			"gpt-5-mini-2025-08-07_1": "gpt-5-mini",
			"gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
			"gpt-5-mini-2025-08-07": "gpt-5-mini",
		}
		dirname = ctrl_model_dirname_map.get(dirname, dirname)

	return dirname


def append_jsonl(output_jsonl_file: str, item: Dict[str, Any]) -> None:
	output_dir = os.path.dirname(output_jsonl_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(output_jsonl_file, "a", encoding="utf-8") as f:
		f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_detail_json(output_json_file: str, detail: Dict[str, Any]) -> None:
	output_dir = os.path.dirname(output_json_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(output_json_file, "w", encoding="utf-8") as f:
		json.dump(detail, f, ensure_ascii=False, indent=2)


def count_done(output_jsonl_file: str) -> int:
	if not os.path.exists(output_jsonl_file):
		return 0
	with open(output_jsonl_file, "r", encoding="utf-8") as f:
		return sum(1 for _ in f)


def save_param_json_if_missing(
	output_dir: str,
	args: argparse.Namespace,
	resolved_values: Dict[str, Any],
) -> Tuple[str, bool]:
	os.makedirs(output_dir, exist_ok=True)
	param_json_file = os.path.join(output_dir, "param.json")
	if os.path.exists(param_json_file):
		return param_json_file, False

	payload = {
		"argparse": vars(args),
		"environment": {
			"MODEL_NAME": os.getenv("MODEL_NAME"),
		},
	}

	with open(param_json_file, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)

	return param_json_file, True


def build_step_history(prompt_history: List[Dict[str, Any]], sparql_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	merged: List[Dict[str, Any]] = []

	for index, item in enumerate(sparql_history):
		if "search_type" in item and "sparql" in item:
			merged.append(
				{
					"trace_id": item.get("trace_id"),
					"fallback_index": index,
					"step": {
						"step_type": "sparql",
						"search_type": item.get("search_type", ""),
						"sparql": item.get("sparql", ""),
						"result_count": item.get("result_count", -1),
						"status": item.get("status", "ok"),
						"timed_out": item.get("timed_out", False),
						"filter_caused_empty": item.get("filter_caused_empty", False),
					},
				}
			)
			continue

		for operation, query in item.items():
			if operation in ("result_count", "trace_id"):
				continue
			merged.append(
				{
					"trace_id": item.get("trace_id"),
					"fallback_index": index,
					"step": {
						"step_type": "sparql",
						"search_type": operation,
						"sparql": query,
						"result_count": item.get("result_count", -1),
						"status": item.get("status", "ok"),
						"timed_out": item.get("timed_out", False),
						"filter_caused_empty": item.get("filter_caused_empty", False),
					},
				}
			)

	base_index = len(merged)
	for index, item in enumerate(prompt_history):
		merged.append(
			{
				"trace_id": item.get("trace_id"),
				"fallback_index": base_index + index,
				"step": {
					"step_type": "llm",
					"operation_type": item.get("type", ""),
					"prompt": item.get("prompt", ""),
					"input_tokens": item.get("input_tokens", -1),
					"output_tokens": item.get("output_tokens", -1),
					"candidate_size": item.get("candidate_size", item.get("candidates_size", -1)),
					"filtered_candidate_size": item.get("filtered_candidate_size", -1),
					"response": item.get("response", ""),
					"parsed_result": item.get("parsed_result", ""),
				},
			}
		)

	has_trace = any(entry.get("trace_id") is not None for entry in merged)
	if has_trace:
		merged.sort(
			key=lambda entry: (
				entry.get("trace_id") if entry.get("trace_id") is not None else 10**9,
				entry.get("fallback_index", 10**9),
			)
		)

	return [entry["step"] for entry in merged]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run agent on dataset")
	parser.add_argument(
		"--method",
		type=str,
		default="tog",
		choices=["tog", "pog", "cor", "cot_prompt", "io_prompt"],
		help="reasoning method",
	)
	parser.add_argument("--dataset", type=str, default="webqsp", choices=["cwq", "webqsp", "qald10_en"])
	parser.add_argument("--kb", type=str, default="freebase", help="knowledge graph backend, e.g. freebase")
	parser.add_argument("--run_size", type=int, default=-1, help="number of questions to process (-1 for all)")
	parser.add_argument("--question_id", type=str, default="", help="run only the specified question id")
	parser.add_argument("--relation_width", type=int, default=3, help="Final kept relations after relation_prune")
	parser.add_argument("--entity_width", type=int, default=3, help="Final kept entities after entity_prune")
	parser.add_argument(
		"--sample_relation_threshold",
		type=int,
		default=1000,
		help="Max relation candidate size entering relation_prune",
	)
	parser.add_argument(
		"--sample_entity_threshold",
		type=int,
		default=500,
		help="Max entity candidate size entering entity_prune",
	)
	parser.add_argument("--depth", type=int, default=3, help="Max search depth")
	parser.add_argument("--temperature_exploration", type=float, default=0.3)
	parser.add_argument("--temperature_reasoning", type=float, default=0.1)
	parser.add_argument("--max_token", type=int, default=512)
	parser.add_argument("--remove_unnecessary_rel", type=str2bool, default=True)
	parser.add_argument("--save_detail", type=str2bool, default=True)
	parser.add_argument(
		"--log_level",
		type=str,
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="python logging level",
	)
	return parser


def load_agent_class(method: str):
	if method == "tog":
		from chain_of_relations.methods.tog.agent import ToGAgent

		return ToGAgent

	if method == "pog":
		from chain_of_relations.methods.pog.agent import PoGAgent

		return PoGAgent

	if method == "cor":
		from chain_of_relations.methods.cor.agent import CoRAgent

		return CoRAgent

	if method == "cot_prompt":
		from chain_of_relations.methods.cot_prompt.agent import CoTPromptAgent

		return CoTPromptAgent

	if method == "io_prompt":
		from chain_of_relations.methods.io_prompt.agent import IOPromptAgent

		return IOPromptAgent

	raise ValueError(f"Unsupported method: {method}")


def main() -> None:
	args = build_parser().parse_args()
	model_name = os.getenv("MODEL_NAME")
	openai_base_url = os.getenv("OPENAI_BASE_URL", "")
	if not model_name:
		raise ValueError("MODEL_NAME is required in environment")

	logging.basicConfig(
		level=getattr(logging, args.log_level.upper(), logging.INFO),
		format="%(asctime)s | %(levelname)s | %(message)s",
	)

	datas, indicator, dataset_path = prepare_dataset(args.dataset)
	logging.info(f"Loaded dataset: {dataset_path}, size={len(datas)}")

	if args.question_id:
		target_qid = str(args.question_id)
		datas = [item for item in datas if str(item.get(indicator["question_id"])) == target_qid]
		if not datas:
			raise ValueError(f"question_id not found in dataset: {args.question_id}")
		logging.info(f"Filtered dataset by question_id={args.question_id}, size={len(datas)}")

	AgentClass = load_agent_class(args.method)
	kg_backend = get_backend(args.kb)
	relation_width = int(args.relation_width)
	entity_width = int(args.entity_width)

	try:
		if args.method == "tog":
			agent = AgentClass(
				model_name=model_name,
				relation_width=relation_width,
				entity_width=entity_width,
				depth=args.depth,
				sample_relation_threshold=args.sample_relation_threshold,
				sample_entity_threshold=args.sample_entity_threshold,
				remove_unnecessary_rel=args.remove_unnecessary_rel,
				temperature_exploration=args.temperature_exploration,
				temperature_reasoning=args.temperature_reasoning,
				max_token=args.max_token,
				backend=kg_backend,
			)
		elif args.method == "pog":
			agent = AgentClass(
				model_name=model_name,
				relation_width=relation_width,
				entity_width=entity_width,
				depth=args.depth,
				sample_relation_threshold=args.sample_relation_threshold,
				sample_entity_threshold=args.sample_entity_threshold,
				remove_unnecessary_rel=args.remove_unnecessary_rel,
				temperature_exploration=args.temperature_exploration,
				temperature_reasoning=args.temperature_reasoning,
				max_token=args.max_token,
				backend=kg_backend,
			)
		elif args.method == "cor":
			agent = AgentClass(
				model_name=model_name,
				relation_width=relation_width,
				depth=args.depth,
				sample_relation_threshold=args.sample_relation_threshold,
				remove_unnecessary_rel=args.remove_unnecessary_rel,
				temperature_exploration=args.temperature_exploration,
				temperature_reasoning=args.temperature_reasoning,
				max_token=args.max_token,
				backend=kg_backend,
			)
		elif args.method == "cot_prompt" or args.method == "io_prompt":
			agent = AgentClass(
				model_name=model_name,
				temperature_reasoning=args.temperature_reasoning,
				max_token=args.max_token,
				backend=kg_backend,
			)
		else:
			raise ValueError(f"Unsupported method: {args.method}")
	except Exception as e:
		logging.error("Failed to initialize %s agent: %s", args.method.upper(), e)
		raise SystemExit(1)

	model_dirname = model_name_to_dirname(model_name, openai_base_url)
	project_root = CURRENT_DIR.parent.parent
	output_dir = str(project_root / "results" / args.method / args.dataset / model_dirname)
	output_jsonl_file = str(Path(output_dir) / "predict.jsonl")
	param_json_file, param_created = save_param_json_if_missing(
		output_dir=output_dir,
		args=args,
		resolved_values={
			"model_name": model_name,
			"model_dirname": model_dirname,
			"output_jsonl_file": output_jsonl_file,
		},
	)
	if param_created:
		logging.info(f"created param file: {param_json_file}")
	else:
		logging.info(f"param file exists, skip create: {param_json_file}")

	done_size = 0 if args.question_id else count_done(output_jsonl_file)
	logging.info(f"already done: {done_size}")
	if model_dirname != str(model_name):
		logging.info(f"model_name path normalized: '{model_name}' -> '{model_dirname}'")

	counter = 0
	for data in tqdm(datas):
		counter += 1
		if counter <= done_size:
			continue
		if args.run_size > 0 and counter > args.run_size:
			break

		valid, err = validate_data_item(data, indicator)
		if not valid:
			logging.error(f"Data validation failed for item {counter}: {err}")
			continue

		question_id = data[indicator["question_id"]]
		question = data[indicator["question"]]
		gold_answer = data.get(indicator["gold_answer"], [])
		sparql = data.get("sparql") or data.get("SPARQL") or ""
		raw_topic_entity = data.get(indicator["topic_entity"], [])
		logging.info(
			"[RUN][%s] Start qid=%s | question=%s | topic_entities=%s",
			args.method.upper(),
			question_id,
			question,
			len(raw_topic_entity) if isinstance(raw_topic_entity, list) else 0,
		)

		topic_entities = normalize_topic_entities(raw_topic_entity)
		if not topic_entities:
			logging.error(f"No valid topic entities for question {question_id}")
			continue

		prompt_history: List[Dict[str, Any]] = []
		sparql_history: List[Dict[str, Any]] = []
		result = agent.answer(
			question=question,
			topic_entities=topic_entities,
			prompt_history=prompt_history,
			sparql_history=sparql_history,
		)

		jsonl_item = {
			"action": result.get("action", ""),
			"id": question_id,
			"question": question,
			"gold_answer": gold_answer if isinstance(gold_answer, list) else [],
			"results": normalize_results(result.get("results", [])),
			"reasoning_chains": result.get("reasoning_chains", ""),
		}
		append_jsonl(output_jsonl_file, jsonl_item)

		if args.save_detail:
			detail_output = str(Path(output_dir) / f"{question_id}.json")
			step_history = result.get("step_history")
			if not isinstance(step_history, list) or len(step_history) == 0:
				step_history = build_step_history(
					prompt_history=result.get("prompt_history", prompt_history),
					sparql_history=result.get("sparql_history", sparql_history),
				)
			detail = {
				"id": question_id,
				"question": question,
				"sparql": sparql,
				"topic_entities": raw_topic_entity if isinstance(raw_topic_entity, list) else [],
				"gold_answer": gold_answer if isinstance(gold_answer, list) else [],
				"results": jsonl_item["results"],
				"reasoning_chains": result.get("reasoning_chains", ""),
				"step_history": step_history,
			}
			save_detail_json(detail_output, detail)

		logging.info(
			f"[{args.method.upper()}][{args.dataset}] Done {question_id} | action={jsonl_item['action']} "
			f"| pred={len(jsonl_item['results'])} | llm_steps={len(prompt_history)} | sparql_steps={len(sparql_history)}"
		)

	print("=== run_agent done ===")
	print("output_jsonl:", output_jsonl_file)


if __name__ == "__main__":
	main()
