#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   graph.py
@Time    :   2026/04/17 04:56:49
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   Plot LLM call-count distributions (x=count, y=frequency).
"""

from collections import Counter
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from chain_of_relations.eval.accuracy import eval_f1, eval_hit

try:
	import matplotlib.pyplot as plt
except ImportError as exc:
	raise ImportError(
		"matplotlib is required for plotting. Please install it first, e.g. pip install matplotlib"
	) from exc


METHOD_LABEL = {
	"tog": "ToG",
	"pog": "PoG",
	"cor": "CoR",
}

METHOD_COLOR = {
	"tog": "#2CA02C",
	"pog": "#D62728",
	"cor": "#1F77B4",
}

DATASET_TITLE = {
	"cwq": "CWQ",
	"webqsp": "WebQSP",
	"qald10_en": "QALD10",
}

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_OUTPUT_DIR = Path("results_final/plots")
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
DEFAULT_DATASETS = ["webqsp", "cwq", "qald10_en"]
DEFAULT_METHODS = ["tog", "pog", "cor"]
DEFAULT_DPI = 220
DEFAULT_CALL_FOCUS_MIN = 0
DEFAULT_CALL_FOCUS_MAX = 40


def load_predictions(predict_file: Path) -> List[dict]:
	rows: List[dict] = []
	with predict_file.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(json.loads(line))
	return rows


def extract_answer_from_braces(text: str):
	if not isinstance(text, str):
		return None

	pattern = r"\{([^}]+)\}"
	matches = re.findall(pattern, text)
	if not matches:
		return None

	answer_text = matches[-1].strip()
	if "," in answer_text:
		return [item.strip() for item in answer_text.split(",") if item.strip()]
	if " and " in answer_text.lower():
		return [item.strip() for item in re.split(r"\s+and\s+", answer_text, flags=re.IGNORECASE) if item.strip()]
	if answer_text:
		return [answer_text]
	return None


def is_integer_text(text: str) -> bool:
	return bool(re.fullmatch(r"[-+]?\d+", str(text or "").strip()))


def postprocess_prediction_for_dataset(dataset: str, prediction: List[str], gold_answer: List[str]) -> List[str]:
	dataset_key = str(dataset or "").strip().lower()
	if dataset_key not in {"quad2", "quad_2"}:
		return prediction

	if not prediction:
		return prediction

	normalized = [str(item).strip() for item in prediction]
	for item in normalized:
		if is_integer_text(item):
			return [item]

	gold_non_empty = [
		str(item).strip() for item in gold_answer
		if str(item).strip() and str(item).strip().lower() != "none"
	]
	gold_is_numeric = bool(gold_non_empty) and all(is_integer_text(item) for item in gold_non_empty)
	if not gold_is_numeric:
		return normalized

	non_empty = [
		str(item).strip() for item in normalized
		if str(item).strip() and str(item).strip().lower() != "none"
	]
	return [str(len(non_empty))]


def parse_prediction(row: dict, dataset: str) -> Tuple[List[str], List[str]]:
	gold_answers = row.get("gold_answer", []) or []
	gold = [ans.get("name", "") for ans in gold_answers if isinstance(ans, dict) and ans.get("name")]

	pred_results = row.get("results", [])
	if not isinstance(pred_results, list) or not pred_results:
		prediction = ["None"]
	else:
		raw_predictions = [
			res.get("name", "") for res in pred_results
			if isinstance(res, dict) and res.get("name")
		]
		prediction: List[str] = []
		for raw_pred in raw_predictions:
			extracted = extract_answer_from_braces(raw_pred)
			if extracted:
				prediction.extend(extracted)
			else:
				prediction.append(raw_pred)
		if not prediction:
			prediction = ["None"]

	prediction = postprocess_prediction_for_dataset(dataset, prediction, gold)
	return prediction, gold


def _safe_int(value) -> int:
	try:
		if value is None:
			return 0
		return int(value)
	except (TypeError, ValueError):
		return 0


def build_llm_cost_index(result_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
	token_by_id: Dict[str, int] = {}
	calls_by_id: Dict[str, int] = {}

	json_files = glob.glob(str(result_dir / "*.json"))
	for json_file in json_files:
		file_name = os.path.basename(json_file)
		if file_name in {"predict.json", "param.json"}:
			continue

		file_stem = os.path.splitext(file_name)[0]
		try:
			with open(json_file, "r", encoding="utf-8") as f:
				data = json.load(f)
		except Exception as e:
			print(f"[WARN] Skip malformed file {json_file}: {e}")
			continue

		step_history = data.get("step_history", [])
		total_tokens = 0
		llm_calls = 0

		if isinstance(step_history, list):
			for step in step_history:
				if not isinstance(step, dict):
					continue
				if step.get("step_type") == "llm":
					llm_calls += 1
					total_tokens += _safe_int(step.get("input_tokens", 0))
					total_tokens += _safe_int(step.get("output_tokens", 0))

		data_id = str(data.get("id", "")).strip()
		candidate_ids = {file_stem}
		if data_id:
			candidate_ids.add(data_id)

		for key in candidate_ids:
			token_by_id[key] = total_tokens
			calls_by_id[key] = llm_calls

	return token_by_id, calls_by_id


def evaluate_distribution(result_dir: Path, dataset: str) -> dict:
	predict_file = result_dir / "predict.jsonl"
	if not predict_file.exists():
		raise FileNotFoundError(f"Missing predict file: {predict_file}")

	predictions = load_predictions(predict_file)
	if not predictions:
		raise ValueError(f"No prediction rows found in {predict_file}")

	token_by_id, calls_by_id = build_llm_cost_index(result_dir)

	cumulative_hit = 0
	cumulative_f1 = 0.0
	cumulative_calls = 0
	total_tokens = 0
	missing_cost_rows = 0

	call_counts: List[int] = []

	for row in predictions:
		prediction, gold = parse_prediction(row, dataset)
		prediction_str = " ".join(prediction)

		hit = eval_hit(prediction_str, gold) if gold else 0
		if gold:
			f1, _, _ = eval_f1(prediction, gold)
		else:
			f1 = 0.0

		sample_id = str(row.get("id", "")).strip()
		sample_tokens = token_by_id.get(sample_id)
		sample_calls = calls_by_id.get(sample_id)

		if sample_tokens is None:
			missing_cost_rows += 1
			sample_tokens = 0
		if sample_calls is None:
			sample_calls = 0

		total_tokens += sample_tokens
		cumulative_hit += hit
		cumulative_f1 += f1
		cumulative_calls += sample_calls
		call_counts.append(sample_calls)

	total_samples = len(predictions)
	return {
		"distribution": {
			"call_counts": call_counts,
		},
		"summary": {
			"samples": total_samples,
			"hit1_percent": (cumulative_hit / total_samples) * 100.0,
			"f1_percent": (cumulative_f1 / total_samples) * 100.0,
			"avg_total_tokens": total_tokens / total_samples,
			"avg_llm_calls": cumulative_calls / total_samples,
			"missing_cost_rows": missing_cost_rows,
		},
	}


def plot_distributions(
	dist_map: Dict[Tuple[str, str], dict],
	datasets: List[str],
	methods: List[str],
	output_path: Path,
	model_name: str,
	dpi: int,
	focus_min: int,
	focus_max: int,
) -> None:
	rows = len(datasets)
	fig, axes = plt.subplots(rows, 1, figsize=(12, 4 * rows), squeeze=False)

	for i, dataset in enumerate(datasets):
		ax = axes[i][0]
		x_series = list(range(focus_min, focus_max + 1))
		method_payloads: List[Tuple[str, dict]] = []
		for method in methods:
			key = (dataset, method)
			payload = dist_map.get(key)
			if not payload:
				continue
			method_payloads.append((method, payload))

		if method_payloads:
			avg_calls_by_method = {}
			for method, payload in method_payloads:
				counter = Counter(payload["distribution"]["call_counts"])
				total_samples = max(1, int(payload["summary"]["samples"]))
				y_series = [counter.get(x, 0) / total_samples for x in x_series]
				color = METHOD_COLOR.get(method, None)
				avg_calls_by_method[method] = payload["summary"].get("avg_llm_calls", 0.0)

				ax.plot(
					x_series,
					y_series,
					label=METHOD_LABEL.get(method, method),
					color=color,
					linewidth=2.2,
				)
				ax.fill_between(
					x_series,
					y_series,
					0,
					color=color,
					alpha=0.18,
				)

			mean_text = (
				f"ToG Mean: {avg_calls_by_method.get('tog', 0.0):.2f}    "
				f"PoG Mean: {avg_calls_by_method.get('pog', 0.0):.2f}    "
				f"CoR Mean: {avg_calls_by_method.get('cor', 0.0):.2f}"
			)
			ax.text(
				0.5,
				0.95,
				mean_text,
				transform=ax.transAxes,
				ha="center",
				va="top",
				fontsize=18,
				color="black",
				bbox={
					"boxstyle": "square,pad=0.2",
					"facecolor": "white",
					"edgecolor": "black",
					"linewidth": 1.0,
					"alpha": 0.95,
				},
			)

		dataset_title = DATASET_TITLE.get(dataset, dataset)
		ax.set_title(dataset_title, fontsize=25, fontweight="bold")
		ax.set_xlabel("LLM Call", fontsize=20)
		ax.set_ylabel("Frequency", fontsize=20)
		tick_step = 5 if (focus_max - focus_min) >= 20 else 2
		ax.set_xticks(list(range(focus_min, focus_max + 1, tick_step)))
		ax.set_xlim(focus_min, focus_max)
		ax.set_ylim(bottom=0)
		ax.tick_params(axis="both", labelsize=13)
		ax.grid(True, linestyle="--", alpha=0.35)
		if method_payloads:
			ax.legend(loc="upper right", fontsize=18)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def save_summary_csv(summary_rows: List[dict], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"dataset",
		"method",
		"samples",
		"hit1_percent",
		"f1_percent",
		"avg_total_tokens",
		"avg_llm_calls",
		"missing_cost_rows",
		"result_dir",
	]
	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in summary_rows:
			writer.writerow(row)


def save_distribution_csv(dist_rows: List[dict], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["dataset", "method", "llm_call_count", "count", "proportion"]
	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in dist_rows:
			writer.writerow(row)


def main() -> None:
	results_root = DEFAULT_RESULTS_ROOT
	output_dir = DEFAULT_OUTPUT_DIR
	model = DEFAULT_MODEL_NAME
	datasets = DEFAULT_DATASETS
	methods = DEFAULT_METHODS

	dist_map: Dict[Tuple[str, str], dict] = {}
	summary_rows: List[dict] = []
	dist_rows: List[dict] = []

	for dataset in datasets:
		for method in methods:
			# Use explicit method/dataset/model path; no wildcard scanning.
			# This intentionally excludes sibling dirs such as cor/cwq_depth3 and cor/cwq_depth5.
			result_dir = results_root / method / dataset / model

			if not result_dir.exists():
				print(f"[WARN] Missing directory, skip: {result_dir}")
				continue

			try:
				payload = evaluate_distribution(result_dir=result_dir, dataset=dataset)
			except Exception as e:
				print(f"[WARN] Failed to evaluate {result_dir}: {e}")
				continue

			dist_map[(dataset, method)] = payload
			summary = payload["summary"]
			total_samples = max(1, int(summary["samples"]))
			summary_rows.append(
				{
					"dataset": dataset,
					"method": method,
					"samples": summary["samples"],
					"hit1_percent": f"{summary['hit1_percent']:.4f}",
					"f1_percent": f"{summary['f1_percent']:.4f}",
					"avg_total_tokens": f"{summary['avg_total_tokens']:.2f}",
					"avg_llm_calls": f"{summary['avg_llm_calls']:.2f}",
					"missing_cost_rows": summary["missing_cost_rows"],
					"result_dir": str(result_dir),
				}
			)

			counter = Counter(payload["distribution"]["call_counts"])
			for llm_call_count, count in sorted(counter.items()):
				dist_rows.append(
					{
						"dataset": dataset,
						"method": method,
						"llm_call_count": llm_call_count,
						"count": count,
						"proportion": f"{count / total_samples:.8f}",
					}
				)

	if not dist_map:
		raise RuntimeError("No valid result directory found for plotting.")

	figure_path = output_dir / f"llm_call_distribution_{model}.png"
	csv_path = output_dir / f"summary_{model}.csv"
	dist_csv_path = output_dir / f"llm_call_distribution_{model}.csv"

	plot_distributions(
		dist_map=dist_map,
		datasets=datasets,
		methods=methods,
		output_path=figure_path,
		model_name=model,
		dpi=DEFAULT_DPI,
		focus_min=DEFAULT_CALL_FOCUS_MIN,
		focus_max=DEFAULT_CALL_FOCUS_MAX,
	)
	save_summary_csv(summary_rows=summary_rows, output_path=csv_path)
	save_distribution_csv(dist_rows=dist_rows, output_path=dist_csv_path)

	print("\nGenerated files:")
	print(f"- Figure: {figure_path}")
	print(f"- Summary: {csv_path}")
	print(f"- Distribution: {dist_csv_path}")


if __name__ == "__main__":
	main()
