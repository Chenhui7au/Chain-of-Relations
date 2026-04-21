#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   efficiency.py
@Time    :   2026/04/06 15:17:14
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import glob
import json
import os


def analyze_detailed_results(result_dir, sample_ids=None):
    """Analyze detailed JSON results to compute LLM call statistics.

    If sample_ids is provided, only include detailed records whose sample id
    matches the given set.
    """
    # Get all JSON files except predict.json and predict.jsonl
    json_files = glob.glob(os.path.join(result_dir, "*.json"))
    json_files = [f for f in json_files if not f.endswith("predict.json")]

    sample_ids_set = None
    if sample_ids is not None:
        sample_ids_set = {str(sid).strip() for sid in sample_ids if str(sid).strip()}

    total_llm_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    file_count = 0

    for json_file in json_files:
        try:
            # Fast path: many result dirs store one sample per <sample_id>.json
            if sample_ids_set is not None:
                file_stem = os.path.splitext(os.path.basename(json_file))[0]
                file_stem_match = file_stem in sample_ids_set
            else:
                file_stem_match = True

            with open(json_file, 'r') as f:
                data = json.load(f)

            if sample_ids_set is not None:
                data_id = str(data.get("id", "")).strip()
                if not file_stem_match and data_id not in sample_ids_set:
                    continue

            # Count LLM calls and tokens from step_history
            step_history = data.get("step_history", [])
            llm_calls = 0
            input_tokens = 0
            output_tokens = 0

            for step in step_history:
                if step.get("step_type") == "llm":
                    llm_calls += 1
                    input_tokens += step.get("input_tokens", 0)
                    output_tokens += step.get("output_tokens", 0)

            total_llm_calls += llm_calls
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            file_count += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    if file_count == 0:
        return None

    return {
        "avg_llm_calls": total_llm_calls / file_count,
        "avg_input_tokens": total_input_tokens / file_count,
        "avg_output_tokens": total_output_tokens / file_count,
        "avg_total_tokens": (total_input_tokens + total_output_tokens) / file_count,
        "total_queries": file_count,
        "total_llm_calls": total_llm_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens
    }
