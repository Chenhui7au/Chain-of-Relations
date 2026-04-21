#!/usr/bin/env bash
DATASET="webqsp"
MODEL_DIR="gpt-4.1-mini"

python -m chain_of_relations.eval.eval \
	--dataset "$DATASET" \
	--output_dir "results/cor/${DATASET}/${MODEL_DIR}"

