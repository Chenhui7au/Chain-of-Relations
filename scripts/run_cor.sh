#!/usr/bin/env bash
DATASET="webqsp"
KG="freebase"

python -m chain_of_relations.run \
	--method cor \
	--dataset "$DATASET" \
	--kb "$KG" \
    --run_size 1
