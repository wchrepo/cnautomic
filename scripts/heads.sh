#!/bin/bash
set -o errexit #exit when error

WORKSPACE=$1 #
CATEGORY=$2 #voluntary_occurences involuntary_occurences states
OUTPUT_FILENAME=$3
## CONFIG ##
HEAD_CONFIG="config/head/default.json"
HEAD_COOKBOOK="config.head_post.default"

## generate head ##
python generate_heads.py ${CATEGORY} --config ${HEAD_CONFIG} \
    --workspace ${WORKSPACE} --output_filename ${OUTPUT_FILENAME}

## postprocess head ##
python postprocess_heads.py \
    "results/${WORKSPACE}/heads/${CATEGORY}/${OUTPUT_FILENAME}.jsonl.txt" \
    --cookbook ${HEAD_COOKBOOK} --score_ratio 0.7 