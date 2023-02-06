#!/bin/bash
set -o errexit #exit when error


WORKSPACE=$1 #
CATEGORY=$2 #voluntary_occurences involuntary_occurences states
SOURCE=$3 #heads file
OUTPUT_FILENAME=$4
## CONFIG ##
TRIPLE_CONFIG="config/triple/${CATEGORY}.json"
HEAD_FILTER_PATH="critic_model/head/"
TAIL_FILTER_PATH="critic_model/tail/"
TRIPLE_FILTER_PATH="critic_model/triple/"
HEAD_FILTER_DEVICE="cuda:0"
TAIL_FILTER_DEVICE="cuda:1"
TRIPLE_FILTER_DEVICE="cuda:2"

## generate triples ##
python generate_triples.py \
    ${SOURCE} \
    --config ${TRIPLE_CONFIG} \
    --workspace ${WORKSPACE} --output_filename ${OUTPUT_FILENAME} --debug

## filter triples ##
python filter_triples.py \
    "results/${WORKSPACE}/triples/${CATEGORY}/${OUTPUT_FILENAME}/" \
    --config config/triple_post/default.json \
    --head_filter_path ${HEAD_FILTER_PATH} --head_filter_device ${HEAD_FILTER_DEVICE} \
    --tail_filter_path ${TAIL_FILTER_PATH} --tail_filter_device ${TAIL_FILTER_DEVICE} \
    --triple_filter_path ${TRIPLE_FILTER_PATH} --triple_filter_device ${TRIPLE_FILTER_DEVICE} 