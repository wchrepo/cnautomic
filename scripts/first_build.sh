#!/bin/bash
set -o errexit #exit when error

WORKSPACE=$1
OUTPUT_FILENAME=`date +%Y%m%d_%a_%H:%M:%S` 
## CONFIG ##
CATEGORIES=("voluntary_occurences" "involuntary_occurences" "states")
ITERATION_CONFIG="config.iteration.default"
## END CONFIG ##
TRIPLE_PATHS=()
for CATEGORY in ${CATEGORIES[@]}
do
    #head
    bash scripts/heads.sh ${WORKSPACE} ${CATEGORY} ${OUTPUT_FILENAME}
    bash scripts/triples.sh ${WORKSPACE} ${CATEGORY} \
     "results/${WORKSPACE}/heads/${CATEGORY}/processed_${OUTPUT_FILENAME}.jsonl.txt" \
     ${OUTPUT_FILENAME}
    TRIPLE_PATHS+=("results/${WORKSPACE}/triples/${CATEGORY}/${OUTPUT_FILENAME}/")
done

python prepare_iteration.py ${TRIPLE_PATHS[@]} --workspace ${WORKSPACE} \
    --cookbook ${ITERATION_CONFIG}
