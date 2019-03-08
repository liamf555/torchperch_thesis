#!/bin/bash

START_NUM=$1
END_NUM=$2

JOBID=$(qsub -t ${START_NUM}-${END_NUM} scripts/bixler_perching.job)

BASE_JOBID=${JOBID%%[.\[]*}

for i in $(seq ${START_NUM} ${END_NUM}); do
    qalter -o output/${BASE_JOBID}-$i/stdout.log \
           -e output/${BASE_JOBID}-$i/stderr.log \
           ${BASE_JOBID}[$i]
done
