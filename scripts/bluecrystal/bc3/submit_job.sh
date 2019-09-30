#!/bin/bash

JOBID=$(qsub scripts/bixler_perching.job)

echo Altering output to: output/${JOBID%%[.\[]*}/stdout.log

qalter -o output/${JOBID%%[.\[]*}/stdout.log \
       -e output/${JOBID%%[.\[]*}/stderr.log \
       ${JOBID}
