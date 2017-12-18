#!/bin/bash

INSTR=./tools/memtime_wrapper.py
EXE=./tools/mip_lal_runner.py

MAX_MEM=$((20*1024*1024))
MAX_TIME=7200

trap "exit" INT
for prop in $(find ./planet/benchmarks/ACAS/ -name "*.rlv" | sort);
do
    target=$(echo $prop | gawk 'match($0, /(property[0-9]+\/.+)\.rlv/, arr) {print "results/ACAS/MIPLAL/" arr[1] ".txt"}')
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $prop "
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $prop
    fi
done
