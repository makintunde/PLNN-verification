#!/bin/bash

INSTR=./tools/memtime_wrapper.py
EXE=./tools/reverify_runner.py

MAX_MEM=$((20*1024*1024))
MAX_TIME=7200



coll_idx=1
for prop in $(find ./planet/benchmarks/collisionDetection/ -name "*.rlv"| sort);
do
    target_fname=$coll_idx-$(basename $prop .rlv)
    target="results/collisionDetection/reverify/$target_fname.txt"
    if [ ! -f $target ]; then
        echo "$I/NSTR $EXE $MAX_MEM $MAX_TIME $target $METHOD $prop"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $METHOD $prop
    fi
    coll_idx=$(($coll_idx + 1))
done


