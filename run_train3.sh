#!/bin/bash

while true; do
    python src/main/py/maximum/industries/play.py \
	   -n 10 \
	   -m tfmodels/r14/1549380964 \
	   -a iter=400,ppom=0.7,vilo=0.7,bpwl=1,toak=1,expl=0.8,pexp=1.0,unif=0.0,temp=0.2,ramp=10
    
    if [ -e exit_run_train ]; then
	exit
    fi
done
