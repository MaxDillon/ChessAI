#!/bin/bash

device=0

for arg in "$@"; do
    case $arg in
        -dev0)  device=0; shift;;
        -dev1)  device=1; shift;;
        -dev2)  device=2; shift;;
        -dev3)  device=3; shift;;
    esac
done

while true; do
    ./mi_play.sh chess2 \
                 -n 10 \
                 -white "model2:prod_model.chess2" \
                 -wargs iter=400,expl=0.15,temp=0.3,pexp 2.0,unif=1.0 \
                 -black "model2:prod_model.chess2" \
                 -wargs iter=400,expl=0.15,temp=0.3,pexp 2.0,unif=1.0 \
		 -device $device \
                 > /dev/null
    
    if [ -e exit_run_train ]; then
	exit
    fi
done
