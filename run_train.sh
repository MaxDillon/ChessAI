#!/bin/bash

lead=0
skip=0
fast=0
dest=/dev/null
device=0

for arg in "$@"; do
    case $arg in
        -lead)  lead=1; shift;;
        -skip)  skip=1; shift;;
        -fast)  fast=1; shift;;
        -show)  dest=/dev/stdout; shift;;
        -dev1)  device=1; shift;;
    esac
done

while true; do
    if [[ $skip == 0 ]]; then
        if [[ $fast == 0 ]]; then
            ./mi_play.sh chess2 \
                -n 10 \
                -white "model2:model.chess2.Model015c.*" \
                -witer 400 -wexpl 0.15 -wtemp 0.3 -wpexp 2.0 -wunif 1.0 \
                -black "model2:model.chess2.Model015c.*" \
                -biter 400 -bexpl 0.15 -btemp 0.3 -bpexp 2.0 -bunif 1.0 \
		-device $device \
                > $dest
        else
            ./mi_play.sh chess2 \
                -n 25 \
                -white "model2:model.chess2.Model015c.*" \
                -witer 1 -wtemp 1.0 \
                -black "model2:model.chess2.Model015c.*" \
                -biter 800 -bexpl 0.15 -btemp 0.3 \
                -one false \
                -fast true \
                -mindepth 30 \
                -rollback 6 \
		-device $device \
                > $dest
        fi
    fi
    
    if [[ $lead != 0 ]]; then
        ./fetch.sh
        ./reshuffle.sh
        ./mi_train.sh chess2 \
            -data shuffled.chess2 \
            -from $(ls -t model.chess2.Model015c.* | head -n 1) \
            -valuemult 0.95 \
            -metf 0.7 \
            -lastn 150 \
            -batch 350 \
            -drawweight 0.2 \
            -saveevery 1000 \
            -updates 1 \
            -rate 5e-3
    fi

    if [ -e exit_run_train ]; then
	exit
    fi
done
