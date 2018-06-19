#!/bin/bash

lead=0
skip=0
fast=0
dest=/dev/null

for arg in "$@"; do
    case $arg in
        -lead)  lead=1; shift;;
        -skip)  skip=1; shift;;
        -fast)  fast=1; shift;;
        -show)  dest=/dev/stdout; shift;;
    esac
done

while true; do
    if [[ $skip == 0 ]]; then
        if [[ $fast == 0 ]]; then
            ./mi_play.sh chess2 \
                -n 10 \
                -white "model2:model.chess2.Model012.*" \
                -witer 400 -wexpl 0.3 -wtemp 0.3 -wpexp 2.0 -wunif 1.0 \
                -black "model2:model.chess2.Model012.*" \
                -biter 400 -bexpl 0.3 -btemp 0.3 -bpexp 2.0 -bunif 1.0 \
                > $dest
        else
            ./mi_play.sh chess2 \
                -n 10 \
                -white "model2:model.chess2.Model012.*" \
                -witer 1 -wtemp 1.0 \
                -black "model2:model.chess2.Model012.*" \
                -biter 800 -bexpl 0.3 -btemp 0.3 \
                -one false \
                -fast true \
                -mindepth 30 \
                -rollback 6 \
                > $dest
        fi
    fi
    
    if [[ $lead != 0 ]]; then
        ./fetch.sh
        ./reshuffle.sh
        ./mi_train.sh chess2 \
            -data shuffled.chess2 \
            -from $(ls -t model.chess2.Model012.* | head -n 1) \
            -valuemult 0.95 \
            -metf 0.7 \
            -lastn 50 \
            -batch 500 \
            -drawweight 0.2 \
            -saveevery 1000 \
            -updates 1
    fi
done
