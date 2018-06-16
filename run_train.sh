#!/bin/bash

lead=0
skip=0

for arg in "$@"; do
    case $arg in
        -lead)  lead=1; shift;;
        -skip)  skip=1; shift;;
    esac
done

while true; do
    if [[ $skip == 0 ]]; then
        ./mi_play.sh chess2 \
            -n 10 \
            -white "model1:model.chess2.Model012.*" \
            -witer 300 -wexpl 0.3 -wtemp 0.3 -wpexp 2.0 -wunif 1.0 \
            -black "model1:model.chess2.Model012.*" \
            -biter 300 -bexpl 0.3 -btemp 0.3 -bpexp 2.0 -bunif 1.0 \
            > /dev/null
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
            -batch 600 \
            -drawweight 0.2 \
            -saveevery 1000 \
            -updates 1
    fi
done
