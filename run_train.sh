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
            -white "model1:model.chess2.Model011.*" \
            -witer 600 -wexpl 0.2 -wtemp 0.1 \
            -black "model1:model.chess2.Model011.*" \
            -biter 600 -bexpl 0.2 -btemp 0.1 \
            > /dev/null
    fi
    
    if [[ $lead != 0 ]]; then
        ./fetch.sh
        ./reshuffle.sh
        ./mi_train.sh chess2 \
            -data shuffled.chess2 \
            -from $(ls -t model.chess2.Model011.* | head -n 1) \
            -valuemult 0.95 \
            -metf 0.7 \
            -lastn 30 \
            -batch 1000 \
            -drawweight 0.2 \
            -saveevery 1000 \
            -updates 1
        ./fetch.sh
    fi
done
