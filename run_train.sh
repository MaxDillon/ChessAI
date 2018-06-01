#!/bin/bash

while true; do
    ./mi_play.sh chess2 \
        -n 10 \
		-white "model1:model.chess2.Model010.*" \
		-witer 250 -wexpl 0.2 -wtemp 0.1 \
		-black "model1:model.chess2.Model010.*" \
		-biter 250 -bexpl 0.2 -btemp 0.1 \
        > /dev/null
    
    if [[ "$1" == "lead" ]]; then
        ./fetch.sh
        ./reshuffle.sh
        ./mi_train.sh chess2 \
		    -data shuffled.chess2 \
		    -from $(ls -t model.chess2.Model010.* | head -n 1) \
            -valuemult 0.95 \
            -metf 0.7 \
		    -lastn 15 \
		    -batch 500 \
		    -drawweight 0.2 \
            -saveevery 1000 \
		    -updates 1
    fi
done
