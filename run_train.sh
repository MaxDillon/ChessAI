#!/bin/bash

while true; do
    ./mi_play.sh chess2 \
		 -n 10 \
		 -white "model:model.chess2.Model010.*" \
		 -witer 250 -wexpl 0.3 -wtemp 0.1 \
		 -black "model:model.chess2.Model010.*" \
		 -biter 250 -bexpl 0.3 -btemp 0.1 \
         > /dev/null

    if [[ "$1" == "lead" ]]; then
        ./reshuffle.sh
        ./mi_train.sh chess2 \
		              -data shuffled.chess2 \
		              -from $(ls -t model.chess2.Model010.* | head -n 1) \
		              -drawweight 0.1 \
		              -batch 800 \
		              -lastn 20 \
		              -updates 1
    fi
done


