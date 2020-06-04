#!/bin/bash

for isim in 140 141 142 143 144 145 146 147 148 149 150 151 152 153 170
do
    echo $isim
    python3 reduce_bh_data.py $isim &
    
    if [[ $((i % 5)) -eq 4 ]]; then
	wait
    fi
done

wait
