#!/bin/bash

dataset='weather_4'
stride_tuple=(1 5 10)
iid_tuple=(0 1 2)

for iid in ${iid_tuple[@]}
do
    for stride in ${stride_tuple[@]}
    do
        python cnn_regressor/oned.py --epochs 100 --norm-type 2 --batch-size 512 --label-type 0 --root-img-path /media/lepoeme20/Data/projects/daewoo/ --trn-dataset $dataset --iid $iid --stride $stride
    done
done
