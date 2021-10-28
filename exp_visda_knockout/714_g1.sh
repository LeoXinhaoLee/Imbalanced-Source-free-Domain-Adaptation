#!/bin/bash

# GPU 1: src, SHOT, SHOT+SLC
# This script is to test SHOT v.s. SHOT+SLC on VisDA-Knockout to verify the universality of SLC
# For this is an initial trial test, only use seed 2020
# Src: train --> Tar: validation (knocked out)

seed=('2020')
exp=('exp1')
g='1'

if false
then
# src
python ../object/image_source.py \
--trte stratified \
--output ./seed2020/exp1/baseline/src \
--da uda \
--gpu_id ${g} \
--dset VISDA-Knockout \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--seed 2020
fi

for e in ${exp[@]}
do
for s in ${seed[@]}
do


# shot
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-Knockout \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/baseline/src \
--output ./seed${s}/${e}/baseline/shot \
--net resnet101 \
--lr 1e-3  \
--seed ${s} &&

# SHOT+SLC
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-Knockout \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/baseline/src \
--output ./seed${s}/${e}/my/shot_SLC \
--net resnet101 \
--lr 1e-3  \
--scd_label \
--seed ${s} &&

echo "==================================="

done
done
