#!/bin/bash

# GPU 1: src, SHOT, SHOT+SLC
# This script is to test SHOT v.s. SHOT+SLC on VisDA-Knockout to verify the universality of SLC
# For this is an initial trial test, only use seed 2020
# Src: train --> Tar: validation (knocked out)

seed=('2020')
exp=('exp1')
g='1'

for e in ${exp[@]}
do
for s in ${seed[@]}
do

# SHOT+SLC
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-Knockout \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/baseline/src \
--output ./seed${s}/${e}/my/shot_SLC_gamma0_01_nodecay \
--net resnet101 \
--lr 1e-3  \
--scd_label \
--scd_lamb 0.01 \
--seed ${s} &&

echo "==================================="

done
done
