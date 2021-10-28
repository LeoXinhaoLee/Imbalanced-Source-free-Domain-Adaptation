#!/bin/bash

# Test Dom by seed 2020, 2525, 3030
# Src: c/p/r
# GPU0: C,P
# GPU1: R
# This script is to test ISFDA w/o Ba gamma=0.02 on src: C, P, R

g='0'
seed=('2020' '2525' '3030')
exp=('exp1')
src=('0' '1')

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# ISFDA w/o Ba src, gamma=0.02
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/my/isfda_wo_ba_gamma_0_02 \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep \
--scd_lamb 0.02 \
--seed ${s} &&

echo ${sid}" Done!" &&
echo "================================" &&
echo ""

done
done
done
