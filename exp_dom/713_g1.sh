#!/bin/bash

# Test Dom by seed 2020, 2525, 3030
# GPU0: C, P
# GPU1: R
# This script is to test whether smaller gamma=0.02

g='1'
seed=('2020' '2525' '3030')
exp=('exp1')
src=('2')

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# ISFDA gamma=0.02
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda_gamma_0_02 \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep \
--scd_lamb 0.02 \
--seed ${s}

echo ${sid}" Done!" &&
echo "================================" &&
echo ""

done
done
done
