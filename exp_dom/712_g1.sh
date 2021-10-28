#!/bin/bash

# Test Dom by seed 2020, 2525, 3030
# GPU0: gamma=0.0002
# GPU1: gamma=0
# This script is to test whether smaller gamma=0 is more suitable for S->C/P/R

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

# ISFDA w/o Ba
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/my/isfda_wo_ba \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep \
--seed ${s}

echo ${sid}" Done!" &&
echo "================================" &&
echo ""

done
done
done
