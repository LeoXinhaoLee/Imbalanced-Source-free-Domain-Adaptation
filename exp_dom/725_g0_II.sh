#!/bin/bash

# Test Dom by seed 2020
# Src: c/p/r
# GPU0: C, P, R, S
# This script is to test ISFDA with only Primary Label
# Since its the initial trial, only use seed 2020

g='0'
seed=('2525' '3030')
exp=('exp1')
src=('2' '3')

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# ISFDA with Primary Label only
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda_lb_1 \
--net resnet50 \
--lr 1e-2  \
--topk \
--intra_dense \
--inter_sep \
--seed ${s} &&


echo ${sid}" Done!" &&
echo "================================" &&
echo ""

done
done
done
