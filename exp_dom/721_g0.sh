#!/bin/bash

# Test Dom by seed 2020
# Src: c/p/r
# GPU0: C,P, R, S
# This script is to test ISFDA curriculum based on Entropy instead of Prob(previous choice)
# Src: C
# Since its the initial trial, only use seed 2020 and Src = C

g='0'
seed=('2020')
exp=('exp1')
src=('1' '2' '3')

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# ISFDA use curriculum based on Entropy, gamma=0.02(new default, better than 0.2)
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda_unmask \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep \
--no_mask \
--seed ${s} &&


echo ${sid}" Done!" &&
echo "================================" &&
echo ""

done
done
done
