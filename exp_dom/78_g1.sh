#!/bin/bash

# Test Dom by seed 2525, 3030
# S->C/P/R are both tested with/without src balanced sampler
# GPU0: src: 0, 1
# GPU1: src: 2, 3

g='1'
seed=('2525' '3030')
exp=('exp1')
src=('2' '3')

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# Source Model with Balanced Sampling
python ../object/image_source.py \
--trte stratified \
--output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset domainnet \
--max_epoch 20 \
--s ${sid} \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--source_balanced \
--seed ${s} &&

# ISFDA: primary & secondary label
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep \
--seed ${s} &&

if [[ ${sid} == '3' ]]
then
# Source Model without Balanced Sampling
python ../object/image_source.py \
--trte stratified \
--output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id ${g} \
--dset domainnet \
--max_epoch 20 \
--s ${sid} \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--seed ${s} &&

# ISFDA: primary & secondary label
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

fi &&

echo ${sid}" Done!" &&
echo "================================" &&
echo ""

done
done
done
