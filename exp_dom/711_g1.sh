#!/bin/bash

# Test Dom by seed 2020, 2525, 3030
# Src: C, P, R
# GPU0: src: 0, 1
# GPU1: src: 2

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

# ISFDA on src model w/o Ba, Gamma=0.2*0.1=0.002
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/my/isfda_wo_ba_gamma002 \
--net resnet50 \
--lr 1e-2  \
--scd_lamb 0.002 \
--seed ${s} &&

# ISFDA on src Ba model
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda_gamma002 \
--net resnet50 \
--lr 1e-2  \
--scd_lamb 0.002 \
--seed ${s} &&


echo ${sid}" Done!" &&
echo "================================" &&
echo ""

done
done
done
