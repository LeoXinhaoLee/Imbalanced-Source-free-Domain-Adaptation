#!/bin/bash

# GPU 1: src, SHOT, SHOT+SLC
# This script is to test SHOT v.s. SHOT+SLC on VisDA-Beta to verify the universality of SLC
# Also test ISFDA as a whole here
# For this is an initial trial test, only use seed 2020
# Src: train_b1_a1 --> Tar: validation_b2.0_a1.2/2.0/2.7

seed=('2020')
exp=('exp1')
g='1'

# src
python ../object/image_source.py \
--trte stratified \
--output ./seed2020/exp1/baseline/src \
--da uda \
--gpu_id ${g} \
--dset VISDA-Beta \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--seed 2020 &&

# ba src
python ../object/image_source.py \
--trte stratified \
--output ./seed2020/exp1/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset VISDA-Beta \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--source_balanced \
--seed 2020 &&

for e in ${exp[@]}
do
for s in ${seed[@]}
do

# shot
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-Beta \
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
--dset VISDA-Beta \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/baseline/src \
--output ./seed${s}/${e}/my/shot_SLC \
--net resnet101 \
--lr 1e-3  \
--scd_label \
--seed ${s} &&

# ISFDA
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-Beta \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda \
--net resnet101 \
--lr 1e-3  \
--scd_label \
--topk_ent \
--intra_dense \
--inter_sep \
--seed ${s} &&

echo "==================================="


done
done
