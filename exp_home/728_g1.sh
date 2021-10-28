#!/bin/bash

seed=('2020' '2525' '3030')
exp_val=('exp3')
exp_str=('exp1')
src=('1')
g='1'

for e in ${exp_val[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# Scd lamb = 0.02 on ba_src_val
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src_val \
--output ./seed${s}/${e}/my/isfda_val_scd0_02 \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--scd_lamb 0.02 \
--intra_dense \
--inter_sep \
--seed ${s} &&

echo "================================"

done
done
done &&



for e in ${exp_str[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# Scd lamb = 0.02 on ba_src_stratified
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda_val_scd0_02 \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--scd_lamb 0.02 \
--intra_dense \
--inter_sep \
--seed ${s} &&

echo "================================"

done
done
done







