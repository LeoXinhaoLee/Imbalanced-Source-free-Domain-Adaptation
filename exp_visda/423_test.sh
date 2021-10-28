#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020')
exp=('exp1')
g='0,2'

for e in ${exp[@]}
do
for s in ${seed[@]}
do

<<BLOCK
# my ba_topk_e_scd_dense_2
python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/my/ba_topk_e_scd_mdd \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep
BLOCK

python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/my/ba_topk_e_scd_third_mdd \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk_ent \
--scd_label \
--third_label \
--intra_dense \
--inter_sep


done
done