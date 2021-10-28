#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp1')
g='1'



for e in ${exp[@]}
do
for s in ${seed[@]}
do

# src
python ../object/image_source.py \
--trte stratified \
--output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset VISDA-RSUT \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--source_balanced \
--seed ${s} &&


# isfda
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_seq \
--seed ${s} &&

echo "==================================="


done
done
