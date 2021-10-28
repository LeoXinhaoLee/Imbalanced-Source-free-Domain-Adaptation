#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp1')
g='1'
dset='VISDA-RSUT-10'

for e in ${exp[@]}
do
for s in ${seed[@]}
do

# src
python ../object/image_source.py \
--trte val --output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id ${g} \
--dset ${dset} \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--seed ${s} &&

# shot
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset ${dset} \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/baseline/shot \
--net resnet101 \
--lr 1e-3  \
--seed ${s} &&


# my ba_src
python ../object/image_source.py \
--trte val --output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset ${dset} \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--seed ${s} \
--source_balanced &&

# target default epoch: 15
# my ba_topk_scd_dense
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset ${dset} \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_e_scd_dense_2 \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep


done
done