#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2525' '3030')
exp=('exp4')
for s in ${seed[@]}
do
for e in ${exp[@]}
do

# my ba_src
python ../object/image_source.py \
--trte val --output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id 0 \
--dset VISDA-RSUT \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--seed ${s} \
--source_balanced &&

# target default epoch: 15

# my ba shot
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 0 \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba \
--net resnet101 \
--lr 1e-3  \
--seed ${s} &&

# my ba_topk
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 0 \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk &&

# my ba_topk_scd
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 0,1 \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_scd \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk \
--scd_label &&

# my ba_topk_scd_dense
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 0,1 \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_scd_dense \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk \
--scd_label \
--intra_dense


done
done