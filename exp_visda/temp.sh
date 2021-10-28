#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2525' '3030')
exp=('exp3')
for s in ${seed[@]}
do
for e in ${exp[@]}
do

:<<BLOCK
# baseline shot
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 0 \
--s 0 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/baseline/shot \
--net resnet101 \
--lr 1e-3  \
--seed ${s} &&

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

BLOCK

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

# my ba_topk balance sample on T
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 0 \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_ba_topk \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk \
--t_balanced


done
done