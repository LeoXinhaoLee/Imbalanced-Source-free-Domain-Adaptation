#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp5' 'exp6')
for e in ${exp[@]}
do
for s in ${seed[@]}
do

# src
python ../object/image_source.py \
--trte val --output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id 1 \
--dset VISDA-RSUT \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--seed ${s}  &&


# shot
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 1 \
--s 0 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/baseline/shot \
--net resnet101 \
--lr 1e-3  \
--seed ${s}


done
done