#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp1' 'exp2')
g='2'

<<BLOCK
for e in ${exp[@]}
do
# ba src
python ../object/image_source.py \
--trte full \
--output ./seed2020/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset VISDA-RSUT \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--source_balanced \
--seed 2020
done
BLOCK

for e in ${exp[@]}
do
for s in ${seed[@]}
do
# isfda
python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--max_epoch 15 \
--s 0 \
--output_src ./seed2020/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda \
--net resnet101 \
--lr 1e-3 \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep \
--seed ${s}

done
done
