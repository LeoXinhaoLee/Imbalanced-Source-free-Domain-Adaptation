#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020')
exp=('exp2')
#src=('1' '2' '3')
src=('3')
g='2'

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

<<BLOCK
# ba src
python ../object/image_source.py \
--output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset office-home-RSUT \
--max_epoch 60 \
--s ${sid} \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--source_balanced \
--seed ${s} &&
BLOCK

python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep \
--seed ${s}

done
done
done