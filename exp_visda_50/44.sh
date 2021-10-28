#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp1')
g='1'

for e in ${exp[@]}
do
for s in ${seed[@]}
do

# my ba_src
python ../object/image_source.py \
--trte val --output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset VISDA-RSUT-50 \
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
--dset VISDA-RSUT-50 \
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