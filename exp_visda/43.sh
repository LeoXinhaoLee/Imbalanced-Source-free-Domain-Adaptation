#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp6')
g='1'

for e in ${exp[@]}
do
for s in ${seed[@]}
do

:<<BLOCK
# my ba_topk_e
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_e \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk_ent

# my ba_topk_e_scd
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_e_scd \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk_ent \
--scd_label &&

# my ba_topk_e_scd_dense_2
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
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
BLOCK

# my ba_topk_e_scd_dense_2
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/intra \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk_ent \
--scd_label \
--intra_dense &&

python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/inter \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk_ent \
--scd_label \
--inter_sep


done
done