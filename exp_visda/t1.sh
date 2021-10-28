#!/bin/bash
#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2525' '3030')
exp=('exp1' 'exp2')
for s in ${seed[@]}
do
for e in ${exp[@]}
do

# target default epoch: 15

# my ba_topk_scd
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 1 \
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
--gpu_id 1 \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_scd_dense \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk \
--scd_label \
--intra_dense &&

# my ba_topk_scd_dense_post_fit
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id 1 \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_scd_dense_post_fit \
--net resnet101 \
--lr 1e-3  \
--seed ${s} \
--topk \
--scd_label \
--intra_dense \
--post_fit

done
done