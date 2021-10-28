#!/bin/bash


#GPU 1: baseline: src+shot, my: ba_src, ba_topk_e_scd_dense_2
#seed=('2020' '2525' '3030')
seed=('3030' '2525')
exp=('exp1')
g='1'

for e in ${exp[@]}
do
for s in ${seed[@]}
do

<<BLOCK
# src
python ../object/image_source.py \
--trte val --output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id ${g} \
--dset VISDA-RSUT-50 \
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
--dset VISDA-RSUT-50 \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/baseline/shot \
--net resnet101 \
--lr 1e-3  \
--seed ${s} &&
BLOCK


if [ ${s} != "2020" ]
then

if [ ${s} == "3030" ]
then
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
# my ba_topk_e_scd_dense_2
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

else
# my ba_topk_e_scd_dense_2
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

fi

fi

done
done