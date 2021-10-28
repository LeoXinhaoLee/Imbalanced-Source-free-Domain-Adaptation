#!/bin/bash

g='1'
seed=('2525' '3030')
exp=('exp2')
src=('0' '1' '2' '3')

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

if [[ ${s} == '2525' && (${sid} == '0' || ${sid} == '1') ]]
then

continue

else

if [[ ${sid} != '3' ]]
then
python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep

else
python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/${e}/baseline/src \
--output ./seed${s}/${e}/my/isfda \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep
fi

fi
done
done
done
