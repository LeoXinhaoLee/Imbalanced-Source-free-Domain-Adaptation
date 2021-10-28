#!/bin/bash

g='0,2'
seed=('2020' '2525' '3030')
exp=('exp1')
src=('0' '1' '2' '3')
p='/media/room/date/xinhao_li/ISFDA-exp/exp_dom'

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do


if [[ ${sid} == '0' && ${s} == '2020' ]]
then
continue

else

if [[ ${sid} != '3' ]]
then
source_file='my/ba_src'
else
source_file='baseline/src'
fi

python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/exp2/${source_file} \
--output ${p}/seed${s}/${e}/my/four_lb \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--third_label \
--fourth_label \
--intra_dense \
--inter_sep \
--paral \
--seed ${s} &&

python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/exp2/${source_file} \
--output ${p}/seed${s}/${e}/my/three_label \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--third_label \
--intra_dense \
--inter_sep \
--paral \
--seed ${s}

fi &&

echo "================================"


done
done
done
