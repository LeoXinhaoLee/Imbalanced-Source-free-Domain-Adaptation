#!/bin/bash

seed=('2020' '2525' '3030')
exp=('exp6')
src=('1' '2' '3')
g='0,2'
p='/media/room/date/xinhao_li/ISFDA-exp/exp_home'

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s ${sid} \
--output_src ${p}/seed2020/exp1/my/ba_src \
--output ${p}/seed${s}/${e}/my/four_lb \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep \
--third_label \
--fourth_label \
--seed ${s} \
--paral &&

echo "================================"

done
done
done &&

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id 2 \
--s ${sid} \
--output_src ${p}/seed2020/exp1/my/ba_src \
--output ${p}/seed${s}/${e}/my/three_lb \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep \
--third_label \
--seed ${s} &&

echo "================================"

done
done
done
