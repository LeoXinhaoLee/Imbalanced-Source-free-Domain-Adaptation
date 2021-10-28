#!/bin/bash

g='1'
seed=('2525' '3030')
exp=('exp1')
src=('1' '2' '3')
p='/media/room/date/xinhao_li/ISFDA-exp/exp_dom'


python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s 3 \
--output_src ./seed2020/exp2/baseline/src \
--output ${p}/seed2020/exp1/my/unmask \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--no_mask \
--intra_dense \
--inter_sep \
--seed 2020 &&


for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

<<BLOCK
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/exp2/my/ba_src \
--output ${p}/seed${s}/${e}/my/four_lb \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--third_label \
--fourth_label \
--intra_dense \
--inter_sep \
--seed ${s} &&

python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/exp2/my/ba_src \
--output ${p}/seed${s}/${e}/my/three_label \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--third_label \
--intra_dense \
--inter_sep \
--seed ${s}
BLOCK

if [[ ${sid} != '3' ]]
then
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/exp2/my/ba_src \
--output ${p}/seed${s}/${e}/my/unmask \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--no_mask \
--intra_dense \
--inter_sep \
--seed ${s}

else
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/exp2/baseline/src \
--output ${p}/seed${s}/${e}/my/unmask \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--no_mask \
--intra_dense \
--inter_sep \
--seed ${s}

fi &&

echo "================================"


done
done
done
