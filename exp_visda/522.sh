#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp2')
g='2'
p='/media/room/date/xinhao_li/ISFDA-exp/exp_visda'

# src
python ../object/image_source.py \
--trte full \
--output ./seed2020/exp2/baseline/src \
--da uda \
--gpu_id ${g} \
--dset VISDA-RSUT \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--seed 2020 &&

for e in ${exp[@]}
do
for s in ${seed[@]}
do

# shot
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/baseline/src \
--output ${p}/seed${s}/${e}/baseline/shot \
--net resnet101 \
--lr 1e-3  \
--seed ${s} &&

# ba src use 2020's

# ba shot
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/my/ba_src \
--output ${p}/seed${s}/${e}/my/ba_shot \
--net resnet101 \
--lr 1e-3  \
--seed ${s} &&

# topk ent
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/my/ba_src \
--output ${p}/seed${s}/${e}/my/topk_e \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--seed ${s} &&

echo "==================================="


done
done
