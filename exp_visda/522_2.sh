#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp2')
g='1'
p='/media/room/date/xinhao_li/ISFDA-exp/exp_visda'

for e in ${exp[@]}
do
for s in ${seed[@]}
do

# curriculum
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/my/ba_src \
--output ${p}/seed${s}/${e}/my/topk_e_scd \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--scd_label \
--seed ${s} &&

# dense
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/my/ba_src \
--output ${p}/seed${s}/${e}/my/dense \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--scd_label \
--intra_dense \
--seed ${s} &&

# sep
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--gpu_id ${g} \
--s 0 \
--output_src ./seed2020/${e}/my/ba_src \
--output ${p}/seed${s}/${e}/my/sep \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--scd_label \
--inter_sep \
--seed ${s} &&

echo "==================================="


done
done
