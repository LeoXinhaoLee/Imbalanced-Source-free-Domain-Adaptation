#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2525' '3030')
exp=('exp1')
src=('1' '2' '3')
g='0'

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/exp2/my/ba_src \
--output ./seed${s}/${e}/my/isfda_src2020_2 \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep \
--seed ${s}

done
done
done