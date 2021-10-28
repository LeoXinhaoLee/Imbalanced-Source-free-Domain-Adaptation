#!/bin/bash

seed=('2525' '3030')
exp=('exp1')
src=('1' '2' '3')
g='1'

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

# All use Src model trained by 2020
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed2020/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda_src2020 \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep \
--seed ${s} &&

echo "================================"

done
done
done

