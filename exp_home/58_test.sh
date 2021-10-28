#!/bin/bash

#GPU 0: baseline: src+shot, my: ba_src, ba(shot), ba_topk
seed=('2020' '2525' '3030')
exp=('exp5')
src=('1' '2' '3')
g='2'

for e in ${exp[@]}
do
for sid in ${src[@]}
do
# ba src
python ../object/image_source.py \
--output ./seed2020/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset office-home-RSUT \
--max_epoch 60 \
--s ${sid} \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--trte full \
--source_balanced \
--seed 2020

done
done

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
--output_src ./seed2020/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda_src2020_full \
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