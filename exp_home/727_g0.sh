#!/bin/bash

seed=('2020' '2525' '3030')
exp=('exp3')
src=('1')
g='0'

for e in ${exp[@]}
do
for s in ${seed[@]}
do
for sid in ${src[@]}
do

if [[ ${s} != '2020' ]]
then
# ba src
python ../object/image_source.py \
--output ./seed${s}/${e}/my/ba_src_val \
--trte val \
--da uda \
--gpu_id ${g} \
--dset office-home-RSUT \
--max_epoch 60 \
--s ${sid} \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--source_balanced \
--seed ${s}
fi &&

python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src_val \
--output ./seed${s}/${e}/my/isfda_val \
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