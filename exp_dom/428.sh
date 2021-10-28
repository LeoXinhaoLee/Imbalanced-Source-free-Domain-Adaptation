#!/bin/bash

### Run on DomainNet: Clipart->Painting, Real, Sketch
g='1'
s='2525'
e='exp1'
src=('1' '2')

for sid in ${src[@]}
do

# Source Model with Balanced Sampling
python ../object/image_source.py \
--trte val \
--output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id 1 \
--dset domainnet \
--max_epoch 20 \
--s ${sid} \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--source_balanced &&


# ISFDA: primary & secondary label
python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/isfda \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep

done

# Source Model without Balanced Sampling
python ../object/image_source.py \
--trte val --output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id 1 \
--dset domainnet \
--max_epoch 20 \
--s 3 \
--t 1 \
--net resnet50 \
--lr 1e-2 &&

# ISFDA: primary & secondary label
python ../object/image_target_multi_lb.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s 3 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/my/isfda \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep
