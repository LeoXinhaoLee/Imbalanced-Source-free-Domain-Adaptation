#!/bin/bash

### Run on VisDA-C (RSUT): Train -> Validation
g='0'
s='2020'
e='exp1'

# Train the Source Model with Random Sampling
python ../object/image_source.py \
--trte val \
--output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id ${g} \
--dset VISDA-RSUT \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 ;

# SHOT
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--max_epoch 15 \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/baseline/shot \
--net resnet101 \
--lr 1e-3 ;

# Source Model with Balanced Sampling
python ../object/image_source.py \
--trte val \
--output ./seed${s}/${e}/my/ba_src \
--da uda \
--gpu_id ${g} \
--dset VISDA-RSUT \
--max_epoch 15 \
--s 0 \
--t 1 \
--net resnet101 \
--lr 1e-3 \
--source_balanced ;

# ISFDA: primary & secondary label
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--max_epoch 15 \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_e_scd_dense_2 \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep ;

# ISFDA: primary & secondary & third label
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--max_epoch 15 \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_e_scd_dense_2 \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--scd_label \
--third_label \
--intra_dense \
--inter_sep ;

# ISFDA: primary & secondary & third & fourth label
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset VISDA-RSUT \
--max_epoch 15 \
--gpu_id ${g} \
--s 0 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_e_scd_dense_2 \
--net resnet101 \
--lr 1e-3  \
--topk_ent \
--scd_label \
--third_label \
--fourth_label \
--intra_dense \
--inter_sep ;
