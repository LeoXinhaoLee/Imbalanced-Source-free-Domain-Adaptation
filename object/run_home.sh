#!/bin/bash

### Run on Office-Home (RSUT): C->P, R
g='0'
s='2020'
e='exp1'

# Train the Source Model with Random Sampling
python ../object/image_source.py \
--trte val \
--output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id 1 \
--dset office-home-RSUT \
--max_epoch 50 \
--s 1 \
--t 1 \
--net resnet50 \
--lr 1e-2 ;

# SHOT
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id 1 \
--s 1 \
--output_src ./seed${s}/${e}/baseline/src \
--output ./seed${s}/${e}/baseline/shot \
--net resnet50 \
--lr 1e-2 ;

# Source Model with Balanced Sampling
python ../object/image_source.py \
--trte val \
--output ./seed${s}/${e}/baseline/src \
--da uda \
--gpu_id 1 \
--dset office-home-RSUT \
--max_epoch 50 \
--s 1 \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--source_balanced ;

# ISFDA: primary & secondary label
python ../object/image_target.py \
--cls_par 0.3 \
--da uda \
--dset office-home-RSUT \
--max_epoch 20 \
--gpu_id ${g} \
--s 1 \
--output_src ./seed${s}/${e}/my/ba_src \
--output ./seed${s}/${e}/my/ba_topk_e_scd_dense_2 \
--net resnet50 \
--lr 1e-2  \
--topk_ent \
--scd_label \
--intra_dense \
--inter_sep
