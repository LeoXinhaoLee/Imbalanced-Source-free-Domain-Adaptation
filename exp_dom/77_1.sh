#!/bin/bash

g='1'
seed=('2020')
exp=('exp1')
src=('0' '1' '2' '3')

s=${seed[0]}
e=${exp[0]}
sid=${src[3]}


### Test on S->C/P/R

# Source Model without Balanced Sampling
python ../object/image_source.py \
--trte stratified \
--output ./seed${s}/${e}/baseline/src_strat \
--da uda \
--gpu_id ${g} \
--dset domainnet \
--max_epoch 20 \
--s ${sid} \
--t 1 \
--net resnet50 \
--lr 1e-2 \
--seed ${s} &&

# ISFDA: primary & secondary label
python ../object/image_target_multi_lb_2.py \
--cls_par 0.3 \
--da uda \
--dset domainnet \
--max_epoch 15 \
--gpu_id ${g} \
--s ${sid} \
--output_src ./seed${s}/${e}/baseline/src_strat \
--output ./seed${s}/${e}/my/isfda_s \
--net resnet50 \
--lr 1e-2  \
--topk \
--scd_label \
--intra_dense \
--inter_sep \
--seed ${s}
