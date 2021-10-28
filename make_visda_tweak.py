"""
Create VisDA-C (Tweak) based on VisDA-C dataset to simulate more situations of label shift.
Target domain: Assign each of 3 randomly picked classes a probability 15%, and the rest of the mass is spread evenly among the other classes.
Source domain: Uniform distribution.

Used for testing the generality of Secondary Label Correction (SLC), i.e., to compare SHOT with SHOT+SLC.

References:
    Lipton et al. Detecting and correcting for label shift with black box predictors. ICML 2018.
    Peng et al. Visda: The visual domain adaptation challenge. arXiv preprint arXiv:1710.06924, 2017.
"""
import os
import numpy as np
import random
import scipy.stats


def get_img_num_per_cls_beta(cls_dict, cls_num, alpha=2.0, beta=2.0):
    total_img = 0
    for k in cls_dict.keys():
        total_img += len(cls_dict[k])
    img_max = total_img / cls_num
    img_num_per_cls = []

    b = beta
    a = alpha

    x = np.arange(0.0, 1.0, 1.0/cls_num) + 1.0 / cls_num * 0.5
    y = scipy.stats.beta.pdf(x, a, b)
    p = y / y.sum()

    for cls_idx in range(cls_num):
        num = max(img_max * p[cls_idx], 1)
        img_num_per_cls.append(int(num))

    return img_num_per_cls

def get_img_num_per_cls_tweak(cls_dict, cls_num, prob_list):
    total_img = 0
    for k in cls_dict.keys():
        total_img += len(cls_dict[k])
    img_max = total_img / cls_num
    img_num_per_cls = []

    for p in prob_list:
        img_num_per_cls.append(int(img_max * p))

    return img_num_per_cls

def gen_imbalanced_data(img_num_per_cls, cls_dict, cls_num):
    new_lines = []

    classes = range(cls_num)
    num_per_cls_dict = dict()

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        if the_img_num <= len(cls_dict[the_class]):
            pick_idx = random.sample(list(range(len(cls_dict[the_class]))), the_img_num)
        else:
            pick_idx = random.choices(list(range(len(cls_dict[the_class]))), k=the_img_num)  # with replacement

        for i in pick_idx:
            new_lines.append(cls_dict[the_class][i])

    return new_lines

cls_num = 12
tweak_num = 3
tweak_prob = 0.15

rand_number = 2020
np.random.seed(rand_number)
random.seed(rand_number)

cls_list = list(range(12))
tweak = random.sample(cls_list, tweak_num)
prob_list = np.array([(1 - tweak_num * tweak_prob) / (cls_num-tweak_num)] * cls_num)
prob_list[tweak] = tweak_prob
prob_list = prob_list.tolist()

train_path = './data/VISDA-C/image_list_train.txt'
val_path = './data/VISDA-C/image_list_val.txt'
save_folder = './data/VISDA-Tweak'

train_cls_dict = dict()
val_cls_dict = dict()

with open(train_path) as f:
    content = f.readlines()
    for l in content:
        if not int(l.split(' ')[1]) in train_cls_dict:
            train_cls_dict[int(l.split(' ')[1])] = []
        train_cls_dict[int(l.split(' ')[1])].append(l)

with open(val_path) as f:
    content = f.readlines()
    for l in content:
        if not int(l.split(' ')[1]) in val_cls_dict:
            val_cls_dict[int(l.split(' ')[1])] = []
        val_cls_dict[int(l.split(' ')[1])].append(l)


target = ['train', 'validation']

for t in target:
    if t == 'train':
        cls_dict = train_cls_dict
    else:
        cls_dict = val_cls_dict

    if t == 'train':
        # source domain subject to uniform distribution (Dirichlet distribution a=b=1)
        alpha = 1
        beta = 1
        img_num_list = get_img_num_per_cls_beta(cls_dict, cls_num, alpha=alpha, beta=beta)
    else:
        img_num_list = get_img_num_per_cls_tweak(cls_dict, cls_num, prob_list)

    new_lines = gen_imbalanced_data(img_num_list, cls_dict, cls_num)

    with open(os.path.join(save_folder, '{}.txt'.format(t)), 'w') as f:
        for l in new_lines:
            f.write(l)
