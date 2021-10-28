"""
Create VisDA-C (Beta Distribution) based on VisDA-C dataset to simulate more situations of label shift.
Draw target domain from Dirichlet distribution with different concentration parameters.
Concentration parameters:
                1) b=2.0, a=1.2
                2) b=2.0, a=2.0
                3) b=2.0, a=2.7
Draw source domain from uniform distribution (b=1.0, a=1.0).
Used for testing the generality of Secondary Label Correction (SLC), i.e., to compare SHOT with SHOT+SLC.

References:
    Lipton et al. Detecting and correcting for label shift with black box predictors. ICML 2018.
    Peng et al. Visda: The visual domain adaptation challenge. arXiv preprint arXiv:1710.06924, 2017.
"""
import os
import numpy as np
import random
import scipy.stats

def get_img_num_per_cls_beta(ls_dict, cls_num, alpha=2.0, beta=2.0):
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


def gen_imbalanced_data(img_num_per_cls, cls_dict, cls_num):
    new_lines = []

    classes = range(cls_num)
    num_per_cls_dict = dict()

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        if the_img_num <= len(cls_dict[the_class]):
            pick_idx = random.sample(list(range(len(cls_dict[the_class]))), the_img_num)
        else:
            pick_idx = random.choices(list(range(len(cls_dict[the_class]))), k=the_img_num) # with replacement

        for i in pick_idx:
            new_lines.append(cls_dict[the_class][i])

    return new_lines

cls_num = 12

rand_number = 2020
np.random.seed(rand_number)
random.seed(rand_number)

train_path = './data/VISDA-C/image_list_train.txt'
val_path = './data/VISDA-C/image_list_val.txt'
save_folder = './data/VISDA-Beta'

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
a_list = [1.2, 2.0, 2.7]
b = 2.0

for t in target:
    for a in a_list:
        if t == 'train':
            cls_dict = train_cls_dict
        else:
            cls_dict = val_cls_dict

        if t == 'train':
            alpha = 1
            beta = 1
        else:
            alpha = a
            beta = b

        img_num_list = get_img_num_per_cls_beta(cls_dict, cls_num, alpha=alpha, beta=beta)
        new_lines = gen_imbalanced_data(img_num_list, cls_dict, cls_num)

        with open(os.path.join(save_folder, '{}_b{}_a{}.txt'.format(t, str(beta), str(alpha))), 'w') as f:
            for l in new_lines:
                f.write(l)
