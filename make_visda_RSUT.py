"""
Create VisDA-C (RSUT) based on VisDA-C dataset
imbalance factor = 100, 50, 10

References:
    Zhou et al. BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition. CVPR 2020.
    Peng et al. Visda: The visual domain adaptation challenge. arXiv preprint arXiv:1710.06924, 2017.
    Tan et al.  Class-Imbalanced Domain Adaptation: An Empirical Odyssey. ECCV 2020.
"""
import os
import numpy as np
import random


def get_img_num_per_cls(cls_dict, cls_num, imb_type, imb_factor, im_curve='RS'):
    total_img = 0
    for k in cls_dict.keys():
        total_img += len(cls_dict[k])
    img_max = total_img / cls_num
    img_num_per_cls = []

    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)

    if im_curve == 'UT':
        img_num_per_cls.reverse()

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
            pick_idx = random.choices(list(range(len(cls_dict[the_class]))), k=the_img_num)

        for i in pick_idx:
            new_lines.append(cls_dict[the_class][i])

    return new_lines

cls_num = 12
imb_factor = 0.01  # Nmin/Nmax
#imb_factor = 0.02
#imb_factor = 0.1
imb_type= 'exp'
rand_number = 2020
np.random.seed(rand_number)
random.seed(rand_number)

train_path = './data/VISDA-C/image_list_train.txt'
val_path = './data/VISDA-C/image_list_val.txt'

save_folder_path = './data/VISDA-RSUT'
#save_folder_path = './data/VISDA-RSUT-50'
#save_folder_path = './data/VISDA-RSUT-10'

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
imb_type_list = ['RS', 'UT']

for t in target:
    for im_curve in imb_type_list:
        if t == 'train':
            cls_dict = train_cls_dict
        else:
            cls_dict = val_cls_dict
        img_num_list = get_img_num_per_cls(cls_dict, cls_num, imb_type, imb_factor, im_curve=im_curve)
        new_lines = gen_imbalanced_data(img_num_list, cls_dict, cls_num)

        with open(os.path.join(save_folder_path,'{}_{}.txt'.format(t, im_curve)), 'w') as f:
            for l in new_lines:
                f.write(l)
