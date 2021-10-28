import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
import math

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    """Conventional Random Sampler or Class-balanced Sampler
    Percentage-unaware
    """
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', cfg=None,
                 balance_sample=True):
        """
        Initialize the ImageList class
        :param image_list: list of image paths
        :param labels: tensor of labels
        :param transform: transform for loaded images
        :param target_transform: target-image-specific transform
        :param mode: how to load images of a certain format
        :param cfg: arguments regarding training and dataset
        :param balance_sample: if True: balanced sampling, else: random sampling
        """
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        assert cfg != None, 'Have not passed arguments needed.'
        self.cls_num = cfg.class_num
        self.balance_sample = balance_sample
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.class_dict = self._get_class_dict()  # each cls's samples' id

    def get_annotations(self):
        annos = []
        for (img_path, lb) in self.imgs:
            annos.append({'category_id': int(lb)})
        return annos

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict


    def __getitem__(self, index):
        if self.balance_sample:
            """Balanced Sampling
            step1. Select one class from uniform distribution.
            step2. Randomly select one sample from the selected class.
            """
            sample_class = random.randint(0, self.cls_num - 1)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        path, target = self.imgs[index]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_PA(Dataset):
    """Percentage-aware (PA) sampling.
    Draw samples with Top-k percent confidence or Lowest k percent entropy within each class to form sample pool in Curriculum Sampling.
    """
    def __init__(self, image_list, label, prob, k_low=0.05, k_up=None, transform=None, target_transform=None,
                 mode='RGB'):
        """Initialize the ImageList_Percent_Aware class
        :param image_list: list of image paths
        :param label: tensor of corresponding pseudo labels
        :param prob: sorting criterion, choices=[confidence, -1 * entropy]. Per Descending order.
        :param k_low: lower percentile threshold
        :param k_up: higher percentile threshold. Sorting interval=[0,k_low] if k_up==None else [k_low,k_up]
        :param transform: transform for loaded images
        :param target_transform: target-image-specific transform
        :param mode: how to load images of a certain format
        """
        imgs = make_dataset(image_list, labels=None)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs =imgs

        # label, prob: follow sequential order of original data
        self.label = label  # tensor
        self.prob = prob
        self.top_k_percent = k_low
        self.k_up = k_up
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.class_dict = self._make_class_dict()
        self.pick_sample_dict = self._sample_top_k_percent()  # picked samples in each class
        self.imgs_all_pick = self._pick_imgs()                # list of all the picked samples

    def _make_class_dict(self):
        class_dict = dict()
        for i, lb in enumerate(self.label.tolist()):
            # i: image index in the original loading sequence
            # lb: pseudo label of the i-th image
            cat_id = lb
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def _sample_top_k_percent(self):
        prob_dict = dict()
        for cls in self.class_dict.keys():
            if not cls in prob_dict:
                prob_dict[cls] = []
            prob_dict[cls].extend(self.prob[self.class_dict[cls]].tolist())

        pick_sample_dict = dict()
        for cls in prob_dict.keys():
            if not cls in pick_sample_dict:
                pick_sample_dict[cls] = []
            prob_cls = np.array(prob_dict[cls])
            idx = np.argsort(prob_cls)
            idx = list(idx)
            idx.reverse()   # descending order
            pick_num = math.ceil(len(idx) * self.top_k_percent)
            if self.k_up == None:
                pick_idx = idx[0:pick_num]
            else:
                pick_num_floor = math.floor(len(idx) * self.top_k_percent)
                pick_num_ceil = math.ceil(len(idx) * self.k_up)
                pick_idx = idx[pick_num_floor:pick_num_ceil]

            pick_sample_dict[cls].extend(list(np.array(self.class_dict[cls])[pick_idx]))

        return pick_sample_dict

    def _pick_imgs(self):
        pick_img = []
        for cls in self.pick_sample_dict.keys():
            for idx in self.pick_sample_dict[cls]:
                path, target = self.imgs[idx]
                pick_img.append((path, target, idx))    # idx is that in original dataset
        return pick_img

    def __getitem__(self, index):
        path, target, real_idx = self.imgs_all_pick[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, real_idx

    def __len__(self):
        return len(self.imgs_all_pick)
