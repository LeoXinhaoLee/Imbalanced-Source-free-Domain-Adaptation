"""
Calculate KS statistic between the label marginal distributions of source and target domain in different datasets.
"""
import os
import math


def getClassDict(total_list):
    cls_dict = dict()
    for l in total_list:
        if not int(l.split(' ')[1].strip()) in cls_dict:
            cls_dict[int(l.split(' ')[1].strip())] = []

        cls_dict[int(l.split(' ')[1].strip())].append(l)

    return cls_dict


folder = './data/domainnet'
domain = ['clipart', 'painting', 'real', 'sketch']
cls_num = 40

"""
folder = './data/VISDA-RSUT'
domain = ['train', 'validation']
cls_num = 12
"""

"""
folder = './data/office-home'
domain = ['Clipart', 'Product', 'RealWorld']
cls_num = 65
"""


for src in domain:
    for tgt in domain:
        if tgt == src:
            continue
        else:
            with open(os.path.join(folder, src+'_train_mini.txt'), 'r') as f:
            #with open(os.path.join(folder, src+'_RS.txt'), 'r') as f:
            #with open(os.path.join(folder, src+'.txt'), 'r') as f:
                src_list = f.readlines()
            with open(os.path.join(folder, tgt+'_test_mini.txt'), 'r') as f:
            #with open(os.path.join(folder, tgt+'_UT.txt'), 'r') as f:
            #with open(os.path.join(folder, tgt+'.txt'), 'r') as f:
                tgt_list = f.readlines()

            src_num = len(src_list)
            tgt_num = len(tgt_list)

            src_cls_dict = getClassDict(src_list)
            tgt_cls_dict = getClassDict(tgt_list)

            ks = 0.
            cum_prob_s = 0.
            cum_prob_t = 0.
            for c in range(cls_num):
                prob_this_cls_s = len(src_cls_dict[c]) * 1.0 / src_num
                prob_this_cls_t = len(tgt_cls_dict[c]) * 1.0 / tgt_num

                cum_prob_s += prob_this_cls_s
                cum_prob_t += prob_this_cls_t
                dist = math.fabs(cum_prob_s - cum_prob_t)

                if dist > ks:
                    ks = dist
            print('KS of '+src+' to '+tgt+': {:.4f}'.format(ks))
    print('\n')
