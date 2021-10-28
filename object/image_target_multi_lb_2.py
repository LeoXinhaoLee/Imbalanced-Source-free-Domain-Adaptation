import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_PA
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import random
import subprocess
import time

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList(txt_tar, transform=image_train(), cfg=args, balance_sample=False)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test(), cfg=args, balance_sample=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders, txt_tar


def cal_acc(loader, netF, netB, netC, visda_flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracy *= 100
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    per_cls_acc_vec = matrix.diagonal() / matrix.sum(axis=1) * 100
    per_cls_avg_acc = per_cls_acc_vec.mean()    # Per-class avg acc
    per_cls_acc_list = [str(np.round(i, 2)) for i in per_cls_acc_vec]
    acc_each_cls = ' '.join(per_cls_acc_list)

    if visda_flag:
        # For VisDA, return acc of each cls to be printed
        # overall acc, acc of each cls: str, per-class avg acc
        return accuracy, acc_each_cls, per_cls_avg_acc

    else:
        # For other datasets, need not return acc of each cls
        # overall acc, acc of each cls: str, mean-ent
        return accuracy, per_cls_avg_acc, mean_ent


def train_target(args):
    dset_loaders, txt_tar = data_load(args)
    dsets = dict()
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    iter_per_epoch = len(dset_loaders["target"])
    print("Iter per epoch: {}".format(iter_per_epoch))
    interval_iter = max_iter // args.interval
    iter_num = 0

    if args.paral:
        netF = torch.nn.DataParallel(netF)
        netB = torch.nn.DataParallel(netB)
        netC = torch.nn.DataParallel(netC)

    netF.train()
    netB.train()
    netC.eval()

    if args.scd_lamb:
        scd_lamb_init = args.scd_lamb   # specify hyperparameter for secondary label correcion manually
    else:
        if args.dset[0:5] == "VISDA" :
            scd_lamb_init = 0.1
        elif args.dset[0:11] == "office-home":
            scd_lamb_init = 0.2
            if args.s == 3 and args.t == 2:
                scd_lamb_init *= 0.1
        elif args.dset[0:9] == "domainnet":
            scd_lamb_init = 0.02

    scd_lamb = scd_lamb_init

    while iter_num < max_iter:
        k = 1.0
        k_s = 0.6
        if iter_num % interval_iter == 0 and args.cls_par > 0:  # interval_itr = itr per epoch
            netF.eval()
            netB.eval()

            label_prob_dict = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label, pseudo_lb_prob = label_prob_dict['primary_lb'], label_prob_dict['primary_lb_prob']
            mem_label, pseudo_lb_prob = torch.from_numpy(mem_label).cuda(), torch.from_numpy(pseudo_lb_prob).cuda()
            if args.scd_label:
                second_label, second_prob = label_prob_dict['secondary_lb'], label_prob_dict['secondary_lb_prob']
                second_label, second_prob = torch.from_numpy(second_label).cuda(), torch.from_numpy(second_prob).cuda()
            if args.third_label:
                third_label, third_prob = label_prob_dict['third_lb'], label_prob_dict['third_lb_prob']
                third_label, third_prob = torch.from_numpy(third_label).cuda(), torch.from_numpy(third_prob).cuda()
            if args.fourth_label:
                fourth_label, fourth_prob = label_prob_dict['fourth_lb'], label_prob_dict['fourth_lb_prob']
                fourth_label, fourth_prob = torch.from_numpy(fourth_label).cuda(), torch.from_numpy(fourth_prob).cuda()
            if args.topk_ent:
                all_entropy = label_prob_dict['entropy']
                all_entropy = torch.from_numpy(all_entropy)

            if args.dset[0:5] == "VISDA" :
                if iter_num // iter_per_epoch < 1:
                    k = 0.6
                elif iter_num // iter_per_epoch < 2:
                    k = 0.7
                elif iter_num // iter_per_epoch < 3:
                    k = 0.8
                elif iter_num // iter_per_epoch < 4:
                    k = 0.9
                else:
                    k = 1.0

                if iter_num // iter_per_epoch >= 8:
                    scd_lamb *= 0.1

            elif args.dset[0:11] == "office-home" or args.dset[0:9] == "domainnet":
                if iter_num // iter_per_epoch < 2:
                    k = 0.2
                elif iter_num // iter_per_epoch < 4:
                    k = 0.4
                elif iter_num // iter_per_epoch < 8:
                    k = 0.6
                elif iter_num // iter_per_epoch < 12:
                    k = 0.8

            if args.topk:
                dsets["target"] = ImageList_PA(txt_tar, mem_label, pseudo_lb_prob, k_low=k, k_up=None,
                                                transform=image_train())
                dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.worker, drop_last=False)

            if args.topk_ent:
                dsets["target"] = ImageList_PA(txt_tar, mem_label, -1.0 * all_entropy, k_low=k, k_up=None,
                                                transform=image_train())
                dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.worker, drop_last=False)

            if args.scd_label:
                # 2nd label threshold: prob top 60%
                dsets["target_scd"] = ImageList_PA(txt_tar, second_label, second_prob, k_low=k_s, k_up=None,
                                                    transform=image_train())
                dset_loaders["target_scd"] = DataLoader(dsets["target_scd"], batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.worker, drop_last=False)

            if args.third_label:
                # 3rd label threshold: prob top 60%
                dsets["target_third"] = ImageList_PA(txt_tar, third_label, third_prob, k_low=k_s, k_up=None,
                                                    transform=image_train())
                dset_loaders["target_third"] = DataLoader(dsets["target_third"], batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.worker, drop_last=False)
            if args.fourth_label:
                # 4th label threshold: prob top 60%
                dsets["target_fourth"] = ImageList_PA(txt_tar, fourth_label, fourth_prob, k_low=k_s, k_up=None,
                                                      transform=image_train())
                dset_loaders["target_fourth"] = DataLoader(dsets["target_fourth"], batch_size=args.batch_size,
                                                          shuffle=True,
                                                          num_workers=args.worker, drop_last=False)
            netF.train()
            netB.train()

        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()  # tar_idx: chosen indices in current itr

        if inputs_test.size(0) == 1:
            continue

        if args.scd_label:
            try:
                inputs_test_scd, _, tar_idx_scd = iter_test_scd.next()
            except:
                iter_test_scd = iter(dset_loaders["target_scd"])
                inputs_test_scd, _, tar_idx_scd = iter_test_scd.next()

            if inputs_test_scd.size(0) == 1:
                continue

        if args.third_label:
            try:
                inputs_test_third, _, tar_idx_third = iter_test_third.next()
            except:
                iter_test_third = iter(dset_loaders["target_third"])
                inputs_test_third, _, tar_idx_third = iter_test_third.next()

            if inputs_test_third.size(0) == 1:
                continue

        if args.fourth_label:
            try:
                inputs_test_fourth, _, tar_idx_fourth = iter_test_fourth.next()
            except:
                iter_test_fourth = iter(dset_loaders["target_fourth"])
                inputs_test_fourth, _, tar_idx_fourth = iter_test_fourth.next()

            if inputs_test_fourth.size(0) == 1:
                continue

        iter_num += 1

        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.scd_label:
            inputs_test_scd = inputs_test_scd.cuda()
            if inputs_test_scd.ndim == 3:
                inputs_test_scd = inputs_test_scd.unsqueeze(0)

            features_test_scd = netB(netF(inputs_test_scd))
            outputs_test_scd = netC(features_test_scd)

            first_prob_of_scd = pseudo_lb_prob[tar_idx_scd]
            scd_prob = second_prob[tar_idx_scd]
            if not args.no_mask:
                mask = (scd_prob / first_prob_of_scd.float()).clamp(max=1.0)
            else:
                mask = torch.ones_like(scd_prob).cuda()

        if args.third_label:
            inputs_test_third = inputs_test_third.cuda()
            if inputs_test_third.ndim == 3:
                inputs_test_third = inputs_test_third.unsqueeze(0)

            features_test_third = netB(netF(inputs_test_third))
            outputs_test_third = netC(features_test_third)

            first_prob_of_third = pseudo_lb_prob[tar_idx_third]
            thi_prob = third_prob[tar_idx_third]

            mask_third = (thi_prob / first_prob_of_third.float()).clamp(max=1.0)

        if args.fourth_label:
            inputs_test_fourth = inputs_test_fourth.cuda()
            if inputs_test_fourth.ndim == 3:
                inputs_test_fourth = inputs_test_fourth.unsqueeze(0)

            features_test_fourth = netB(netF(inputs_test_fourth))
            outputs_test_fourth = netC(features_test_fourth)

            first_prob_of_fourth = pseudo_lb_prob[tar_idx_fourth]
            fth_prob = fourth_prob[tar_idx_fourth]

            mask_fourth = (fth_prob / first_prob_of_fourth.float()).clamp(max=1.0)

        if args.intra_dense or args.inter_sep:
            intra_dist = torch.zeros(1).cuda()
            inter_dist = torch.zeros(1).cuda()
            pred = mem_label[tar_idx]
            same_first = True
            diff_first = True
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            for i in range(pred.size(0)):
                for j in range(i, pred.size(0)):
                    # dist = torch.norm(features_test[i] - features_test[j])
                    dist = 0.5 * (1 - cos(features_test[i].unsqueeze(0), features_test[j].unsqueeze(0)))
                    if pred[i].item() == pred[j].item():
                        if same_first:
                            intra_dist = dist.unsqueeze(0)
                            same_first = False
                        else:
                            intra_dist = torch.cat((intra_dist, dist.unsqueeze(0)))

                    else:
                        if diff_first:
                            inter_dist = dist.unsqueeze(0)
                            diff_first = False
                        else:
                            inter_dist = torch.cat((inter_dist, dist.unsqueeze(0)))

            intra_dist = torch.mean(intra_dist)
            inter_dist = torch.mean(inter_dist)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)  # self-train by pseudo label
            classifier_loss *= args.cls_par

            if args.scd_label:
                pred_scd = second_label[tar_idx_scd]
                classifier_loss_scd = nn.CrossEntropyLoss(reduction='none')(outputs_test_scd,
                                                                            pred_scd)  # self-train by pseudo label

                classifier_loss_scd = torch.mean(mask * classifier_loss_scd)

                classifier_loss_scd *= args.cls_par

                classifier_loss += classifier_loss_scd * scd_lamb

            if args.third_label:
                pred_third = third_label[tar_idx_third]
                classifier_loss_third = nn.CrossEntropyLoss(reduction='none')(outputs_test_third,
                                                                            pred_third)  # self-train by pseudo label

                classifier_loss_third = torch.mean(mask_third * classifier_loss_third)

                classifier_loss_third *= args.cls_par

                classifier_loss += classifier_loss_third * scd_lamb    # TODO: better weighting is possible

            if args.fourth_label:
                pred_fourth = fourth_label[tar_idx_fourth]
                classifier_loss_fourth = nn.CrossEntropyLoss(reduction='none')(outputs_test_fourth,
                                                                            pred_fourth)  # self-train by pseudo label

                classifier_loss_fourth = torch.mean(mask_fourth * classifier_loss_fourth)

                classifier_loss_fourth *= args.cls_par

                classifier_loss += classifier_loss_fourth * scd_lamb    # TODO: better weighting is possible

            if iter_num < interval_iter and (args.dset == "VISDA-C" or args.dset == "VISDA-RSUT" or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10'):
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))  # Minimize local entropy

            if args.scd_label:
                softmax_out_scd = nn.Softmax(dim=1)(outputs_test_scd)
                entropy_loss_scd = torch.mean(mask * loss.Entropy(softmax_out_scd))  # Minimize local entropy

            if args.third_label:
                softmax_out_third = nn.Softmax(dim=1)(outputs_test_third)
                entropy_loss_third = torch.mean(mask_third * loss.Entropy(softmax_out_third))  # Minimize local entropy

            if args.fourth_label:
                softmax_out_fourth = nn.Softmax(dim=1)(outputs_test_fourth)
                entropy_loss_fourth = torch.mean(mask_fourth * loss.Entropy(softmax_out_fourth))  # Minimize local entropy

            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss  # Maximize global entropy

                if args.scd_label:
                    msoftmax_scd = softmax_out_scd.mean(dim=0)
                    gentropy_loss_scd = torch.sum(-msoftmax_scd * torch.log(msoftmax_scd + args.epsilon))
                    entropy_loss_scd -= gentropy_loss_scd  # Maximize global entropy
                if args.third_label:
                    msoftmax_third = softmax_out_third.mean(dim=0)
                    gentropy_loss_third = torch.sum(-msoftmax_third * torch.log(msoftmax_third + args.epsilon))
                    entropy_loss_third -= gentropy_loss_third  # Maximize global entropy
                if args.fourth_label:
                    msoftmax_fourth = softmax_out_fourth.mean(dim=0)
                    gentropy_loss_fourth = torch.sum(-msoftmax_fourth * torch.log(msoftmax_fourth + args.epsilon))
                    entropy_loss_fourth -= gentropy_loss_fourth  # Maximize global entropy

            im_loss = entropy_loss * args.ent_par
            if args.scd_label:
                im_loss += entropy_loss_scd * args.ent_par * scd_lamb
            if args.third_label:
                im_loss += entropy_loss_third * args.ent_par * scd_lamb    # TODO: better weighting is possible
            if args.fourth_label:
                im_loss += entropy_loss_fourth * args.ent_par * scd_lamb   # TODO: better weighting is possible

            classifier_loss += im_loss

        if args.intra_dense:
            classifier_loss += args.lamb_intra * intra_dist.squeeze()
        if args.inter_sep:
            classifier_loss += args.lamb_inter * inter_dist.squeeze()

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10':
                # For VisDA, print the acc of each cls
                acc_s_te, acc_list, acc_cls_avg = cal_acc(dset_loaders['test'], netF, netB, netC, visda_flag=True)    # flag for VisDA -> need cls avg acc.
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Cls Avg Acc = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te, acc_cls_avg) + '\n' + acc_list
            else:
                # In imbalanced setting, use per-class avg acc as metric
                # For Office-Home, DomainNet, no need to print the acc of each cls
                acc_s_te, acc_cls_avg, _ = cal_acc(dset_loaders['test'], netF, netB, netC, visda_flag=False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Cls Avg Acc = {:.2f}%'.format(args.name, iter_num,
                                                                            max_iter, acc_s_te, acc_cls_avg)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)  # output logits
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)  # pred prob
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)  # the bigger ent is, the smaller weight this class has
    _, predict = torch.max(all_output, 1)

    all_entropy = torch.sum(-1.0 * all_output.float() * torch.log(all_output.float()), dim=1).numpy()
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)  # acc before & after clustering


    pseudo_lb_prob = np.zeros(pred_label.shape)
    for i in range(pred_label.shape[0]):
        pseudo_lb_prob[i] = all_output[i][pred_label[i]]

    label_prob_dict = dict()
    label_prob_dict['primary_lb'] = pred_label.astype('int')
    label_prob_dict['primary_lb_prob'] = pseudo_lb_prob
    label_prob_dict['entropy'] = all_entropy

    ## Secondary labels
    if args.scd_label:
        second_lb = torch.zeros(predict.size()).numpy()
        second_prob = torch.zeros(predict.size()).numpy()

        for i in range(second_lb.shape[0]):
            idx = np.argsort(all_output[i].numpy())[-2]
            second_lb[i] = idx
            second_prob[i] = all_output[i][idx]

        label_prob_dict['secondary_lb'] = second_lb.astype('int')
        label_prob_dict['secondary_lb_prob'] = second_prob

    ## Third labels
    if args.third_label:
        third_lb = torch.zeros(predict.size()).numpy()
        third_prob = torch.zeros(predict.size()).numpy()

        for i in range(third_lb.shape[0]):
            idx = np.argsort(all_output[i].numpy())[-3]
            third_lb[i] = idx
            third_prob[i] = all_output[i][idx]

        label_prob_dict['third_lb'] = third_lb.astype('int')
        label_prob_dict['third_lb_prob'] = third_prob

    ## Fourth labels
    if args.fourth_label:
        fourth_lb = torch.zeros(predict.size()).numpy()
        fourth_prob = torch.zeros(predict.size()).numpy()

        for i in range(fourth_lb.shape[0]):
            idx = np.argsort(all_output[i].numpy())[-4]
            fourth_lb[i] = idx
            fourth_prob[i] = all_output[i][idx]

        label_prob_dict['fourth_lb'] = fourth_lb.astype('int')
        label_prob_dict['fourth_lb_prob'] = fourth_prob

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return label_prob_dict


def wait_for_GPU_avaliable(gpu_id):
    isVisited = False
    while True:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,nounits,noheader']).decode()
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        available_memory = gpu_memory_map[int(gpu_id)]

        # wait unless GPU memory is more than 10000
        #if available_memory < 10000:
        if available_memory < 6500:
            if not isVisited:
                print("GPU full, wait...........")
                isVisited = True
            time.sleep(120)
            continue
        else:
            print("Empty GPU! Start process!")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['office-home-RSUT', 'domainnet', 'VISDA-RSUT', 'VISDA-RSUT-50', 'VISDA-RSUT-10',
                                 'VISDA-Beta', 'VISDA-Tweak', 'VISDA-Knockout'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='../result/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--topk', default=False, action='store_true')
    parser.add_argument('--topk_ent', default=False, action='store_true')
    parser.add_argument('--scd_label', default=False, action='store_true')
    parser.add_argument('--scd_lamb', type=float, default=None)
    parser.add_argument('--third_label', default=False, action='store_true')
    parser.add_argument('--fourth_label', default=False, action='store_true')
    parser.add_argument('--intra_dense', default=False, action='store_true')
    parser.add_argument('--inter_sep', default=False, action='store_true')
    parser.add_argument('--no_mask', default=False, action='store_true')
    parser.add_argument('--lamb_intra', type=float, default=1.0)
    parser.add_argument('--lamb_inter', type=float, default=-0.1)
    parser.add_argument('--paral', default=False, action='store_true')

    args = parser.parse_args()

    if args.dset == 'office-home-RSUT':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']  # only 1,2,3 are available
        args.class_num = 65
    if args.dset == 'domainnet':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 40
    if args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10' \
            or args.dset == 'VISDA-Tweak' or args.dset == 'VISDA-Knockout':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'VISDA-Beta':
        names = ['train_b1_a1', 'validation_b2.0_a1.2', 'validation_b2.0_a2.0', 'validation_b2.0_a2.7']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    wait_for_GPU_avaliable(args.gpu_id)

    for i in range(len(names)):
        if i == args.s:
            continue
        if args.dset == 'office-home-RSUT' and names[i] == 'Art':
            continue
        args.t = i

        folder = '../data/'
        if args.dset == 'office-home-RSUT':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
        elif args.dset == 'domainnet':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_train_mini.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test_mini.txt'
        elif args.dset == 'VISDA-RSUT' or args.dset == 'VISDA-RSUT-50' or args.dset == 'VISDA-RSUT-10':
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_RS.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_UT.txt'
        else:
            args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())

        if args.dset != 'VISDA-Beta':
            args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() +
                                       names[args.t][0].upper())
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
        else:
            args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() +
                                       names[args.t][0].upper() + names[args.t][-4:])
            args.name = names[args.s][0].upper() + names[args.t][0].upper() + names[args.t][-4:]

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        train_target(args)
