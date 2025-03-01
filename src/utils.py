import os
import numpy as np
import logging
import matplotlib.pyplot as plt


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # ensure a handler is added only once
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    return logger


def get_pred_coordinates(pred_map, name, w, h, all_pred_labels):
    """
    decode heatmap to coordinates
    :param pred_map:    Tensor  CPU     size:(batch_size, 21, 46, 46)
    :param name:        string list,     length: batch_size
    :param w:           origin image width
    :param h:           origin image width
    :param all_pred_labels:     a dict to save all prediction coordinates
    :return:
    """
    w = w.numpy()
    h = h.numpy()
    for b in range(pred_map.shape[0]):  # for image in one batch
        label_list = []
        for k in range(21):
            tmp_pre = np.asarray(pred_map[b, k, :])   # 2D array  size:(2)

            # get coordinate of keypoints in origin image scale
            x = int(tmp_pre[0] * w[b])
            y = int(tmp_pre[1] * h[b])
            label_list.append([x, y])  # save img label to json

        # save prediction result to dict
        all_pred_labels[name[b]] = {}
        all_pred_labels[name[b]]['pred_label'] = label_list
        all_pred_labels[name[b]]['resol'] = float(w[b])
    return all_pred_labels


def get_pck_with_sigma(predict_labels_dict, gt_labels, sigma_list):
    """
    Get PCK with different sigma threshold
    :param predict_labels_dict:  dict  element:  'img_name':{'prd_label':[list, coordinates of 21 keypoints],
                                                             'resol': origin image size}
    :param gt_labels:            dict  element:  'img_name': [list, coordinates of 21 keypoints ]
    :param sigma_list:       list    different sigma threshold
    :return:
    """
    # print(predict_labels_dict)
    pck_dict = {}
    for im in predict_labels_dict:
        gt_label = gt_labels[im]        # list    len:21      element:[x, y]
        pred_label = predict_labels_dict[im]['pred_label']  # list    len:21      element:[x, y]
        im_size = predict_labels_dict[im]['resol']
        for sigma in sigma_list:
            if sigma not in pck_dict:
                pck_dict[sigma] = []
            pck_dict[sigma].append(PCK(pred_label, gt_label, im_size/2.2, sigma))
            # Attention!
            # since our cropped image is 2.2 times of hand tightest bounding box,
            # we simply use im_size / 2,2 as the tightest bounding box

    pck_res = {}
    for sigma in sigma_list:
        pck_res[sigma] = sum(pck_dict[sigma]) / len(pck_dict[sigma])
    return pck_res


def PCK(predict, target, bb_size=256, sigma=0.1):
    """
    Calculate PCK
    :param predict: list    len:21      element:[x, y]
    :param target:  list    len:21      element:[x, y]
    :param bb_size: tightest bounding box length of hand
    :param sigma:   threshold, we use 0.1 in default
    :return: scala range [0,1]
    """
    pck = 0
    for i in range(21):
        pre = predict[i]
        tar = target[i]
        dis = np.sqrt((pre[0] - tar[0]) ** 2 + (pre[1] - tar[1]) ** 2)
        if dis < sigma * bb_size:
            pck += 1
    return pck / 21.0

