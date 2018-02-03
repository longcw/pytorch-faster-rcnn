#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.pvanet import pvanet

import torch

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

# CLASSES = ('__background__', 'person',
#            'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
#            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
#            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
#            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
#            )


CLASSES = ('__background__', 'person',)


NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections_cv2(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    for i in inds:
        bbox = dets[i, :4].astype(np.int)
        score = dets[i, -1]
        thick = int(max(sum(im.shape[0:2]) / 600., 2))
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=thick)
        # cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (bbox[0], bbox[1] - 2), 0, 1, (255, 0, 0), thickness=2)
        cv2.putText(im, '{:.3f}'.format(score), (bbox[0], bbox[1] - 2), 0,  1e-3 * im.shape[0], (255, 0, 0), thickness=thick//3)

    max_size = 1000
    if max(im.shape[:2]) > max_size:
        scale = float(max_size) / max(im.shape[:2])
        im = cv2.resize(im, None, fx=scale, fy=scale)

    return im


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name, conf_threshold):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = conf_threshold
    NMS_THRESH = 0.3

    # cls_ind = 1
    # cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    # cls_scores = scores[:, cls_ind]
    # dets = np.hstack((cls_boxes,
    #                   cls_scores[:, np.newaxis])).astype(np.float32)
    # keep = nms(torch.from_numpy(dets), NMS_THRESH)
    # dets = dets[keep.numpy(), :]
    #
    # return dets

    person_detection = None
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        im = vis_detections_cv2(im, cls, dets, thresh=CONF_THRESH)
        if CLASSES[cls_ind] == 'person':
            person_detection = dets
    return im, person_detection


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res50')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    # saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
    #                           NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))
    # saved_model = '/extra/models/routianluo/voc_0712_80k-110k.tar'
    # saved_model = '/extra/models/routianluo/res101_faster_rcnn_iter_1190000.pth'
    # saved_model = '/data/models/routianluo/longc/res50_faster_rcnn_iter_335000.pth'
    saved_model = '/extra/models/routianluo/longc/res50_person2_faster_rcnn_iter_490000.pth'
    # saved_model = '/extra/models/routianluo/longc/pvanet_faster_rcnn_iter_485000.pth'
    # im_root = '/data/2DMOT2015/demo/Demo2/img1'
    # im_root = '/extra/Syncs/Walmart/images/'
    # im_root = '/extra/Syncs/Walmart/demo'

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'res50':
        net = resnetv1(num_layers=50)
    elif demonet == 'pvanet':
        net = pvanet()
    else:
        raise NotImplementedError
    # net.create_architecture(81,
    #                       tag='default', anchor_scales=[4,8,16,32])
    # net.create_architecture(2,
    #                       tag='default', anchor_scales=[4,8,16,32])
    net.create_architecture(2,
                          tag='default', anchor_scales=[4,8,16], anchor_ratios=(1, 2, 3))

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))


    def run_detection(im_root, result_root, conf_threshold):
        print('detection in {}'.format(im_root))

        im_names = sorted(os.listdir(im_root))

        for i, im_name in enumerate(im_names):
            im_file = os.path.join(im_root, im_name)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('run detection for {}'.format(im_file))
            im, dets = demo(net, im_file, conf_threshold)
            dets = dets[dets[:, 4] > conf_threshold]

            with open(os.path.join(result_root, '{:04d}.txt'.format(i)), 'w') as f:
                f.write('{}\n'.format(len(dets)))
                for det in dets:
                    x1, y1, x2, y2, s = det
                    w = x2 - x1
                    h = y2 - y1
                    f.write('0 {} {} {} {} {}\n'.format(s, w, h, x1, y1))

            # MOT format
            # with open(os.path.join(result_root, '{}.txt'.format(seq_name)), 'a') as f:
            #     for det in dets:
            #         x1, y1, x2, y2, s = det
            #         w = x2 - x1
            #         h = y2 - y1
            #         frame = i + 1
            #
            #         f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(frame, x1, y1, w, h, s))

            cv2.imshow('test', im)
            cv2.waitKey(1)

    conf_threshold = 0.7

    im_root = '/data/MCMTT/EPFL/Terrace/'
    result_root = os.path.join(im_root, 'faster_rcnn_detection')
    for sub_view in os.listdir(im_root):
        if sub_view.startswith('View_'):
            dst_root = os.path.join(result_root, sub_view)
            mkdirs(dst_root)
            run_detection(os.path.join(im_root, sub_view), dst_root, conf_threshold)

    # im_root = '/data/KITTI/training/image_02'
    # result_root = '/data/KITTI/training/faster_rcnn_dets_02'
    # for seq_name in os.listdir(im_root):
    #     dst_root = os.path.join(result_root, seq_name)
    #     mkdirs(dst_root)
    #     run_detection(os.path.join(im_root, seq_name), dst_root, conf_threshold)

    # im_root = '/data/MOT16/train'
    # result_root = '/data/MOT16/faster_rcnn_dets'
    # for seq_name in os.listdir(im_root):
    #     if seq_name != 'PETS09-S2L1':
    #         continue
    #     dst_root = result_root
    #     mkdirs(dst_root)
    #     run_detection(os.path.join(im_root, seq_name, 'img1'), dst_root, conf_threshold)
