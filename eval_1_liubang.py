#!/usr/bin/python
# -*- coding:utf-8 -*-
# coding=utf-8
# Author: houkai
# Mail: houkai.hk@alibaba-inc.com
# Created Time: 2018-10-08 15:21
# Filename: eval_liubang.py
# Description: 
#
import sys
import os, os.path
import cv2
import random

import math
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim import queues
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def bboxes_draw_on_img(img, _scores, _bboxes, colors, thickness=2):
    shape = img.shape
    #print _scores #map
    #print _bboxes #map
    for cls, scores in _scores.iteritems():
        color = colors[cls]
        bboxes = _bboxes[cls]
        for b in range(scores.shape[0]):
            bbox_b = bboxes[b]
            scores_b = scores[b]
            for i in range(scores.shape[1]):
                bbox = bbox_b[i]
                # Draw bounding box...
                p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
                p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
                cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
                # Draw text...
                s = '%s/%.3f' % (cls, scores_b[i])
                p1 = (p1[0]-5, p1[1])
                cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()

isess = tf.InteractiveSession()

from datasets import pascalvoc_2007
DATASET_DIR = '/mogu/liubang/mytf/tfrecords/'
SPLIT_NAME = 'val'
BATCH_SIZE = 16
# Dataset provider loading data from the dataset.
dataset = pascalvoc_2007.get_split(SPLIT_NAME, DATASET_DIR)
provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                          shuffle=False,
#                                                           num_epochs=1,
                                                          common_queue_capacity=2 * BATCH_SIZE,
                                                          common_queue_min=BATCH_SIZE)
[image, shape, bboxes, labels] = provider.get(['image', 'shape', 'object/bbox', 'object/label'])
print('Dataset:', dataset.data_sources, '|', dataset.num_samples)

# with queues.QueueRunners(sess):
# Start populating queues.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

from nets import ssd_vgg_512
from nets import ssd_common

from preprocessing import ssd_vgg_preprocessing

ckpt_filename = '/mogu/liubang/mytf/SSD-Tensorflow/logs2/model.ckpt-122449'
NUM=7

# SSD object.
reuse = True if 'ssd' in locals() else None
params = ssd_vgg_512.SSDNet.default_params
ssd_params = params._replace(num_classes=NUM)
ssd = ssd_vgg_512.SSDNet(ssd_params)

# Image pre-processimg
out_shape = ssd.params.img_shape
image_pre, labels_pre, bboxes_pre, bbox_img = \
    ssd_vgg_preprocessing.preprocess_for_eval(image, labels, bboxes, out_shape, 
                                              resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

image_4d = tf.expand_dims(image_pre, 0)

# SSD construction.
with slim.arg_scope(ssd.arg_scope(weight_decay=0.0005)):
    predictions, localisations, logits, end_points = ssd.net(image_4d, is_training=False, reuse=reuse)
    
# SSD default anchor boxes.
img_shape = out_shape
layers_anchors = ssd.anchors(img_shape, dtype=np.float32)

for k in sorted(end_points.keys()):
    print(k, end_points[k].get_shape())
    
# Targets encoding.
target_labels, target_localizations, target_scores = \
    ssd_common.tf_ssd_bboxes_encode(labels, bboxes_pre, layers_anchors, 
                                    num_classes=params.num_classes, no_annotation_label=params.no_annotation_label)

nms_threshold = 0.5

# Output decoding.
localisations = ssd.bboxes_decode(localisations, layers_anchors)
tscores, tbboxes = ssd.detected_bboxes(predictions, localisations, select_threshold=0.8, nms_threshold=0.5)

# Initialize variables.
init_op = tf.global_variables_initializer()
isess.run(init_op)
# Restore SSD model.
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# Run model.
[rimg, rpredictions, rlocalisations, rscores, rbboxes, glabels, gbboxes, rbbox_img, rt_labels, rt_localizations, rt_scores] = \
    isess.run([image_4d, predictions, localisations, tscores, tbboxes,
               labels, bboxes_pre, bbox_img, 
               target_labels, target_localizations, target_scores])

#print(rscores)

# Draw bboxes
img_bboxes = np.copy(ssd_vgg_preprocessing.np_image_unwhitened(rimg[0]))
bboxes_draw_on_img(img_bboxes, rscores, rbboxes, colors_tableau, thickness=1)

# bboxes_draw_on_img(img_bboxes, glabels, np.zeros_like(glabels), gbboxes, colors_tableau, thickness=1)

print('Grountruth labels:', list(glabels))

fig = plt.figure(figsize = (12,12))
plt.imshow(img_bboxes)
plt.savefig("gg.png")



