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
from tensorflow.python.tools import freeze_graph

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


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
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
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3), name='input')
image_pre, labels_pre, bboxes_pre, bbox_img = \
    ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, out_shape, 
                                              resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

image_4d = tf.expand_dims(image_pre, 0)

# SSD construction.
with slim.arg_scope(ssd.arg_scope(weight_decay=0.0005)):
    predictions, localisations, logits, end_points = ssd.net(image_4d, is_training=False, reuse=reuse)
    
# SSD default anchor boxes.
img_shape = out_shape
layers_anchors = ssd.anchors(img_shape, dtype=np.float32)

nms_threshold = 0.5
# Output decoding.
localisations = ssd.bboxes_decode(localisations, layers_anchors)
#tscores, tbboxes = ssd.detected_bboxes(predictions, localisations, select_threshold=0.01, nms_threshold=0.45)
#各个类别卡阈值->排序top_k->nms取keep_top_k
tscores, tbboxes = ssd.detected_bboxes(predictions, localisations, select_threshold=0.1, nms_threshold=0.5, 
                                       top_k=40, keep_top_k=10)
with tf.name_scope(None, 'ssd_bboxes_class_select'):
    with tf.variable_scope("result"):
        l_classes = []
        l_scores = []
        l_bboxes = []
        for c in tscores.keys():
            scores_ = tscores[c]
            bboxes_ = tbboxes[c]
            classes_ = tf.multiply(tf.ones(tf.shape(scores_), dtype=tf.int32), c)

            l_classes.append(classes_)
            l_scores.append(scores_)
            l_bboxes.append(bboxes_)
        fclasses = tf.concat(l_classes, axis=1)
        fscores = tf.concat(l_scores, axis=1)
        fbboxes = tf.concat(l_bboxes, axis=1)

        fscores, idxes = tf.nn.top_k(fscores, k=60, sorted=True)
        fscores = tf.identity(fscores, name='fscores')
        #trick :map for each element
        def fn_gather(bbs, idxes):
            bb = tf.gather(bbs, idxes)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]), [fclasses, idxes], dtype=[fclasses.dtype],
                          parallel_iterations=10, back_prop=False, swap_memory=False, infer_shape=True)
        fclasses = tf.identity(r[0], name='fclasses')
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]), [fbboxes, idxes], dtype=[fbboxes.dtype],
                          parallel_iterations=10, back_prop=False, swap_memory=False, infer_shape=True)
        fbboxes = tf.identity(r[0], name='fbboxes')
        fnum = tf.shape(fclasses)
        fnum = tf.identity(fnum, name='fnum')

# Initialize variables.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())
        
# Restore SSD model.
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

#查看图结构，确定输出
# detection_graph = tf.Graph()
# detection_graph = sess.graph
# summary_writer = tf.summary.FileWriter('outs/', detection_graph)
# summary_writer.flush()
# summary_writer.close()
#exit(-1)

#保存图
tf.train.write_graph(sess.graph_def, './pb', 'lb.pb')

#把图和参数结构一起
freeze_graph.freeze_graph('./pb/lb.pb',
                          '',
                          False,
                          ckpt_filename, 
                          'ssd_bboxes_class_select/result/fclasses,ssd_bboxes_class_select/result/fscores,'\
                          'ssd_bboxes_class_select/result/fbboxes,ssd_bboxes_class_select/result/fnum',
                          'save/restore_all',
                          'save/Const:0',
                          'pb/frozen_lb.pb',
                          False,#
                          '')

exit(-1)

# input
img = mpimg.imread(
    '/mogu/liubang/mytf/VOCclothsonepiece/JPEGImages/100091894_ifrtmytdmvrtcnlchazdambqhayde_800x1200.jpg')

# Run model.
[rimg, rscores, rclasses, rbboxes, rnum] = \
    sess.run([image_4d, fscores, fclasses, fbboxes, fnum], feed_dict={img_input: img}) #fclasses, fbboxes, fnum

print rscores
print rbboxes
print rclasses
print rnum
#print(rscores)

# Draw bboxes
img_bboxes = np.copy(ssd_vgg_preprocessing.np_image_unwhitened(rimg[0]))
bboxes_draw_on_img(img_bboxes, rclasses[0], rscores[0], rbboxes[0], colors_tableau, thickness=1)

fig = plt.figure(figsize = (12,12))
plt.imshow(img_bboxes)
plt.savefig("gg3.png")



