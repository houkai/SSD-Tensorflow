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
import argparse


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



class Convert:
    def __init__(self, frozen_graph_filename):
        '''
        :param frozen_graph_filename: pb文件
        '''
        # if tf.__version__ < '1.4.0':
        #     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    #tensorboard --logdir=logs/
    def savegraphlog(self):
        '''
        :return: 可以查看网络结构
        '''
        summary_writer = tf.summary.FileWriter('outs2/', self.detection_graph)
        summary_writer.flush()
        summary_writer.close()

    def tosavemodel(self):
        '''
        :return: 转化为savemodel
        '''
        with tf.Session(graph=self.detection_graph) as sess:
            builder = tf.saved_model.builder.SavedModelBuilder("mybuild")
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
            builder.save()

    def print_op(self):
        '''
        :return: 打印图结构
        '''

        with tf.Session(graph=self.detection_graph) as sess:

            print(sess.graph.as_graph_def())
            for op in tf.get_default_graph().get_operations():
                print(op.name, op.values())

    # def load_model(self):
    #     tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], "build")


    def detect(self, x_batch):
        sleeve_name = ['其他', '无袖', '短袖', '五分袖', '七分袖', '长袖']
        collar_name = ['其他', '西装领', '衬衫领', '一字领', '方领', '连帽', 'V领', '半高领', '高领', '圆领', '棒球领', '立领']
        with tf.Session(graph=self.detection_graph) as sess:
            input_x = sess.graph.get_tensor_by_name("ssd_preprocessing_train/ToFloat:0")  # 具体名称看上一段代码的input.name
            pre_classes = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fclasses:0")
            pre_bboxes = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fbboxes:0")
            pre_scores = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fscores:0")
            pre_num = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fnum:0")

            (classes, bboxes, scores, num) = sess.run([pre_classes, pre_bboxes, pre_scores, pre_num], feed_dict={input_x: x_batch})
            print scores
            print bboxes
            print classes
            print num

import time
from PIL import Image
def main(argv):
    '''测试函数'''
    pytf_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    # Required 参数
    parser.add_argument(
        "--input_file",
        default='/mogu/liubang/mytf/VOCclothsonepiece/JPEGImages/100091894_ifrtmytdmvrtcnlchazdambqhayde_800x1200.jpg',
        help="list of filenames"
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(pytf_dir, "./pb/frozen_lb.pb"),
        help="model path."
    )
    args = parser.parse_args()

    # Make detector.
    detector = Convert(args.model_path)
    #detector.tosavemodel()
    #detector.savegraphlog()
    #return
    
    image = Image.open(args.input_file).convert("RGB")
    image = np.array(image)
    image = image.astype(np.float32)
    print image.shape
    
    t0 = time.time()
    objRsts = detector.detect(image)
    line = "use： @%.3fs" % (time.time() - t0)
    print line
if __name__ == '__main__':
    import sys
    main(sys.argv)        
        




