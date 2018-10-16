#!/usr/bin/python
# -*- coding:utf-8 -*-
# coding=utf-8
# Author: houkai
# Mail: houkai.hk@alibaba-inc.com
# Created Time: 2018-10-09 17:21
# Filename: ssd_tf_pinggu.py
# Description: 
#
import sys
import os, os.path
import cv2
import random

import argparse
import numpy as np
import tensorflow as tf

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
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        self.sess = tf.Session(graph=self.detection_graph, config=config)
        self.input_x = self.sess.graph.get_tensor_by_name("ssd_preprocessing_train/ToFloat:0")  # 具体名称看上一段代码的input.name
        self.pre_classes = self.sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fclasses:0")
        self.pre_bboxes = self.sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fbboxes:0")
        self.pre_scores = self.sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fscores:0")
        self.pre_num = self.sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fnum:0")
        
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
#         config = tf.ConfigProto()
#         config.gpu_options.per_process_gpu_memory_fraction = 0.25

#         with tf.Session(graph=self.detection_graph, config=config) as sess:
#             input_x = sess.graph.get_tensor_by_name("ssd_preprocessing_train/ToFloat:0")  # 具体名称看上一段代码的input.name
#             pre_classes = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fclasses:0")
#             pre_bboxes = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fbboxes:0")
#             pre_scores = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fscores:0")
#             pre_num = sess.graph.get_tensor_by_name("ssd_bboxes_class_select/result/fnum:0")

        (classes, bboxes, scores, num) = self.sess.run([self.pre_classes, self.pre_bboxes, self.pre_scores, self.pre_num], \
                                                      feed_dict={self.input_x: x_batch})
        return classes[0], scores[0], bboxes[0]


import time
from PIL import Image
def main(argv):
    '''测试函数'''
    pytf_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    # Required 参数
    parser.add_argument(
        "--root",
        default='VOCclothsonepiece',
        help="list of filenames"
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(pytf_dir, "./pb/frozen_lb.pb"),
        help="model path."
    )
    args = parser.parse_args()
    
    valfile = os.path.join('/mogu/liubang/mytf/', args.root, 'ImageSets/Main/val.txt')
    
    imageroot = os.path.join('/mogu/liubang/mytf/', args.root, 'JPEGImages')
    annoroot = os.path.join('/mogu/liubang/mytf/', args.root, 'Annotations')
    
    #输出文件
    outfiles = {}
    for i in range(1,7):
        filename = './pinggu/{}_{}_det.txt'.format(i, args.root)
        outfile = open(filename, 'w')
        outfiles[i] = outfile
    
    
    with open(valfile ,'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    
    # Make detector.
    detector = Convert(args.model_path)
    
    ii=0
    t0 = time.time()
    for imagename in imagenames:
        imagepath = "%s/%s.jpg" % (imageroot, imagename)
        if not os.path.exists(imagepath):
            continue
        #分析图片
        image = Image.open(imagepath).convert("RGB")
        image = np.array(image)
        image = image.astype(np.float32)
        h,w,_ = image.shape
        clses, scores, bboxes = detector.detect(image) #ymin, xmin, ymax, xmax
        #写到文件
        for i, score in enumerate(scores):
            if score < 0.01:
                break
            cls = clses[i]
            bbox = bboxes[i]
            ymin = int(min(max(bbox[0],0),1)*h + 0.5)
            xmin = int(min(max(bbox[1],0),1)*w + 0.5)
            ymax = int(min(max(bbox[2],0),1)*h + 0.5)
            xmax = int(min(max(bbox[3],0),1)*w + 0.5)
            outline = "%s %s %d %d %d %d\n" % (imagename, score, xmin, ymin, xmax, ymax)
            outfiles[cls].write(outline)
        ii += 1
        if ii == 10:
            ii = 0
            line = "use： @%.3fs" % (time.time() - t0)
            print line
            t0 = time.time()
    for v in outfiles.values():
        v.close()
    
if __name__ == '__main__':
    import sys
    main(sys.argv) 