#!/usr/bin/python
# -*- coding:utf-8 -*-
# coding=utf-8
# Author: houkai
# Mail: houkai.hk@alibaba-inc.com
# Created Time: 2018-09-29 11:23
# Filename: printparam.py
# Description: 
#
import sys
import os, os.path
import tensorflow as tf

if __name__ == '__main__':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model.ckpt-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        tvs = [v for v in tf.trainable_variables()]
        for v in tvs:
            print(v.name)
            print(sess.run(v))
