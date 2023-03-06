from __future__ import division
from networks import *
from utils import *
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time

EPS = 1e-12


class DeWordShadower(object):
    def __init__(self, model_path, vgg_19_path, use_gpu, hyper):
        self.vgg_19_path = vgg_19_path
        self.model = model_path
        self.hyper = hyper
        self.channel = 64
        if use_gpu < 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)
        self.setup_model()

    def setup_model(self):
        # set up the model and define the graph
        with tf.variable_scope(tf.get_variable_scope()):
            # self.input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            # self.input = tf.placeholder(tf.float32, shape=[1, 792, 612, 3])
            self.input = tf.placeholder(tf.float32, shape=[1, 480, 480, 3])

            # build the model
            self.shadow_free_image = build_aggasatt_joint2(self.input, self.channel,
                                                           vgg_19_path=self.vgg_19_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.model)

        print("[i] contain checkpoint: ", ckpt)
        saver_restore = tf.train.Saver([var for var in tf.trainable_variables() if 'g_' in var.name])
        print('loaded ' + ckpt.model_checkpoint_path)
        saver_restore.restore(self.sess, ckpt.model_checkpoint_path)

        # 仅保持生成器ckpt
        saver = tf.train.Saver(max_to_keep=None)
        saver.save(self.sess, "pd_model/g_kpt/g_lasted_model.ckpt")

        sys.stdout.flush()

    def run(self, img):
        iminput = expand(img)
        st = time.time()
        imoutput = self.sess.run([self.shadow_free_image], feed_dict={self.input: iminput})
        print("Test time  = %.3f " % (time.time() - st))
        imoutput = decode_image(imoutput)
        return imoutput
