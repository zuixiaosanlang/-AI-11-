from __future__ import division

import argparse

from deshadower import DeWordShadower
import cv2
import glob
import numpy as np
import os

import time

EPS = 1e-12


def prepare_image(img, test_w=-1, test_h=-1):
    if test_w > 0 and test_h > 0:
        img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC)
    return img / 255.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='Dataset/inpainting/train_A/',
                        help="path to sample images")
    parser.add_argument("--result_dir", default='Dataset/result/',
                        help="path to the result dir")
    parser.add_argument("--model", default='logs/pre-trained', type=str,
                        help="path to folder containing the model")
    parser.add_argument("--vgg_19_path", default='Models/imagenet-vgg-verydeep-19.mat', type=str,
                        help="path to vgg 19 path model")
    parser.add_argument("--use_gpu", default=0, type=int, help="which gpu to use")
    parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")

    ARGS = parser.parse_args()

    if not os.path.isdir("pd_model/g_kpt"):
        os.makedirs("pd_model/g_kpt")

    deshadower = DeWordShadower(ARGS.model, ARGS.vgg_19_path, ARGS.use_gpu, ARGS.is_hyper)

    if not os.path.isdir(ARGS.result_dir):
        os.makedirs(ARGS.result_dir)

    test_h = 480
    test_w = 480
    result = {}

    st = time.time()
    for image_filename in glob.glob(ARGS.input_dir + '/*.jpg'):
        print('process: ' + image_filename)
        img = cv2.imread(image_filename)
        src = img.copy()

        img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC) / 255.0

        oimg = deshadower.run(img)

        if not os.path.isdir(ARGS.result_dir):
            os.makedirs(ARGS.result_dir)
        output_filename = "%s/%s.jpg" % (ARGS.result_dir, os.path.splitext(os.path.basename(image_filename))[0])
        cv2.imwrite(output_filename, oimg)

    print("total time  = %.3f " % (time.time() - st))
