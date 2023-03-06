# 代码示例
# python paddle_infer.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2

import paddle
import math

import cv2
import numpy as np
import glob
import os
from pd_model.x2paddle_code import TFModel


def resize_to_test(img, sz=(640, 480)):
    imw, imh = sz
    return cv2.resize(np.float32(img), (imw, imh), cv2.INTER_CUBIC)


def decode_image(img, resize=False, sz=(640, 480)):
    imw, imh = sz
    img = np.squeeze(np.minimum(np.maximum(img, 0.0), 1.0))
    if resize:
        img = resize_to_test(img, sz=(imw, imh))
    img = np.uint8(img * 255.0)
    if len(img.shape) == 2:
        return np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    else:
        return img


def process_src(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    test_h = 480
    test_w = 480

    paddle.disable_static()
    params = paddle.load(r'pd_model/model.pdparams')
    model = TFModel()
    model.set_dict(params, use_structured_name=False)
    model.eval()

    result = {}

    for image_path in image_paths:
        filename = os.path.split(image_path)[1]

        # do something
        img = cv2.imread(image_path)

        pad = 32
        h, w, c = img.shape
        new_h, new_w = h + 2 * pad, w + 2 * pad
        new_img = np.zeros(shape=(new_h, new_w, 3), dtype=np.uint8)
        new_img[pad:pad + h, pad:pad + w] = img
        # img = new_img.copy()

        src = new_img.copy()
        img = cv2.resize(np.float32(new_img), (test_w, test_h), cv2.INTER_CUBIC) / 255.0
        img = img[np.newaxis, :, :, :]
        img = paddle.to_tensor(img)

        oimg = model(img)
        oimg = decode_image(oimg.numpy())

        if filename not in result:
            result[filename] = []

        # _, mask = cv2.threshold(oimg[:, :, 0], 128, 255, cv2.THRESH_BINARY)
        _, mask = cv2.threshold(oimg, 128, 255, cv2.THRESH_BINARY)
        fx = src.shape[1] / test_w
        fy = src.shape[0] / test_h

    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    test_h = 480
    test_w = 480

    paddle.disable_static()
    params = paddle.load(r'pd_model/model.pdparams')
    model = TFModel()
    model.set_dict(params, use_structured_name=False)
    model.eval()

    for image_path in image_paths:
        filename = os.path.split(image_path)[1]

        # do something
        img = cv2.imread(image_path)

        src = img.copy()
        img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC) / 255.0
        img = img[np.newaxis, :, :, :]
        img = paddle.to_tensor(img)

        oimg = model(img)
        oimg = decode_image(oimg.numpy())

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        output_filename = "%s/%s.jpg" % (save_dir, os.path.splitext(os.path.basename(image_path))[0])
        cv2.imwrite(output_filename, oimg)


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)
