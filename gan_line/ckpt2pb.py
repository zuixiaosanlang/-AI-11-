import argparse
import glob
import os
import time

import cv2
import tensorflow as tf
import numpy as np

from deshadower import DeWordShadower


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # output_node_names = "g_conv_img/BiasAdd,g_conv_mask/BiasAdd"
    output_node_names = "g_conv_img/BiasAdd"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


def prepare_image(img, test_w=-1, test_h=-1):
    if test_w > 0 and test_h > 0:
        img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC)
    return img / 255.0


def expand(im):
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    im = np.expand_dims(im, axis=0)
    return im


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


def pb_test(pb_path, input_img_dir, output_img_dir):
    '''
        :param pb_path:pb文件的路径
        :param image_path:测试图片的路径
        :return:
        '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for op in sess.graph.get_operations():
                print(op.name, [inp for inp in op.inputs])

            # 定义输入的张量名称,对应网络结构的输入张量
            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")

            # 定义输出的张量名称
            output_tensor_name_img = sess.graph.get_tensor_by_name("g_conv_img/BiasAdd:0")

            test_h = 792
            test_w = 612
            st = time.time()
            for image_filename in glob.glob(input_img_dir + '/*.jpg'):
                img = cv2.imread(image_filename, -1)
                src = img.copy()

                if src.shape[1] != test_w or src.shape[0] != test_h:
                    img = prepare_image(img, test_w, test_h)
                else:
                    img = np.float32(img) / 255.0

                img = expand(img)

                st = time.time()
                oimg = sess.run([output_tensor_name_img],
                                feed_dict={input_image_tensor: img,
                                           })
                print("Test time  = %.3f " % (time.time() - st))

                oimg = decode_image(oimg)

                if src.shape[1] != test_w or src.shape[0] != test_h:
                    oimg = cv2.resize(oimg, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_CUBIC)

                if not os.path.isdir(output_img_dir):
                    os.makedirs(output_img_dir)
                output_filename = "%s/%s.jpg" % (output_img_dir, os.path.splitext(os.path.basename(image_filename))[0])
                cv2.imwrite(output_filename, oimg)

            print("Test time  = %.3f " % (time.time() - st))


# checkpoint -> .pb
def ckpt2pb():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='Dataset/test_dataset/',
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

    if not os.path.exists("pd_model/g_kpt"):
        os.makedirs("pd_model/g_kpt")

    # save infer model ckpt
    deshadower = DeWordShadower(ARGS.model, ARGS.vgg_19_path, ARGS.use_gpu, ARGS.is_hyper)

    input_checkpoint = 'pd_model/g_kpt/g_lasted_model.ckpt'
    out_pb_path = "pd_model/frozen_model.pb"
    freeze_graph(input_checkpoint, out_pb_path)


ckpt2pb()

# test .pb
input_img_dir = 'Dataset/test_dataset_1/'
output_img_dir = 'Dataset/result/'
pb_path = "pd_model/frozen_model.pb"
# pb_test(pb_path, input_img_dir, output_img_dir)
