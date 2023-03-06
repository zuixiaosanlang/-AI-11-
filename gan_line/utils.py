import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os,time,cv2,scipy.io,random
from PIL import Image
from PIL import ImageEnhance,ImageFilter
from networks import build_vgg19, build_vgg19_2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def prepare_data(train_path, stage=['train_A']):
    input_names=[]
    image1=[]
    for dirname in train_path:
        for subfolder in stage:
            train_b = dirname + "/"+ subfolder+"/"
            for root, _, fnames in sorted(os.walk(train_b)):
                for fname in fnames:
                    if is_image_file(fname):
                        input_names.append(os.path.join(train_b, fname))
    return input_names
    
def decode_image(img,resize=False,sz=(640,480)):
    imw,imh = sz
    img = np.squeeze(np.minimum(np.maximum(img,0.0),1.0))
    if resize:
        img = resize_to_test(img,sz=(imw,imh))
    img = np.uint8(img*255.0)
    if len(img.shape) ==2:
        return np.repeat(np.expand_dims(img,axis=2),3,axis=2)
    else:
        return img

def expand(im):
  if len(im.shape) == 2:
    im = np.expand_dims(im,axis=2)
  im = np.expand_dims(im,axis=0)
  return im


def resize_to_test(img,sz=(640,480)):
  imw,imh = sz
  return cv2.resize(np.float32(img),(imw,imh),cv2.INTER_NEAREST)


def encode_image(img_path,sz=(256,256),resize=True):
  # print("img_path: " + img_path)
  imw,imh = sz
  input_image = cv2.imread(img_path)
  
  if resize:
    input_image=cv2.resize(np.float32(input_image),(imw,imh), cv2.INTER_NEAREST)

  return input_image/255.0


def np_scale_to_shape(image, shape, align=True):
    """Scale the image.

    The minimum side of height or width will be scaled to or
    larger than shape.

    Args:
        image: numpy image, 2d or 3d
        shape: (height, width)

    Returns:
        numpy image
    """
    height, width = shape
    imgh, imgw = image.shape[0:2]
    if imgh < height or imgw < width or align:
        scale = np.maximum(height/imgh, width/imgw)
        image = cv2.resize(
            image,
            (math.ceil(imgw*scale), math.ceil(imgh*scale)))
    return image


def np_random_crop(image, imtarget, shape, random_h=None, random_w=None, align=True):
    """Random crop.

    Shape from image.

    Args:
        image: Numpy image, 2d or 3d.
        shape: (height, width).
        random_h: A random int.
        random_w: A random int.

    Returns:
        numpy image
        int: random_h
        int: random_w

    """
    height, width = shape
    image = np_scale_to_shape(image, shape, align=align)
    imtarget = np_scale_to_shape(imtarget, shape, align=align)
    imgh, imgw = image.shape[0:2]
    if random_h is None:
        random_h = np.random.randint(imgh-height+1)
    if random_w is None:
        random_w = np.random.randint(imgw-width+1)
    return image[random_h:random_h+height, random_w:random_w+width, :], imtarget[random_h:random_h+height, random_w:random_w+width, :]


# dataload for images
def parpare_image(val_path,sz=(640,480),da=False,stage=['_M','_T','_B']):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))

  # val_path='Dataset/inpainting/train_C/dehw_train_00744.jpg'
  val_path_ = val_path.replace('.jpg', '.png')
  imtarget = encode_image(val_path_.replace('_A',stage[1]),(imw,imh))
  gtmask = encode_image(val_path_.replace('_A',stage[2]),(imw,imh))

  gtmask = np.expand_dims(gtmask[:, :, 0], axis=-1)

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,imtarget,gtmask] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,imtarget,gtmask = [np.rot90(x,_c) for x in [iminput,imtarget,gtmask] ]

  iminput,imtarget,gtmask = [expand(x) for x in (iminput,imtarget,gtmask) ]

  return iminput,imtarget,gtmask

def parpare_image2(val_path,sz=(640,480),da=False,stage=['_M','_T','_B']):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))

  # val_path='Dataset/inpainting/train_C/dehw_train_00744.jpg'
  # val_path_ = val_path.replace('.jpg', '.png')
  split_path = os.path.split(val_path)
  imtarget = encode_image(split_path[0].replace('_A',stage[1]) + "/" + split_path[1],(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,imtarget = [cv2.flip(x,_c) for x in [iminput,imtarget] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,imtarget = [np.rot90(x,_c) for x in [iminput,imtarget] ]

  iminput,imtarget = [expand(x) for x in (iminput,imtarget) ]

  return iminput,imtarget


def parpare_image3(img_path, json_data, sz=(640, 480), da=False, pad=32):
    img_path_split = os.path.split(img_path)
    imw, imh = sz

    src = cv2.imread(img_path)
    iminput = cv2.resize(np.float32(src), (imw, imh)) / 255.0

    fx = imw / src.shape[1]
    fy = imh / src.shape[0]

    mask = np.zeros_like(iminput)
    data_list = list(json_data[img_path_split[1]])

    label_names = ["table", "row", "column", "spanning_cell"]
    spanning_cell_list = []
    for i in range(len(data_list)):
        data = data_list[i]

        box = data["box"]
        label = data["label"]

        new_box = [0.0, 0.0, 0.0, 0.0]
        new_box[0] = int((box[0] + pad) * fx)
        new_box[1] = int((box[1] + pad) * fy)
        new_box[2] = int((box[2] + pad) * fx)
        new_box[3] = int((box[3] + pad) * fy)

        label_index = int(label_names.index(label))

        if label_index != 3:
            cv2.rectangle(mask, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (255, 255, 255), 2)

        if label_index == 3:
            spanning_cell_list.append(new_box)

    for span in spanning_cell_list:
        x_min, y_min, x_max, y_max = span
        pre = 1
        x_min = x_min + pre
        y_min = y_min + pre
        x_max = x_max - pre
        y_max = y_max - pre
        mask[y_min:y_max, x_min:x_max, :] = 0

    imtarget = np.float32(mask) / 255.0

    if da:
        if np.random.random_sample() > 0.75:
            _c = random.choice([-1, 0, 1])
            # data augumentation
            iminput, imtarget = [cv2.flip(x, _c) for x in [iminput, imtarget]]

        if imw == imh:
            # rotate
            _c = random.choice([0, 1, 2, 3])
            # data augumentation
            iminput, imtarget = [np.rot90(x, _c) for x in [iminput, imtarget]]

    iminput, imtarget = [expand(x) for x in (iminput, imtarget)]

    return iminput, imtarget


# 行线、列线、合并单元格 各在一个通道
def parpare_image4(img_path, json_data, sz=(640, 480), da=False, pad=32):
    img_path_split = os.path.split(img_path)
    imw, imh = sz

    src = cv2.imread(img_path)
    iminput = cv2.resize(np.float32(src), (imw, imh)) / 255.0

    fx = imw / src.shape[1]
    fy = imh / src.shape[0]

    col_mask = np.zeros(shape=iminput.shape[:2], dtype=np.uint8)
    row_mask = np.zeros(shape=iminput.shape[:2], dtype=np.uint8)
    span_mask = np.zeros(shape=iminput.shape[:2], dtype=np.uint8)

    data_list = list(json_data[img_path_split[1]])

    label_names = ["table", "row", "column", "spanning_cell"]
    spanning_cell_list = []
    for i in range(len(data_list)):
        data = data_list[i]

        box = data["box"]
        label = data["label"]

        new_box = [0.0, 0.0, 0.0, 0.0]
        new_box[0] = int((box[0] + pad) * fx)
        new_box[1] = int((box[1] + pad) * fy)
        new_box[2] = int((box[2] + pad) * fx)
        new_box[3] = int((box[3] + pad) * fy)

        label_index = int(label_names.index(label))

        if label_index != 3:
            # cv2.rectangle(mask, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (255, 255, 255), 2)
            cv2.line(col_mask, (new_box[0], new_box[1]), (new_box[0], new_box[3]), 255, 2)
            cv2.line(col_mask, (new_box[2], new_box[1]), (new_box[2], new_box[3]), 255, 2)

            cv2.line(row_mask, (new_box[0], new_box[1]), (new_box[2], new_box[1]), 255, 2)
            cv2.line(row_mask, (new_box[0], new_box[3]), (new_box[2], new_box[3]), 255, 2)

        if label_index == 3:
            spanning_cell_list.append(new_box)

    for span in spanning_cell_list:
        x_min, y_min, x_max, y_max = span
        row_mask[(y_min + 1):(y_max - 1), (x_min + 1):(x_max - 1)] = 0
        col_mask[(y_min + 1):(y_max - 1), (x_min + 1):(x_max - 1)] = 0

        pre = 4
        x_min = x_min + pre
        y_min = y_min + pre
        x_max = x_max - pre
        y_max = y_max - pre
        cv2.rectangle(span_mask, (x_min, y_min), (x_max, y_max), 255, -1)

    mask = np.concatenate([col_mask[:, :, np.newaxis], row_mask[:, :, np.newaxis], span_mask[:, :, np.newaxis]], axis=2)

    imtarget = np.float32(mask) / 255.0

    if da:
        if np.random.random_sample() > 0.75:
            _c = random.choice([-1, 0, 1])
            # data augumentation
            iminput, imtarget = [cv2.flip(x, _c) for x in [iminput, imtarget]]

        if imw == imh:
            # rotate
            _c = random.choice([0, 1, 2, 3])
            # data augumentation
            iminput, imtarget = [np.rot90(x, _c) for x in [iminput, imtarget]]

    iminput, imtarget = [expand(x) for x in (iminput, imtarget)]

    return iminput, imtarget


def parpare_image_randcup(val_path, sz=(640,480), da=False, stage=['_M','_T','_B']):
  iminput = cv2.imread(val_path) / 255.0
  imtarget = cv2.imread(val_path.replace('_A',stage[1])) / 255.0

  if da:
    # if np.random.random_sample() > 0.75:
    #   _c = random.choice([-1,0,1])
    #   # data augumentation
    #   iminput,imtarget = [cv2.flip(x,_c) for x in [iminput,imtarget] ]
    #
    # if imw == imh:
    #   # rotate
    #   _c = random.choice([0,1,2,3])
    #   # data augumentation
    #   iminput,imtarget = [np.rot90(x,_c) for x in [iminput,imtarget] ]

    if np.random.random_sample() > 0.75:
        iminput, imtarget = np_random_crop(iminput, imtarget, sz, random_h=None, random_w=None, align=False)
    else:
        neww = np.random.randint(256, 300)  # w is the longer width[]
        newh = round((neww / iminput.shape[1]) * iminput.shape[0])
        iminput = encode_image(val_path, (neww, newh))
        imtarget = encode_image(val_path.replace('_A', stage[1]), (neww, newh))

  iminput,imtarget = [expand(x) for x in (iminput,imtarget) ]

  return iminput,imtarget


# dataload for synthesized images
def parpare_image_syn(val_path,sz=(640,480),da=False,stage='train_shadow_free'):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))
  val_mask_name = val_path.split('/')[-1].split('_')[-1]
  gtmask = encode_image(val_path.replace(stage,'train_B').replace(val_path.split('/')[-1],val_mask_name),(imw,imh))

  val_im_name = '_'.join(val_path.split('/')[-1].split('_')[0:-1])+'.jpg'
  imtarget = encode_image(val_path.replace(stage,'shadow_free').replace(val_path.split('/')[-1],val_im_name),(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,imtarget,gtmask] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,imtarget,gtmask = [np.rot90(x,_c) for x in [iminput,imtarget,gtmask] ]

  iminput,imtarget,gtmask = [expand(x) for x in (iminput,imtarget,gtmask) ]

  return iminput,imtarget,gtmask

#### LOSSES
def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))

def compute_percep_loss(input, output, reuse=False, vgg_19_path='None'):
    vgg_real=build_vgg19(output*255.0,vgg_path=vgg_19_path,reuse=reuse)
    vgg_fake=build_vgg19(input*255.0,vgg_path=vgg_19_path,reuse=True)
    p0=compute_l1_loss(vgg_real['input'],vgg_fake['input'])
    p1=compute_l1_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/2.6
    p2=compute_l1_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/4.8
    p3=compute_l1_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/3.7
    p4=compute_l1_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/5.6
    p5=compute_l1_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/1.5
    return p0+p1+p2+p3+p4+p5


def compute_percep_loss_2(input, output, reuse=False, vgg_19_path='None'):
    vgg_real=build_vgg19_2(output*255.0,vgg_path=vgg_19_path,reuse=reuse)
    vgg_fake=build_vgg19_2(input*255.0,vgg_path=vgg_19_path,reuse=True)
    p0=compute_l1_loss(vgg_real['input'],vgg_fake['input'])
    p1=compute_l1_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/2.6
    p2=compute_l1_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/4.8
    # p3=compute_l1_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/3.7
    # p4=compute_l1_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/5.6
    # p5=compute_l1_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/1.5
    return p0+p1+p2


def parpare_image_fake_generator(val_path,im_mask_path,sz=(640,480)):

  imw,imh = sz
  immask  = encode_image(im_mask_path,(imw,imh))
  imshadowfree = encode_image(val_path,(imw,imh))

  imshadowfree,immask = [expand(x) for x in (imshadowfree,immask) ]
  
  return imshadowfree,immask

