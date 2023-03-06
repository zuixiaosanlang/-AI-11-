import json

import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def dataproc_table_structure():
    JSON = "G:/work/for_fun/fei_jiang/table_structure/train/annos.txt"
    src_dir = "G:/work/for_fun/fei_jiang/table/Dataset/inpainting/train_A/"
    src_mask_dir = 'G:/work/for_fun/fei_jiang/table_structure/table_structure/Dataset/result/'  # 480*480

    dst_imgs_dir = "G:/work/for_fun/fei_jiang/table_structure/yolov5-7.0/dataset/table/images/"
    dst_label_dir = "G:/work/for_fun/fei_jiang/table_structure/yolov5-7.0/dataset/table/labels/"

    if not os.path.exists(dst_imgs_dir):
        os.makedirs(dst_imgs_dir)
    if not os.path.exists(dst_label_dir):
        os.makedirs(dst_label_dir)

    test_w = 480
    test_h = 480
    pad = 32

    label_names = ["spanning_cell", "row", "column", "table"]
    with open(JSON) as infile:
        json_data = json.load(infile)

        image_names = list(json_data.keys())
        for image_name in tqdm(image_names):
            img_path = src_dir + image_name
            img = cv2.imread(img_path)

            fx = img.shape[1] / test_w
            fy = img.shape[0] / test_h

            img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC)

            mask_path = src_mask_dir + image_name
            mask = cv2.imread(mask_path)

            img[mask[:, :, 2] > 128] = (0, 0, 255)

            img[mask[:, :, 0] > 128] = (255, 0, 0)
            img[mask[:, :, 1] > 128] = (0, 255, 0)

            data_list = list(json_data[image_name])

            # save label
            out_label_file = open(dst_label_dir + image_name.split('.')[0] + '.txt', 'w')
            for i in range(len(data_list)):
                data = data_list[i]

                box = data["box"]
                label = data["label"]

                cls_id = label_names.index(label)
                x0, y0, x1, y1 = box
                x0, y0, x1, y1 = (x0 + pad) / fx, (y0 + pad) / fy, (x1 + pad) / fx, (y1 + pad) / fy

                if cls_id == 3:
                    continue
                yolov5_label = convert([test_w, test_h], [x0, y0, x1, y1])
                out_label_file.write(str(cls_id) + " " + " ".join([str(a) for a in yolov5_label]) + '\n')
            out_label_file.close()

            # save img
            save_path = dst_imgs_dir + image_name
            cv2.imwrite(save_path, img)


# dataproc_table_structure()

def cal_ac():
    import utils.autoanchor as autoAC

    # 对数据集重新计算 anchors
    new_anchors = autoAC.kmean_anchors('./data/table.yaml', 9, 480, 6.0, 1000, True)
    print(new_anchors)


cal_ac()
