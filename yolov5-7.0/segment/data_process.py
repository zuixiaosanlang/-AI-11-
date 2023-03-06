import json

import cv2
import numpy as np


def get_a_coco_pic():
    pic_path = r"dataset\coco128-seg\images\train2017\000000000009.jpg"
    txt_path = r"dataset\coco128-seg\labels\train2017\000000000009.txt"
    img = cv2.imread(pic_path)
    height, width, _ = img.shape
    print(height, width)
    # cv2.imshow("111", img)
    # cv2.waitKey()
    file_handle = open(txt_path)
    cnt_info = file_handle.readlines()
    new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]
    print(len(new_cnt_info))
    print("---====---")  # 45 bowl 碗 49 橘子 50 西兰花
    color_map = {"49": (0, 255, 255), "45": (255, 0, 255), "50": (255, 255, 0)}
    for new_info in new_cnt_info:
        print(new_info)
        s = []
        for i in range(1, len(new_info), 2):
            b = [float(tmp) for tmp in new_info[i:i + 2]]
            s.append([int(b[0] * width), int(b[1] * height)])
        print(s)
        cv2.polylines(img, [np.array(s, np.int32)], True, color_map.get(new_info[0]), 2)
        cv2.imshow('img2', img)
        cv2.waitKey()


# get_a_coco_pic()

def dataproc():
    JSON = "G:/work/for_fun/zuixiaosanlang_20220123/table_dataset/train/annos.txt"
    src_dir = "G:/work/for_fun/fei_jiang/table/Dataset/inpainting_kpt/train_A/"
    dst_images_dir = "dataset/table-seg/images/"
    dst_labels_dir = "dataset/table-seg/labels/"

    with open(JSON) as infile:
        json_data = json.load(infile)

        image_names = list(json_data.keys())
        for image_name in image_names:
            img_path = src_dir + image_name
            img = cv2.imread(img_path)

            pad = 32
            h, w, c = img.shape
            new_h, new_w = h + 2 * pad, w + 2 * pad
            new_img = np.zeros(shape=(new_h, new_w, 3), dtype=np.uint8)
            new_img[pad:pad + h, pad:pad + w] = img

            data_list = list(json_data[image_name])

            txt_path = dst_labels_dir + image_name.split('.')[0] + '.txt'
            txt_file = open(txt_path, 'w')
            for i in range(len(data_list)):
                data = data_list[i]
                kpt_lb = data["lb"]
                kpt_lt = data["lt"]
                kpt_rt = data["rt"]
                kpt_rb = data["rb"]

                pts = [kpt_lb, kpt_lt, kpt_rt, kpt_rb]

                points_nor_list = []

                for point in pts:
                    points_nor_list.append((point[0] + pad) / (w + 2 * pad))
                    points_nor_list.append((point[1] + pad) / (h + 2 * pad))

                points_nor_list = list(map(lambda x: str(x), points_nor_list))
                points_nor_str = ' '.join(points_nor_list)

                label_str = str(0) + ' ' + points_nor_str + '\n'
                txt_file.writelines(label_str)

            txt_file.close()

            save_path = dst_images_dir + image_name
            cv2.imwrite(save_path, new_img)


dataproc()
