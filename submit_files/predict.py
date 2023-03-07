# 代码示例
# python predict.py [src_image_dir] [results]


import sys
import json
import time
import cv2
import numpy as np
import glob
import os


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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个boundingbox的面积
    order = scores.argsort()[::-1]  # boundingbox的置信度排序
    keep = []  # 用来保存最后留下来的boundingbox
    while order.size > 0:
        i = order[0]  # 置信度最高的boundingbox的index
        keep.append(i)  # 添加本次置信度最高的boundingbox的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    out = np.array(keep, dtype=int)
    return out


def non_max_suppression_np(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    # output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output = [np.zeros((0, 6 + nm), dtype=np.float32)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            # v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v = np.zeros((len(lb), nc + nm + 5), dtype=np.float32)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            # x = torch.cat((x, v), 0)
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            # x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            # conf, j = x[:, 5:mi].max(1, keepdim=True)
            conf, j = np.max(x[:, 5:mi], 1)[:, np.newaxis], np.argmax(x[:, 5:mi], 1)[:, np.newaxis]
            # x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            cls = x[:, 5:6].astype(np.int32).reshape(-1)
            sel_index = []
            for s in range(cls.shape[0]):
                if cls[s] in classes:
                    sel_index.append(s)
            x = x[sel_index]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            # x = x[x[:, 4].argsort(descending=True)]  # sort by confidence
            x = x[x[:, 4].argsort()[::-1]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    test_h = 480
    test_w = 480

    import paddle.inference as pdi

    config = pdi.Config('best_paddle_model/inference_model/model.pdmodel',
                        'best_paddle_model/inference_model/model.pdiparams')
    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
    predictor = pdi.create_predictor(config)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    output_names = predictor.get_output_names()

    config_gan = pdi.Config('pd_model/inference_model/model.pdmodel',
                            'pd_model/inference_model/model.pdiparams')
    config_gan.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
    predictor_gan = pdi.create_predictor(config_gan)
    input_handle_gan = predictor_gan.get_input_handle(predictor_gan.get_input_names()[0])
    output_names_gan = predictor_gan.get_output_names()

    result = {}
    label_names = ["spanning_cell", "row", "column", "table"]

    for image_path in image_paths:
        filename = os.path.split(image_path)[1]

        # do something
        im0 = cv2.imread(image_path)

        pad = 32
        h, w, c = im0.shape
        new_h, new_w = h + 2 * pad, w + 2 * pad
        img_pad = np.zeros(shape=(new_h, new_w, 3), dtype=np.uint8)
        img_pad[pad:pad + h, pad:pad + w] = im0

        src = img_pad.copy()
        im = cv2.resize(np.float32(img_pad), (test_w, test_h), cv2.INTER_CUBIC)

        im = im.astype(np.float32) / 255.0

        # gan
        input_handle_gan.copy_from_cpu(im[None])
        predictor_gan.run()
        pred_gan = [predictor_gan.get_output_handle(x).copy_to_cpu() for x in output_names_gan]
        out_gan = decode_image(pred_gan)

        # yolov5 only line
        im[out_gan[:, :, 0] > 127] = (0, 0, 1)

        im2 = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im2 = np.ascontiguousarray(im2)

        input_handle.copy_from_cpu(im2[None])
        predictor.run()
        pred = [predictor.get_output_handle(x).copy_to_cpu() for x in output_names]

        pred = non_max_suppression_np(pred[0], 0.5, 0.45)

        fx = src.shape[1] / test_w
        fy = src.shape[0] / test_h

        if filename not in result:
            result[filename] = []

        box = [1000, 1000, 0, 0]
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes([test_h, test_w], det[:, :4], im.shape).round()

                for i, det0 in enumerate(det):
                    x0, y0, x1, y1 = det0[:4]
                    conf = det0[4]
                    cls = det0[5]

                    if int(cls) == 0 and conf < 0.6:
                        continue

                    obj_box = [x0 * fx - pad, y0 * fy - pad, x1 * fx - pad, y1 * fy - pad]
                    obj_box[0] = np.clip(obj_box[0], 0, im0.shape[1])  # x1, x2
                    obj_box[2] = np.clip(obj_box[2], 0, im0.shape[1])  # x1, x2
                    obj_box[1] = np.clip(obj_box[1], 0, im0.shape[0])  # y1, y2
                    obj_box[3] = np.clip(obj_box[3], 0, im0.shape[0])  # y1, y2

                    result[filename].append({
                        "box": obj_box,
                        "label": label_names[int(cls)]
                    })

                    box[0] = min(obj_box[0], box[0])
                    box[1] = min(obj_box[1], box[1])
                    box[2] = max(obj_box[2], box[2])
                    box[3] = max(obj_box[3], box[3])

        result[filename].append({
            "box": box,
            "label": label_names[3]
        })

    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)
