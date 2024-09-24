import cv2
import numpy as np
from math import ceil
from itertools import product as product

cfg_blaze = {
    'name': 'Blaze',
    # origin anchor
    # 'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
    # kmeans and evolving for 640x640
    'min_sizes': [[8, 11], [14, 19, 26, 38, 64, 149]], 
    'steps': [8, 16],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1,
    'cls_weight': 6,
    'landm_weight': 0.1, 
    'gpu_train': True,
    'batch_size': 256,
    'ngpu': 1,
    'epoch': 200,
    'decay1': 130,
    'decay2': 160,
    'decay3': 175,
    'decay4': 185,
    'image_size': 320,
    'num_classes':2
}

class PriorBox:
    def __init__(self, cfg, image_size=None, phase='train'):
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        anchors = np.array(anchors).reshape(-1, 4)
        if self.clip:
            anchors = np.clip(anchors, a_min=0.0, a_max=1.0)
        return anchors

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo the encoding.
    Args:
        loc: (numpy array) Location predictions, Shape [num_priors, 4].
        priors: (numpy array) Prior boxes, Shape [num_priors, 4].
        variances: (list of float) Variances for decoding.
    Returns:
        boxes: (numpy array) Decoded bounding boxes, Shape [num_priors, 4].
    """
    # Calculate decoded center coordinates
    boxes_cx = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    
    # Calculate decoded width and height
    boxes_wh = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    
    # Combine center coordinates and sizes
    boxes = np.concatenate([boxes_cx, boxes_wh], axis=1)
    
    # Convert from center-size to corner coordinates
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes

def decode_landm(pre, priors, variances, num_landmarks=5):
    """
    Decode landmarks from predictions using priors to undo the encoding.
    Args:
        pre: (numpy array) Landmark predictions, Shape [num_priors, 10].
        priors: (numpy array) Prior boxes, Shape [num_priors, 4].
        variances: (list of float) Variances for decoding.
        num_landmarks: (int) Number of landmarks.
    Returns:
        landms: (numpy array) Decoded landmarks, Shape [num_priors, 10].
    """
    landms = []
    for i in range(num_landmarks):
        landm = priors[:, :2] + pre[:, 2*i:2*i+2] * variances[0] * priors[:, 2:]
        landms.append(landm)
    landms = np.concatenate(landms, axis=1)
    return landms


def nms(boxes, scores, overlap=0.5, top_k=200):
    """
    Apply Non-Maximum Suppression (NMS) to avoid detecting too many overlapping boxes.
    Args:
        boxes: (numpy array) Bounding boxes, Shape [num_priors, 4].
        scores: (numpy array) Confidence scores, Shape [num_priors].
        overlap: (float) Overlap threshold for suppression.
        top_k: (int) Maximum number of boxes to consider.
    Returns:
        keep: (list) Indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Descending order

    if top_k > len(order):
        top_k = len(order)

    order = order[:top_k]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        current_box = boxes[i]
        rest_boxes = boxes[order[1:]]

        # Compute intersection
        xx1 = np.maximum(current_box[0], rest_boxes[:, 0])
        yy1 = np.maximum(current_box[1], rest_boxes[:, 1])
        xx2 = np.minimum(current_box[2], rest_boxes[:, 2])
        yy2 = np.minimum(current_box[3], rest_boxes[:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        # Compute IoU
        union = area[i] + area[order[1:]] - inter
        iou = inter / union

        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= overlap)[0]
        order = order[inds + 1]

    return keep


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep