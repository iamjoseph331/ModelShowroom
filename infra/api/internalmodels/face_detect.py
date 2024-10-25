import os
import cv2
import time
import base64
import numpy as np
import onnxruntime as ort
from infra.common.fd_utils import cfg_blaze, PriorBox, decode, letterbox, decode_landm, py_cpu_nms
from infra.common.utils import data_uri_to_cv2_img
from domain.interface import result, point, Error
import josephlogging.log as log

DEBUG = False
logger = log.getLogger(__name__)
providers =  ['CPUExecutionProvider']
ort_session = None
priorbox = PriorBox(cfg_blaze, image_size=(256, 256))
priors = priorbox.forward()

def init(modelname):
    try:
        global ort_session
        ort_session = ort.InferenceSession(modelname, providers=providers)
        return True
    except Exception as e:
        logger.error(f'Error initializing model: {e}')
        return False

def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

def predict(image):
    img_raw = data_uri_to_cv2_img(logger, image)
    if img_raw is None:
        return Error(what='decoding error', where='base64 image')
    return inference(img_raw)

def detectxyxy(img):
    fd = inference(img)
    bbs = fd.bounding_box
    ret = [[bb[0].x, bb[0].y, bb[1].x, bb[1].y] for bb in bbs]
    return ret

def loosedetectxyxy(img, fx):
    bbs = detectxyxy(img)
    h,w = img.shape[:2]
    new_det = []
    for det in bbs:
        cw,ch = (det[0]+det[2])//2, (det[1]+det[3])//2
        dw,dh = (det[2]-det[0])*fx//2, (det[3]-det[1])*fx//2
        new_bb =  [max(0,cw-dw), max(0,ch-dh), min(w,cw+dw), min(h,ch+dh)]
        new_det.append([int(x) for x in new_bb])
    return new_det

def inference(img_raw, vis_thres=0.6):
    outlog = {}
    img_size = 256
    nms_thresh = 0.6

    f = os.path.join('infra', 'model', 'face_detect.onnx')
    if ort_session is None:
        if not init(f):
            return Error(f'FD Model {f}', 'initialization failed')

    img0 = np.uint8(img_raw)
    img = letterbox(img0, color=(104, 117, 123), new_shape=img_size, auto=False)[0]
    
    img = img.transpose(2, 0, 1).copy() #(256,256,3)BGR to (3,256,256)BGR
    data = np.expand_dims(img, axis=0).astype(np.float32) #(1,3,256,256)
    in_name = ort_session.get_inputs()[0].name # 'input'
    out_name = [ort_session.get_outputs()[x].name for x in range(len(ort_session.get_outputs()))] # 'output'

    t0 = time.time()
    loc, conf, landmk = ort_session.run(out_name, {in_name: data})
    outlog['speed'] = ('took {}'.format(time.time() - t0))
    
    cfg = cfg_blaze
    prior_data = priors
    boxes = decode(loc.squeeze(), prior_data, cfg['variance'])
    landms = decode_landm(landmk.squeeze(), prior_data, cfg['variance'], num_landmarks=106)

    # Apply softmax to the 'conf' array along axis=2
    conf = softmax(conf, axis=2)

    # Extract scores for the second class (index 1)
    scores = conf.squeeze()[:, 1]

    # Ignore low scores based on the threshold
    inds = np.where(scores > nms_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # Process the image
    img = np.float32(img_raw)  # [height, width, 3]
    img = img.transpose(2, 0, 1)  # [3, height, width]
    img = np.expand_dims(img, axis=0)  # [1, 3, height, width]

    # Determine the size for resizing
    r = max(img.shape[2], img.shape[3])
    off_y = (r - img.shape[2]) // 2
    off_x = (r - img.shape[3]) // 2

    # Resize back using scaling
    scale = np.array([r] * 4)
    boxes = boxes * scale - np.array([off_x, off_y] * 2)  # (height, width)

    # Since we're using NumPy, no need to convert to CPU or back to NumPy
    # boxes are already NumPy arrays
    # If needed, ensure they are in the desired data type
    boxes = boxes.astype(np.float32)

    scale_landm = np.array([r] * 212)
    landms = landms * scale_landm - np.array([off_x, off_y] * 106)  # (x, y)
    landms = landms.astype(np.float32)

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]#args.topk = 5000
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_thresh) #args.nms_threshold = 0.4
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    keep_top_k = 750 #args keep_top_k
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    bbs = []
    lmks = []
    txt = []
    scores = []
    dets = np.concatenate((dets, landms), axis=1)
    # show image
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "(({},{}),({},{})) Confidence: {:.4f}".format(int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4])
        x1,y1,x2,y2 = int(b[0]),int(b[1]),int(b[2]),int(b[3])
        bbs.append((point(x1,y1),point(x2,y2)))
        lmks.append([point(b[5+i*2],b[6+i*2]) for i in range(106)])
        txt.append(text)
        scores.append(float(b[4]))

        if DEBUG:
            b = list(map(int, b))
            cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if 'bbox' not in outlog:
                outlog['bbox'] = ''
            outlog['bbox'] += (f'{x1}, {y1}, {x2}, {y2}\n')
            cx = x1
            cy = y1 - 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            for i in range(106):
                cv2.circle(img_raw, (b[5+i*2], b[6+i*2]), 1, (0, 0, 255), 2)
    
    # save image
    if DEBUG:
        cv2.imwrite('output.jpg', img_raw)
    outlog['outcome'] = txt
    print(outlog)
    res = result(name='face_detect', bb=bbs, lms=lmks, scores=scores, imgtxt=txt, outstr=outlog)
    return res

if __name__ == '__main__':
    with open("test_img/0.jpg", "rb") as image_file:
        encode = base64.b64encode(image_file.read()).decode('utf-8')
    st = time.time()
    bb = predict(encode).bounding_box[0]
    print(bb[0].to_str(), bb[1].to_str())
    print(f'average took: {(time.time() - st)} s')
    # img = cv2.rectangle(cv2.imread("/Users/macbook/Code/MVP/JosephPlatform/test_img/0.jpg"), (bb[0].x, bb[0].y), (bb[1].x, bb[1].y), (0, 0, 255), 2)
    # cv2.imwrite('output.jpg', img)