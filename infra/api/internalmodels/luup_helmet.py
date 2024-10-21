import os
import math
import numpy as np
import concurrent.futures
import onnxruntime as ort
from PIL import Image, ImageOps
from infra.common.fd_utils import letterbox
from domain.interface import result, Error, point
from infra.api.internalmodels.face_detect import loosedetectxyxy
from infra.common.utils import data_uri_to_cv2_img, mean_image_subtraction, pad_to_square
import josephlogging.log as log

DEBUG = False

logger = log.getLogger(__name__)
providers =  ['CPUExecutionProvider', 'CUDAExecutionProvider']
ort_session = {}
emotion_session = None
attribute_task = {
    'helmet': 'luuphelmet1018x1.onnx'
}

hat_class = {0: 'No Helmet', 1: 'Helmet'}

def init(task:str, modelname:str):
    try:
        global ort_session
        ort_session[task] = ort.InferenceSession(modelname, providers=providers)
        return True
    except Exception as e:
        logger.error(f'Error initializing model: {e}')
        return False

def get_attributes(img):
    # set orts
    for k,v in attribute_task.items():
        file_path = os.path.join('infra', 'model', v)
        if k not in ort_session:
            if not init(k, file_path):
                return Error(f'FA Model {file_path}', 'initialization failed')
    # preprocess
    img = pad_to_square(img)
    img256 = img.resize((256, 256)).convert('RGB')
    n_img = mean_image_subtraction(np.array(img256).astype(np.float32) / 255.0)
    n_img = n_img.transpose(2, 0, 1).copy() #(256,256,3)BGR to (3,256,256)BGR
    data = np.expand_dims(n_img, axis=0).astype(np.float32) #(1,3,256,256)
    res = {}
    # inference
    with concurrent.futures.ThreadPoolExecutor() as executor:
    # inference
        futures = {k: executor.submit(run_session, session, data) for k, session in ort_session.items()}
        for k, future in futures.items():
            res[k] = postprocess(k, future.result())
    return res

def run_session(session, data):
    in_name = session.get_inputs()[0].name
    out_name = [session.get_outputs()[x].name for x in range(len(session.get_outputs()))]
    output = session.run(out_name, {in_name: data})
    raw_score = output[0][0]
    return raw_score

def predict(base64_image: str): 
    rgb_image = data_uri_to_cv2_img(logger, base64_image)
    pilimg = Image.fromarray(rgb_image)
    bbs = loosedetectxyxy(img=rgb_image, fx=1.5)
    res_bbs = []
    ms = []
    confs = []
    for bb in bbs:
        tup = (point(x=bb[0],y=bb[1]), point(x=bb[2],y=bb[3]))
        res_bbs.append(tup)
    res = get_attributes(pilimg)
    for k,v in res.items():
        ms.append(v[0])
        confs.append(v[1])
    
    output = {'detected faces': len(bbs), 'attributes':ms, 'confidence':confs}
    if DEBUG: cv2.imwrite('letout.jpg', img)   
    return result(name='attributes', bb=res_bbs ,imgtxt=ms, outstr=output)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def postprocess(task:str, score: float):
    if task == 'helmet':
        res = softmax(score)
        res_class = np.argmax(res)
        ret_hat = hat_class[res_class]
        return ret_hat, round(float(res[res_class]), 4)
    else:
        return Error(what='unsupported attribute', where='face_attr.py')

def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

if __name__ == '__main__':
    img = Image.open("/Users/macbook/Code/MVP/JosephPlatform/test_img/6.jpg")
    try:
        bbs = loosedetectxyxy(img=img, fx=1.5)
        bb = bbs[0]
        img2 = img[bb[1]:bb[3],bb[0]:bb[2],:]
    except:
        img2 = img
    res = get_attributes(img)
    print(res)
    
    