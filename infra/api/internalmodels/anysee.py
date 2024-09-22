import time
import base64
import requests 
from requests.structures import CaseInsensitiveDict
from domain.interface import result, point, Error
from infra.common.utils import get_axis_points
from domain.utils import trim_base64_header
from config.config import cfg
import josephlogging.log as log

requests.packages.urllib3.util.connection.HAS_IPV6 = False
logger = log.getLogger(__name__)

def sendRequest(img, model='JCV_FACE_K25000', pos=False, ang=False, lnd=False, qua=False, att=False):
    try:
        endpoint = cfg['endpoints']['anysee_detect']
        url = endpoint['url']
        apikey = endpoint['apikey']
    except Exception as e:
        logger.error(f'error loading anysee config {e}')
        return Error(where='config',what=f'{__name__} not found')

    headers = CaseInsensitiveDict()
    headers['Content-Type'] = 'application/json'
    headers['api-key'] = apikey

    data = {}
    image = {}
    rd = {}
    rd["position"] = pos
    rd["angle"] = ang
    rd["landmarks"] = lnd
    rd["quality"] = qua
    rd["attributes"] = att
    image['data'], _ = trim_base64_header(logger, img)
    image['autoRotate'] = True
    image['returnDetails'] = rd
    data['model'] = model
    data['image'] = image

    try:
        res = requests.post(url, json=data, headers=headers, timeout=10)
    except Exception as e:
        logger.warning(f'Anysee timeout! {e}')
        res = Error('Anysee endpoint', 'timeout')
    return res

def sendCompareRequest(img1, img2, model='JCV_FACE_K25000', pos=False, ang=False, lnd=False, qua=False, att=False):
    try:
        endpoint = cfg['endpoints']['anysee_compare']
        url = endpoint['url']
        apikey = endpoint['apikey']
    except Exception as e:
        logger.error(f'error loading anysee config: {e}')
        return Error(where='config',what=f'{__name__} not found')

    headers = CaseInsensitiveDict()
    headers['Content-Type'] = 'application/json'
    headers['api-key'] = apikey

    data = {}
    image = {}
    rd = {}
    rd["position"] = pos
    rd["angle"] = ang
    rd["landmarks"] = lnd
    rd["quality"] = qua
    rd["attributes"] = att
    image['data'], _ = trim_base64_header(logger, img1)
    image['autoRotate'] = True
    image['returnDetails'] = rd
    data['model'] = model
    data['imageOne'] = image
    image2 = image.copy()
    image2['data'], _ = trim_base64_header(logger, img2)
    data['imageTwo'] = image2

    try:
        res = requests.post(url, json=data, headers=headers, timeout=10)
    except Exception as e:
        logger.warning(f'Anysee timeout! {e}')
        res = Error('Anysee endpoint', 'timeout')
    return res
    
def face_attributes(image: str):
    bounding_boxes = []
    image_text = []
    res = sendRequest(image, pos=True, att=True)
    out_str = 'N/A'
    try:
        entities = res.json()['entities']
        for e in entities:
            bb = e['details']['position']
            p1 = point(bb['left'],bb['top'])
            p2 = point(bb['left'] + bb['width'], bb['top'] + bb['height'])
            bounding_boxes.append((p1, p2))
            out_str = e['details']['attributes']
            age = out_str['age']['value']
            gender = out_str['gender']['value']
            mask = out_str['mask']['value']

            emotions = out_str['emotions']
            max_emotion = emotions[0]
            for e in emotions:
                if e['certainty'] > max_emotion['certainty']:
                    max_emotion = e
            emotion = max_emotion['value']

            image_text.append(f'Age: {age}, Gender: {gender}, emotion: {emotion}, {mask}')
        out_str['Endpoint'] = cfg['endpoints']['anysee_detect']['url'] 
    except Exception as e:
        out_str = res.json()
        logger.error(f'Face Attributes Error!{e}')
    
    return result(name='Anysee', bb=bounding_boxes, imgtxt=image_text, outstr=out_str)

def face_detection(image: str):
    bounding_boxes = []
    image_text = []
    lmks = []
    res = sendRequest(image, pos=True, lnd=True)
    out_str = res.json()
    try:
        entities = res.json()['entities']
        for e in entities:
            bb = e['details']['position']
            p1 = point(bb['left'],bb['top'])
            p2 = point(bb['left'] + bb['width'], bb['top'] + bb['height'])
            bounding_boxes.append((p1, p2))
            image_text.append((p1.to_str(), p2.to_str()))
            lm = e['details']['landmarks']
            lmks.append([point(l['x'],l['y']) for l in lm])
        out_str['Endpoint'] = cfg['endpoints']['anysee_detect']['url'] 
    except Exception as e:
        logger.error(f'Face Detection Error!{e}')
    return result(name='Anysee', bb=bounding_boxes, imgtxt=image_text, lms=lmks, outstr=out_str)

def face_pose(image: str):
    bounding_boxes = []
    image_text = []
    res = sendRequest(image, pos=True, ang=True)
    lms = [[],[],[]]
    out_str = 'N/A'
    try:
        entities = res.json()['entities']
        for e in entities:
            bb = e['details']['position']
            p1 = point(bb['left'],bb['top'])
            p2 = point(bb['left']+bb['width'], bb['top']+bb['height'])
            bounding_boxes.append((p1, p2))

            out_str = e['details']['angle']
            yaw = out_str['yaw']
            pitch = out_str['pitch']
            roll = out_str['roll']

            axis = get_axis_points(yaw,pitch,roll,bb['left']+bb['width']/2, bb['top']+bb['height']/2,point_cnt=10)
            #img = data_uri_to_cv2_img(logger, image)
            #cv2.imwrite('out.jpg', draw_axis(img, yaw, pitch, roll))
            lms[0] += axis[0]
            lms[1] += axis[1]
            lms[2] += axis[2]
            image_text.append(f'Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}')
        out_str['Endpoint'] = cfg['endpoints']['anysee_detect']['url'] 
    except Exception as e:
        out_str = res.json()
        logger.error(f'Face Pose Error! {e}')
    return result(name='Anysee', bb=bounding_boxes, imgtxt=image_text, lms=lms, outstr=out_str)

def face_quality(image: str):
    bounding_boxes = []
    image_text = []
    res = sendRequest(image, pos=True, ang=True, att=True, qua=True)
    out_str = {}
    outcome_str = []
    try:
        entities = res.json()['entities']
        cnt = 1
        for e in entities:
            bb = e['details']['position']
            p1 = point(bb['left'],bb['top'])
            p2 = point(bb['left'] + bb['width'], bb['top'] + bb['height'])
            bounding_boxes.append((p1, p2))
            out_str[f'entity #{cnt}'] = e['details']['quality']
            attr = e['details']['attributes']

            angle = e['details']['angle']
            yaw = angle['yaw']
            pitch = angle['pitch']
            roll = angle['roll']
            check, outcome = check_quality(out_str[f'entity #{cnt}'], attr, yaw, pitch, roll, len(entities))
            outcome_str.append(f'{outcome}\n') 
            res = 'Pass' if check else 'Fail'
            image_text.append(f'Quality: {res}')
            cnt += 1
        out_str['Endpoint'] = cfg['endpoints']['anysee_detect']['url']
    except Exception as e:
        out_str = res.json()
        logger.error(f'Face Quality Error! {e}')
    out_str['outcome'] = outcome_str
    return result(name='Anysee', bb=bounding_boxes, imgtxt=image_text, outstr=out_str)

def face_recognition(image1: str, image2: str, model='JCV_FACE_K25000'):
    bounding_boxes = []
    image_text = []
    res = sendCompareRequest(image1, image2, model=model, pos=True)
    out_str = res.json()
    try:
        entities = res.json()['entities']
        for e in entities:
            ent = entities[e]
            bb = ent['details']['position']
            p1 = point(bb['left'],bb['top'])
            p2 = point(bb['left'] + bb['width'], bb['top'] + bb['height'])
            bounding_boxes.append((p1, p2))

            score = res.json()['similarityScore']
            image_text.append(f'Similarity Score: {score}')
        out_str['Endpoint'] = cfg['endpoints']['anysee_compare']['url']
    except Exception as e:
        out_str = res.json()
        logger.error(f'Face Recognition Error! {e}')
    
    return result(name='Anysee', bb=bounding_boxes, imgtxt=image_text, outstr=out_str)

def check_quality(qual, attr, yaw, pitch, roll, count):
    try:
        brightness = qual['brightness']
        sharpness = qual['sharpness']
        mouthClosed = qual['mouthClosed']
        size = qual['size']
        integrity = qual['integrity']
        completeness = qual['completeness']['total']
        mouth = qual['completeness']['mouth']
        lEye = qual['completeness']['leftEye']
        rEye = qual['completeness']['rightEye']
        lBrow = qual['completeness']['leftEyeBrow']
        rBrow = qual['completeness']['rightEyeBrow']
        nose = qual['completeness']['nose']
        contour = qual['completeness']['faceContour']
        mask = attr['mask']['value']
        hat = attr['hat']['value']
        glasses = attr['glasses']['value']
    except Exception as e:
        logger.error(f'Parse Quality Fail! {e}')
        return False, f'{e}'

    check_list = {}
    check_list['count = 1'] = (count == 1)
    check_list['-15 <= yaw <= 15'] = ((yaw <= 15) & (yaw >= -15))
    check_list['-20 <= pitch <= 20'] = ((pitch <= 20) & (pitch >= -20))
    check_list['-15 <= roll <= 15'] = ((roll <= 15) & (roll >= -15))
    check_list['-0.5 <= brightness <= 0.5'] = ((brightness <= 0.5) & (brightness >= -0.5))
    check_list['sharpness >= 0.8'] = (sharpness >= 0.8)
    check_list['mouthClosed >= 0.6'] = (mouthClosed >= 0.6)
    check_list['0 <= size <= 0.85'] = ((size <= 0.85) & (size >= 0))
    check_list['integrity = 1'] = (integrity == 1.0)
    check_list['total completeness >= 0.9'] = (completeness >= 0.9)
    check_list['mouth completeness >= 0.9'] = (mouth >= 0.9)
    check_list['leftEye completeness >= 0.9'] = (lEye >= 0.9)
    check_list['rightEye completeness >= 0.9'] = (rEye >= 0.9)
    check_list['leftEyeBrow completeness >= 0.8'] = (lBrow >= 0.8)
    check_list['rightEyeBrow completeness >= 0.8'] = (rBrow >= 0.8)
    check_list['nose completeness >= 0.9'] = (nose >= 0.9)
    check_list['face contour completeness >= 0.9'] = (contour >= 0.9)
    check_list['without mask'] = (mask == 'without_mask')
    check_list['without hat'] = (hat == 'without_hat')
    check_list['no sunglasses allowed'] = (glasses != 'with_sunglasses')

    q_pass = True
    outcome = 'Failed cases: '
    for item in check_list:
        q_pass &= check_list[item]
        if not check_list[item]:
            outcome += f'{item}, '
    return q_pass, outcome

if __name__ == '__main__':
    test_path = 'test_img/0.jpg'
    with open(test_path, "rb") as image_file:
        encode = base64.b64encode(image_file.read()).decode('utf-8')
    st = time.time()
    TIMES = 1
    for i in range(TIMES):
        print(face_attributes(encode).json())
    print(f'average took: {(time.time() - st) / TIMES} ms')