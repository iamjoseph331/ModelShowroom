import time
import base64
from google.cloud import vision
import josephlogging.log as log
from domain.interface import result, point, Error

'''
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
'''

def detect_faces(image: str):
    """Detects faces in an image."""
    return result(name='Google', bb=[], imgtxt='text', outstr='outstr')
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )
    print("Faces:")

    for face in faces:
        print(f"anger: {likelihood_name[face.anger_likelihood]}")
        print(f"joy: {likelihood_name[face.joy_likelihood]}")
        print(f"surprise: {likelihood_name[face.surprise_likelihood]}")

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in face.bounding_poly.vertices
        ]

        print("face bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

if __name__ == '__main__':
    test_path = '/Users/macbook/Code/MVP/JosephPlatform/test_img/0.jpg'
    with open(test_path, "rb") as image_file:
        encode = base64.b64encode(image_file.read()).decode('utf-8')
    st = time.time()
    TIMES = 1
    for i in range(TIMES):
        print(detect_faces(test_path).json())
    print(f'average took: {(time.time() - st) / TIMES} ms')