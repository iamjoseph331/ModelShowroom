import time
import base64
from google.cloud import vision
import josephlogging.log as log
from domain.interface import result, point, Error
from domain.utils import trim_base64_header

logger = log.getLogger(__name__)

def detect_faces(image: str):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()
    image, _ = trim_base64_header(logger, image)
    content = base64.b64decode(image)
    print(image[:20])
    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations
    if response.error.message:
        logger.error(response.error.message)
        return Error(what="google vision error", where="detect")
    
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )
    print(f"Faces: {len(faces)}")
    bounding_boxes = []
    confs = []
    landmarks = []
    orientations = []
    headwear_likelihoods = []
    for face in faces:
        print(f"headwear_likelihood: {likelihood_name[face.headwear_likelihood]}")
        print(f"detection_confidence: {face.detection_confidence}")

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in face.bounding_poly.vertices
        ]

        print("face bounds: {}".format(",".join(vertices)))
        bounding_boxes.append((point(face.bounding_poly.vertices[0].x, face.bounding_poly.vertices[0].y), point(face.bounding_poly.vertices[2].x, face.bounding_poly.vertices[2].y)))
        confs.append(face.detection_confidence)
        landmarks_dict = {}
        for landmark in face.landmarks:
            landmarks_dict[landmark.type_] = point(landmark.position.x, landmark.position.y)
        # dict into list for now
        landmarks = [point(landmark.position.x, landmark.position.y) for landmark in face.landmarks]
        orientations.append(face.roll_angle)
        headwear_likelihoods.append(likelihood_name[face.headwear_likelihood])

    conf_str = [f"confidence: {face.detection_confidence}" for face in faces]
    head_str = ",".join([f"{likelihood_name[face.headwear_likelihood]}" for face in faces])
    return result(name='Google', bb=bounding_boxes, imgtxt=conf_str, lms=landmarks, ori=orientations, scores=confs, outstr={"headwear_likelihood:" : head_str})
    

if __name__ == '__main__':
    test_path = '/Users/macbook/Code/MVP/JosephPlatform/test_img/1.jpg'
    with open(test_path, "rb") as image_file:
        encode = base64.b64encode(image_file.read()).decode('utf-8')
    st = time.time()
    TIMES = 1
    for i in range(TIMES):
        print(detect_faces(test_path).json())
    print(f'average took: {(time.time() - st) / TIMES} ms')