import os
import cv2
import json
import time
import base64
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
import josephlogging.log as log
from domain.interface import result, point, Error
from domain.utils import trim_base64_header

logger = log.getLogger(__name__)
client = None

# Define PPE Labels
PPE_LABELS = {"mask", "gloves", "helmet", "goggles", "protective eyewear", "safety glasses"}

def init():
    global client
    # Load credentials from environment variable
    GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')

    if not GOOGLE_CREDENTIALS_JSON:
        raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set")

    # Parse the JSON credentials
    try:
        credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON in GOOGLE_CREDENTIALS_JSON") from e

    # Initialize the Vision API client with the credentials
    client = vision.ImageAnnotatorClient(credentials=credentials)

def detect_faces(image: str):
    if client is None:
        init()
    """Detects faces in an image."""
    image, _ = trim_base64_header(logger, image)
    content = base64.b64decode(image)
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
    return result(name='Google_Cloud_Vision', bb=bounding_boxes, imgtxt=headwear_likelihoods, lms=landmarks, ori=orientations, scores=confs, outstr={"headwear_likelihood:" : head_str})
    
def detect_ppe(imagestr: str):
    """
    Detects PPE items in an uploaded image using Google Vision API's Object Localization.
    Returns bounding boxes for detected PPE items.
    """
    if client is None:
        init()
    
    imagestr, _ = trim_base64_header(logger, imagestr)
    content = base64.b64decode(imagestr)
    image = vision.Image(content=content)

    # Decode the image to get its dimensions
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_height, image_width, _ = img.shape

    # Perform Object Localization
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    ppe_objects = []
    bbs = []
    scores = []
    detection = []  # list of detected objects
    outstr = ""

    for obj in objects:
        label = obj.name.lower()
        if label in PPE_LABELS:
            # Extract bounding polygon
            bounding_poly = obj.bounding_poly

            # Convert vertices to a list of dicts and multiply by image width and height
            vertices = [{"x": vertex.x * image_width, "y": vertex.y * image_height} for vertex in bounding_poly.normalized_vertices]
            bbs.append((point(int(bounding_poly.normalized_vertices[0].x * image_width), int(bounding_poly.normalized_vertices[0].y * image_height)), point(int(bounding_poly.normalized_vertices[2].x * image_width), int(bounding_poly.normalized_vertices[2].y * image_height))))
            scores.append(obj.score)
            detection.append(obj.name)

            ppe_objects.append({
                "name": obj.name,
                "score": obj.score,
                "bounding_poly": vertices
            })
            outstr += str(ppe_objects) + "\n"

    if response.error.message:
        logger.error(response.error.message)
        return Error(what="google vision error", where="detect")
    
    if outstr == "":
        outstr = "No PPE objects detected."

    return result(name='Vertex_Vision_PPE', bb=bbs, imgtxt=detection, scores=scores, outstr=outstr)
    
if __name__ == '__main__':
    test_path = '/Users/macbook/Code/MVP/JosephPlatform/test_img/1.jpg'
    with open(test_path, "rb") as image_file:
        encode = base64.b64encode(image_file.read()).decode('utf-8')
    st = time.time()
    TIMES = 1
    for i in range(TIMES):
        print(detect_ppe(encode).json())
    print(f'average took: {(time.time() - st) / TIMES} ms')
