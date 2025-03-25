import os
import time
from ultralytics import YOLO
from PIL import Image
from domain.interface import result, Error, point
from infra.common.utils import data_uri_to_cv2_img
from infra.common.datacollection import upload_frame
import josephlogging.log as log

DEBUG = False
DATACOLLECTION = False

logger = log.getLogger(__name__)
providers =  ['CPUExecutionProvider']
analyzer = None

class RidePicAnalyzer:
    def __init__(self, model_path:str, device:str):
        self.model = YOLO(model_path)
        self.device = device
        self.fields = {
            "Model": model_path,
            "HasLuupVehicle": "bool",
            "LuupVehicleCount": "int",
            "LuupVehicleType": "[str]",
            "LuupVehicleBBox": "[(int(x1), int(y1), int(x2), int(y2))]",
            "HasLuupPort": "bool",
            "LuupPortCount": "int",
            "LuupPortType": "[str]",
            "LuupPortBBox": "[(int(x1), int(y1), int(x2), int(y2))]",
            "IsOutOfPort": "bool",
            "DetectedPortID": "str",
            "DetectedVehicleNumber": "[str]",
            "DetectedInPortOtherVehicleCount": "int",
            "DetectedInPortOtherVehicleType": "[str]",
            "DetectedInPortOtherVehicleBBox": "[(int(x1), int(y1), int(x2), int(y2))]",
            "DetectedInPortOtherVehicleNumber": "[str]",
            #align with v1
            "result": "str",
            "probability": "float"
        }

    def analyze_one(self, image: Image, threshold: float):
        results = self.model(image, device=self.device ,verbose=False)
        analyze_results = self.fields.copy()
        # initialize all fields to None
        for key in analyze_results:
            analyze_results[key] = None
        # check if there is a Luup vehicle
        luup_results, luup_type, conf_luup = self.bbox_of_Luup(results, threshold)
        if len(luup_results) > 0:
            analyze_results["HasLuupVehicle"] = True
            analyze_results["LuupVehicleCount"] = len(luup_results)
            analyze_results["LuupVehicleType"] = luup_type
            analyze_results["LuupVehicleBBox"] = luup_results
        # check if there is a Luup port
        port_results, port_type, conf_port = self.port_exist(results, threshold)
        if len(port_results) > 0:
            analyze_results["HasLuupPort"] = True
            analyze_results["LuupPortCount"] = len(port_results)
            analyze_results["LuupPortType"] = port_type
            analyze_results["LuupPortBBox"] = port_results
        # check if the vehicle is out of port
        analyze_results["IsOutOfPort"] = self.classify_out_of_port(analyze_results)
        analyze_results["result"] = "normal" if analyze_results["HasLuupVehicle"] else "abnormal"
        # probablity of luup vehicle existing
        analyze_results["probability"] = float(conf_luup[0]) if analyze_results["HasLuupVehicle"] else float(1.0)
        return analyze_results

    def port_exist(self, results, threshold: float):
        bboxes = []
        port_types = []
        conf = []
        if len(results) == 0:
            return bboxes, port_types, conf
        for box in results[0].boxes:
            bx = box.xyxy.to('cpu').numpy()
            cf = box.conf.to('cpu').numpy()
            for i, b in enumerate(bx):
                x1, y1, x2, y2 = b
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = results[0].names[int(box.cls)]  # Get Label name
                if label == "LuupSign" and cf[i] > threshold:
                    bboxes.append((x1, y1, x2, y2))
                    port_types.append("LuupSign")
                    conf.append(cf[i])
                if label == "LuupPortLine" and cf[i] > threshold:
                    bboxes.append((x1, y1, x2, y2))
                    port_types.append("LuupPortLine")
                    conf.append(cf[i])
        return bboxes, port_types, conf

    def bbox_of_Luup(self, results, threshold: float):
        bbox = []
        luup_type = []
        conf = []
        for box in results[0].boxes:
            bx = box.xyxy.to('cpu').numpy()
            cf = box.conf.to('cpu').numpy()
            for i, b in enumerate(bx):
                x1, y1, x2, y2 = b
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = results[0].names[int(box.cls)]  # Get Label name
                if label == "LuupBicycle" and cf[i] > threshold:
                    bbox.append((x1, y1, x2, y2))
                    luup_type.append("LuupBicycle")
                    conf.append(cf[i])
                elif label == "LuupKB" and cf[i] > threshold:
                    bbox.append((x1, y1, x2, y2))
                    luup_type.append("LuupKB")
                    conf.append(cf[i])
        return bbox, luup_type, conf

    def classify_out_of_port(self, analyze_results: dict):
        if not analyze_results["HasLuupVehicle"]:
            return True
        # if port type has LuupSign in it, then it is not out of port
        if analyze_results["HasLuupPort"] and "LuupSign" in analyze_results["LuupPortType"]:
            return False
        # if there are multiple luup vehicles, then it is not out of port
        if analyze_results["LuupVehicleCount"] > 1:
            return False
        # if there is no port and luup count = 1 then it is possibly out of port
        if not analyze_results["HasLuupPort"] and analyze_results["LuupVehicleCount"] == 1:
            return True
        # if the only luup and LuupPortLine have no overlap, possibly out of port
        luupbb = analyze_results["LuupVehicleBBox"][0]
        for portbb in analyze_results["LuupPortBBox"]:
            if not self.bbox_overlap(luupbb, portbb):
                return True
        return False

    def bbox_overlap(self, box1: list[int], box2: list[int]):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        # in case the box is not in the correct order
        if x1 > x2:
            x1, x2 = x2, x1
        if x3 > x4:
            x3, x4 = x4, x3
        if y1 > y2:
            y1, y2 = y2, y1
        if y3 > y4:
            y3, y4 = y4, y3
        if (x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3):
            return True
        return False

    def analyze_batch(self, images: list[Image.Image], threshold: float, names: list[str] = None):
        results = []
        for i in range(len(images)):
            image = images[i]
            resjson = self.analyze_one(image, threshold)
            if names:
                resjson["filename"] = names[i]
            results.append(resjson)
        return results
    
def init(modelname:str, device:str = "cpu"):
    try:
        global analyzer
        analyzer = RidePicAnalyzer(modelname, device)
        logger.info(f'Model initialized: {modelname}')
        return True
    except Exception as e:
        logger.error(f'Error initializing model: {e}')
        return False

def predict(base64_image: str, threshold: float): 
    global analyzer
    if not analyzer:
        logger.info('Initializing model')
        analyzer = RidePicAnalyzer("infra/model/LuupDetectV2.1.0.3.pt", "cpu")
    rgb_image = data_uri_to_cv2_img(logger, base64_image)
    rgb_image = rgb_image[:, :, ::-1]
    pilimg = Image.fromarray(rgb_image)
    res_bbs = []
    ms = []
    confs = []
    results = analyzer.model(pilimg, device=analyzer.device, verbose=False)
    for box in results[0].boxes:
        bx = box.xyxy.to('cpu').numpy()
        cf = box.conf.to('cpu').numpy()
        for i, b in enumerate(bx):
            x1, y1, x2, y2 = b
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = results[0].names[int(box.cls)]  # Get Label name
            if cf[i] > threshold:
                res_bbs.append((x1, y1, x2, y2))
                ms.append(label)
                confs.append(cf[i])
    output = {'detection': len(res_bbs), 'items':ms, 'confidence':confs}
    if DEBUG: pilimg.save("test.jpg")   
    if DATACOLLECTION:  
        try:
            timestamp = str(int(time.time()))
            upload_frame(timestamp, base64_image, 'vehicle', analyzer.model, {}, float(confs[0]), '0', 'jpg')
        except Exception as e:
            logger.error(f'Data collection error: {e}')
    return result(name='vehicle', bb=res_bbs ,imgtxt=ms, outstr=output)

if __name__ == "__main__":
    model = RidePicAnalyzer("infra/model/LuupDetectV2.1.0.3.pt", "cpu")
    test_folder = "/Users/macbook/Code/datasets/LuupSegement/放置疑惑0127_0128"
    images_and_names = [(Image.open(f"{test_folder}/{img}"), img) for img in os.listdir(test_folder) if img.endswith(".jpg")]
    test_images = [img[0] for img in images_and_names[:3]]
    test_names = [img[1] for img in images_and_names[:3]]
    results = model.analyze_batch(test_images, 0.5, test_names)
    print(results)