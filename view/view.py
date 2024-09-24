from fastapi import Depends
from config.config import cfg
from domain.interface import Image, Error
from core.attributes import FaceAttrbutesHandler
from core.detection import FaceDetectionHandler

class ModelView:
    tasks = {}
    def __init__(self, fd=Depends(FaceDetectionHandler), fa=Depends(FaceAttrbutesHandler)):    
        self.tasks['face_attributes'] = fa
        self.tasks['face_detection'] = fd

    def predict(self, image: Image):
        if image.cv_task not in self.tasks:
            return Error(where='cv_task', what='not found').json()
        handler = self.tasks[image.cv_task]
        return handler.predict(image)

    def get_models(self):
        task_dict = {}
        for task in cfg['task']:
            models = cfg['task'][task]['models']
            models_list = []
            for model in models:
                models_list += [x for x in model]
            task_dict[task] = models_list
        return task_dict


if __name__ == "__main__":
    v = ModelView()
    print(v.get_models())