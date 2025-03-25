from fastapi import Depends
from config.config import cfg
from domain.interface import Image, Error
from core.attributes import AttrbutesHandler
from core.detection import DetectionHandler

class ModelView:
    tasks = {}
    def __init__(self, fd=Depends(DetectionHandler), fa=Depends(AttrbutesHandler)):    
        self.tasks['attributes'] = fa
        self.tasks['detection'] = fd

    def predict(self, image: Image, threshold:float=0.5):
        if image.cv_task not in self.tasks:
            return Error(where='cv_task', what='not found').json()
        handler = self.tasks[image.cv_task]
        return handler.predict(image, threshold)

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