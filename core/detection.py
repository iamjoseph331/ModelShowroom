from domain.interface import Image, Error
from config.config import cfg
from josephlogging import log

class DetectionHandler:
    model_list = {}
    def __init__(self):
        self.logger = log.getLogger(__name__)
        models = cfg['task']['detection']['models']
        for model in models:
            for key in model:
                model_name = key
                func = self.load_attr(model[key])
                self.model_list[model_name] = func

    def load_attr(self, path):
        funcpth = path.split('.')
        mod = '.'.join(funcpth[:-1])
        m = __import__(mod)
        for comp in funcpth[1:]:
            m = getattr(m, comp)
        return m

    def details(self):
        self.logger.info(f'FD models: {[m for m in self.model_list]}')

    def predict(self, image: Image, threshold:float=0.5):
        model_name = image.mdl_name
        if model_name not in self.model_list:
            return Error(where='mdl_name', what='not found').json()
        res = self.model_list[model_name](image.image, threshold)
        return res

if __name__ == "__main__":
    hand = DetectionHandler()
    hand.details()