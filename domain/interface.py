# -*- coding: utf-8 -*-\
from typing import List, Dict
from pydantic import BaseModel

class Image(BaseModel):
    mdl_name: str
    cv_task: str
    image: str
    metadata: str = ''

class point:
    def __init__(self, x=0, y=0) -> None:
        self.x = int(x)
        self.y = int(y)
    def to_str(self) -> str:
        return f'({self.x},{self.y})'

class orien:
    def __init__(self, y=0, p=0, r=0) -> None:
        self.y = float(y)
        self.p = float(p)
        self.r = float(r) 
    def to_str(self) -> str:
        return f'({self.y},{self.p},{self.r})'
    def get_max(self):
        return max(self.y, self.p, self.r)

class result:
    model_name = ''
    bounding_box = [(point(0,0), point(1,1))]
    orient = []
    landmarks = []
    scores = []
    image_text = []
    output_string = ''

    def __init__(self, name='', bb=[(point(0,0), point(1,1))], ori=[], lms=[], scores=[], imgtxt=[], outstr=''):
        self.model_name = name
        self.bounding_box = bb
        self.orient = ori
        self.landmarks = lms
        self.scores = scores
        self.image_text = imgtxt
        self.output_string = outstr

    def json(self) -> dict:
        d = {}
        d['model_name'] = self.model_name
        d['boundingBox'] = self.bounding_box
        d['orientation'] = self.orient
        d['landmarks'] = self.landmarks
        d['scores'] = self.scores
        d['image_text'] = self.image_text
        d['output_string'] = self.output_string
        return d

class Error:
    def __init__(self, where, what) -> None:
        self.where = where
        self.what = what
        self.status_code = 500
    def json(self) -> dict:
        return {'error': {'where': self.where, 'what': self.what}}