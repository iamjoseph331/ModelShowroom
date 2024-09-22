import uvicorn

from core.attributes import FaceAttrbutesHandler

from view.view import ModelView
from config.config import cfg, public_host

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from domain.interface import Image

import josephlogging.log as log

app = FastAPI()
logger = log.getLogger(__name__)

def init():
    fa_handle = FaceAttrbutesHandler()
    return ModelView(fa=fa_handle)
     
origins = [
    'http://localhost',
    public_host
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

view = init()
@app.get('/api/init/{model_name}')
async def init_model(model_name):
    return {'initialized': model_name}

@app.post('/api/predict')
async def predict(image: Image):
    logger.info(f'predict to {image.mdl_name}')
    return view.predict(image)
    
@app.get('/api/getmodels')
async def get_models():
    logger.info(f'get available models')
    return view.get_models()

if __name__ == '__main__':
    print('Starting server...')
    uvicorn.run('main:app', host=cfg['host'], port=cfg['port'], log_level='info', workers=4)