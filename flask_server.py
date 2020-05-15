from flask import Flask,render_template,request
from flask_cors import CORS
from flask_socketio import SocketIO,emit
import json
import time
import sys
sys.path.append("..")

from easydict import EasyDict
context = EasyDict()

from loguru import logger
logger.remove()
logger.add(sys.stdout, format='<y>{level}</>|{process.name}\
(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
:<c>{line}</> -  {message}', level="DEBUG")

app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*')
CORS(app, resources=r'/*')
app.config['SECRET_KEY'] = 'secret'

@app.route('/')
def index():
    return render_template('index.html')

def socketio_wait(flagfn,timeout=0.01):
    while flagfn():
        socketio.sleep(timeout)

from utils.pickle import getArgs
from server_process import MainProcess
from utils.pickle import drawResultToImg_MutiPerson as draw
@app.route('/init',methods=['POST'])
def init():
    try:
        data = json.loads(request.get_data())
        args = getArgs(data)
        logger.debug('args:\n{}',args)
        context.mainp = MainProcess(args,draw)
        logger.info('inited,start loading model...')
        context.mainp.load_model()
        context.mainp.loadedEvent.wait()
        logger.info('model loaded')
        return {'status':'ok','appconfig':str(args)},201
    except BaseException as e:
        logger.exception(e)
        return {'status':'fail'},400

import numpy as np
from queue import Empty as EmptyException
from utils.pickle import npImgToEncodeBytes
from utils.pickle import packResult
from utils.F import loop
@logger.catch
def commitImage(context):
    logger.debug('try read...')
    for i in loop():
        try:
            if(context.stope.is_set()):return 
            frame,out,result = context.mainp.read(0.01) 
            tagI2W = context.mainp.classifier_cfg.tagI2W
            if isinstance(frame, np.ndarray):
                enconde_img = npImgToEncodeBytes(frame)
                toSend = packResult(enconde_img,out,result,tagI2W)
                socketio.emit('image_frame',toSend,namespace='/vis')
                socketio.sleep(0.01)
        except EmptyException as e:
            socketio.sleep(3)
            logger.debug('{} item waitting',context.mainp.count())
            continue
        except BaseException as e:
            logger.error('error: ',e)
            logger.exception(e)

from utils.pickle import getInputInfo
import multiprocessing as mp
@logger.catch
@app.route('/start',methods=['POST'])
def start():
    try:
        data = json.loads(request.get_data())
        vinfo = getInputInfo(data)
        logger.info('input:{}',vinfo)
        context.vinfo = vinfo
        logger.info('starting... ')
        input_source = vinfo.inp
        context.mainp.start(input_source)
        context.stope = mp.Event()
        if(not context.mainp.opt.localvis):
            socketio.start_background_task(commitImage,context)
        return {'status':'ok','input_info':str(vinfo)}
    except BaseException as e:
        logger.exception(e)
        return {'status':'fail'},400

@app.route('/stop',methods=['POST'])
def stop():
    try:
        data = json.loads(request.get_data())
        logger.debug(data)
        context.mainp.stop()
        context.stope.set()
        return {'status':'ok'}
    except BaseException as e:
        logger.exception(e)
        return {'status':'fail'},500

@app.route('/pause',methods=['POST'])
def pause():
    try:
        context.mainp.hangUp()
        return {'status':'ok','app_info':context.mainp.opt}
    except BaseException as e:
        logger.exception(e)
        return {'status':'fail'},500

import os
if __name__ == "__main__":
    try:
        logger.warning('run __main__')
        print('Process (%s)' % os.getpid())
        print('name:%s\n'%__name__)
        print(app)
        socketio.run(app,debug=True)
    except BaseException as e:
        logger.exception(e)
