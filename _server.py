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
app.config['SECRET_KEY'] = 'secret!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new')
def hello():
    print(os.getpid())
    return 'hello'

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
        print(type(data))
        args = getArgs(data)
        logger.debug('args:\n{}',args)
        context.mainp = MainProcess(args,draw)
        logger.info('inited,start loading model...')
        context.mainp.load_model()
        context.mainp.loadedEvent.wait()
        return 'ok'
    except BaseException as e:
        logger.exception(e)
        return 'fail'

@app.route('/start')
def start():
    try:
        input_source = 'rtmp://58.200.131.2:1935/livetv/gdtv'
        # input_source='rtmp://58.200.131.2:1935/livetv/hunantv'
        logger.info('input:{}',input_source)
        logger.info('starting... ')
        context.mainp.start(input_source)
        return 'ok'
    except BaseException as e:
        logger.exception(e)
        return 'fail'

@app.route('/stop',methods=['POST'])
def stop():
    try:
        data = json.loads(request.get_data())
        print(data)
        context.mainp.stop()
        # p.join()
        return {'status':'ok'}
    except BaseException as e:
        logger.exception(e)
        return {'status':'fail'}
    finally:
        pass
        # sys.exit()
    
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
