import cv2
from loguru import logger
import numpy as np
import torch
from easydict import EasyDict as edict

def argGetter(obj,key,default):
    if (hasattr(obj,key)):
        return getattr(obj,key)
    return default

# @logger.catch
def getArgs(new):
    #json.loads
    new = edict(new)
    from config.default_args import args
    for k in args.keys():
        args[k] = argGetter(new,k,args[k])
    # set
        # args.detector = new.detector
        # args.posemodel = new.posemodel
        # args.classmodel = new.classmodel
        # args.gpus = new.gpus
        # args.debug = new.debug
        # args.timeout = new.timeout
        # args.realtime = new.realtime
        # args.detbatch = new.detbatch
        # args.posebatch = new.posebatch
        # args.inqsize = new.inqsize
        # args.outqsize = new.outqsize
    args.tracking = (args.detector == 'tracker')
    from config.apis import set_posemodel_cfg
    args = set_posemodel_cfg(args)
    args.localvis = new.python_vis
    if new.usegpu:
        args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        if(args.gpus[0]==-1):logger.info('no cuda device available')
        args.detbatch = args.detbatch * len(args.gpus)
        args.posebatch = args.posebatch * len(args.gpus)
    else: args.device = "cpu"
    if args.realtime:
        args.inqsize=2
        args.outqsize=2
    return args

@logger.catch
def getInputInfo(data):
    data = edict(data)
    res = edict()
    res.inp = 0
    if data.input_type == 'rtmp':
        res.inp = data.input_path
    elif data.input_type != 'camera':
        raise NotImplementedError
    stream = cv2.VideoCapture(res.inp)
    assert stream.isOpened()
    res.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    res.fps = stream.get(cv2.CAP_PROP_FPS)
    res.w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    res.h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream.release()
    return res

def npImgToEncodeBytes(imgframe:np.ndarray):
    # assert isinstance(imgframe, np.ndarray),'input is not np.ndarray'
    # enconde_data = cv2.imencode('.png', imgframe)[1]
    h = imgframe.shape[0]
    w = imgframe.shape[1]
    k = 0.5
    imgframe = cv2.resize(imgframe,(int(k*w),int(k*h)))
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    enconde_data = cv2.imencode('.jpg', imgframe,img_param)[1]
    return enconde_data.tostring()

import random
@logger.catch
def packResultMock(imgencoded,tagI2W:list):
    pack = {}
    pack['img'] = imgencoded
    thslen = 3
    out = np.random.rand(thslen,len(tagI2W))
    pack['datas'] = [{
        'tag':tagI2W [random.randint(0,len(tagI2W)-1)],
        'act_scores':{tagI2W[j]:out[i,j] for j in range(len(tagI2W))}
    } for i in range(thslen)]
    # print(pack)
    return pack
    
@logger.catch
def packResult(imgencoded,out,res,tagI2W:list):
    pack = edict()
    pack.img = imgencoded
    if out is None:return pack
    pack.datas = [{
        'tag':tagI2W [np.argmax(out[i])],
        'act_scores':{tagI2W[j]:float(out[i][j]) for j in range(len(tagI2W))}
    } for i in range(len(out))]
    return pack
    # 'box': 
    # 'keypoints': 
    # 'kp_score': 
    # 'proposal_score': 
    # 'idx':

def drawResultToImg_MutiPerson(img:np.ndarray,out:list,tagI2W:list):
    # logger.debug('visualizing with opencv...')
    if img is None:return img
    if out is not None:
        h = 50
        for i in range(len(out)):
            tag = tagI2W [np.argmax(out[i])]
            img = cv2.putText(img, tag, (20,h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
            h+=50
    # img = np.array(img, dtype=np.uint8)[:, :, ::-1]
    return img

def drawTagToImg(self,img:np.ndarray,tag:str):
    img = cv2.putText(img, tag, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # img = np.array(img, dtype=np.uint8)[:, :, ::-1]
    return img

def drawAllOutPutToImg_SinglePerson(self,img:np.ndarray,out:torch.Tensor):
    if img is None:return img
    if out is not None:
        h = 20
        for i in range(len(self.cfg.tagI2W)):
            text = '{}  {}'.format(self.cfg.tagI2W[i],out[-1,i]) #dnn
            img = cv2.putText(img, text, (20,h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            h+=20
        tag = self.cfg.tagI2W [np.argmax(out[i])]
        img = cv2.putText(img, tag, (20,h+200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)
    else:
        # img = cv2.putText(img, "None", (20,300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)
        pass
    # img = np.array(img, dtype=np.uint8)[:, :, ::-1]
    return img

    
    