import cv2
from loguru import logger
import numpy as np
import torch
from easydict import EasyDict as edict

def argGetter(obj,key,default):
    if (hasattr(obj,key)):
        return getattr(obj,key)
    return default

def getArgs(new):
    #json.loads
    new = edict(new)
    from config.default_args import args
    # for k in args.keys():
    #     args[k] = argGetter(new,k,args[k])
    args.detector = new.detector
    args.tracking = (args.detector == 'tracker')
    args.posemodel = new.posemodel
    from config.apis import set_posemodel_cfg
    args = set_posemodel_cfg(args)
    args.classmodel = new.classmodel
    args.vislocal = new.python_vis
    args.gpus = new.gpus
    if new.usegpu:
        args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        if(args.gpus[0]==-1):logger.info('no cuda device available')
        args.detbatch = args.detbatch * len(args.gpus)
        args.posebatch = args.posebatch * len(args.gpus)
    else: args.device = "cpu"
    args.debug = new.debug
    args.timeout = new.timeout
    args.realtime = new.realtime
    args.detbatch = new.detbatch
    args.posebatch = new.posebatch
    args.inqsize = new.inqsize
    args.outqsize = new.outqsize
    if args.realtime:
        args.inqsize=2
        args.outqsize=2
    return args

def getInput(data):
    pass # if args.input = 'cam':args.input=0 ###

def npImgToEncodeBytes(imgframe:np.ndarray):
    # assert isinstance(imgframe, np.ndarray),'input is not np.ndarray'
    enconde_data = cv2.imencode('.png', imgframe)[1]
    return enconde_data.tostring()

def drawResultToImg_MutiPerson(img:np.ndarray,out:list,tagI2W:list):
    # logger.debug('visualizing with opencv...')
    if img is None:return img
    if out is not None:
        h = 50
        for i in range(len(out)):
            tag = tagI2W [out[i].argmax(1)]
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
        tag = self.cfg.tagI2W [out.argmax(1)]
        img = cv2.putText(img, tag, (20,h+200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)
    else:
        # img = cv2.putText(img, "None", (20,300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)
        pass
    # img = np.array(img, dtype=np.uint8)[:, :, ::-1]
    return img

    
    