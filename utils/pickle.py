import cv2
from loguru import logger
import numpy as np
import torch

def npImgToEncodeBytes(imgframe:np.ndarray):
    # assert isinstance(imgframe, np.ndarray),'input is not np.ndarray'
    enconde_data = cv2.imencode('.png', imgframe)[1]
    return enconde_data.tostring()

def drawResultToImg_MutiPerson(img:np.ndarray,out:list,tagI2W:list):
    # logger.debug('visualizing with opencv...')
    if img is None:return
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
    if img is None:return
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

    
    