import torch
import os
import torch.multiprocessing as mp
from server_process import ActClassifier,WebcamDetector
from config.default_args import args
from loguru import logger
if not args.sp:
    # torch.multiprocessing.set_start_method('fork', force=True)
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

from alphapose.utils.config import update_config
from config.default_classifier_cfg import cfg as classifier_cfg
from config.default_tracker_cfg import cfg as detector_cfg
from multiprocessing.synchronize import Event as EventType
from utils.F import check_input,loop
from actRec.models import get_model
from tqdm import tqdm
import time
import numpy as np
import cv2
# actRecg = ActClassifier(classifier_cfg,args)
def ModelInMainProcess():
    # actRecg = ActClassifier(classifier_cfg,args)
    actRecg.load_model()
    actRecg.loadedEvent.wait()
    print('loaded')
    
def ModelInSubProcess():
    # actRecg = ActClassifier(classifier_cfg,args)
    startEvent = mp.Event()
    actRecg.start(startEvent)
    actRecg.loadedEvent.wait()
    print('loaded')

def loadInMainUseInSub2():
    actRecg = ActClassifier(classifier_cfg,args)
    actRecg.load_model()
    actRecg.loadedEvent.wait()
    startEvent = mp.Event()
    actRecg.start(startEvent)
    print('loaded')

def loadInMainUseInSub():
    md = load_model(classifier_cfg)
    # actRecg.loadedEvent.wait()
    print('loaded')
    startEvent = mp.Event()
    p  = test_start(startEvent,md)

def test_start(startEvent=None,model=None):
    startEvent = startEvent
    p = mp.Process(target=test_work, args=(startEvent,model,))
    p.start()
    p.join()
    return p

def load_model(classifier_cfg):
    model = get_model(classifier_cfg.MODEL,classifier_cfg.tagI2W)
    ckpt = torch.load(classifier_cfg.CHEKPT,map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(args.device)
    model.eval()
    logger.info(model)
    # model.share_memory()
    return model

def test_work(startEvent,model):
    logger.info('Sub Process (%s)' % os.getpid())
    ###
    logger.info(model)
    if(isinstance(startEvent,EventType)):startEvent.wait()
    ###    

def showimg(img,tag=None):
    # height, width = img.shape[:2]
    # img = cv2.putText(img, tag, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    img = np.array(img, dtype=np.uint8)[:, :, ::-1]
    cv2.imshow("AlphaPose Demo", img)
    k = cv2.waitKey(100) 
import sys
logger.remove()
logger.add(sys.stdout, format='<g>{time:MM-DD HH:mm:ss}</> |\
{process.name}(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
:<c>{line}</> - {level}\n{message}', level="INFO")

import alphapose
if __name__ == "__main__":
    logger.info('Main Process (%s)' % os.getpid())
    logger.info('Device:(%s)'%args.device)
    pose_cfg = update_config(args.cfg)
    mode, input_source = check_input(args)
    # input_source='rtmp://58.200.131.2:1935/livetv/gdtv'
    logger.info('mode:{}',mode)
    logger.info('input_source:{}',input_source)
    try:
        det_loader = WebcamDetector(pose_cfg,detector_cfg,args)
        toStartEvent = mp.Event()
        det_loader.start(toStartEvent)
        det_loader.loadedEvent.wait()
        toStartEvent.set()
        # im_names_desc = tqdm(loop())
        im_names_desc = tqdm(range(30))
        det_loader.run(input_source)
        for i in im_names_desc:
            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
            showimg(orig_img)
        det_loader.hangUp()
        cv2.destroyAllWindows()

        time.sleep(5)
        im_names_desc = tqdm(range(30))
        logger.info('run again')
        det_loader.run(input_source)
        for i in im_names_desc:
            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
            showimg(orig_img)
        det_loader.stop()
        cv2.destroyAllWindows()

        
    except KeyboardInterrupt:
        # actRecg.stop()
        pass
    
    