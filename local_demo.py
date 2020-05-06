#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

import pysnooper
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import json

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader

from utils.F import check_input
from config.default_args import args
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


from detector.tracker_api import Tracker
from config.default_tracker_cfg import cfg as detector_cfg
from config.default_classifier_cfg import cfg as classifier_cfg
from server_process import MainProcess
logger.remove()
# logger.add(sys.stdout, format='<g>{time:MM-DD HH:mm:ss}</> |\
# {process.name}(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
# :<c>{line}</> - {level}\n{message}', level="INFO")

logger.add(sys.stdout, format='<y>{level}</>|\
{process.name}(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
:<c>{line}</> -  {message}', level="DEBUG")

import cv2
def showimg_muti(img,out,tagI2W):
    # logger.debug('vis_muti')
    if img is None:return
    if out is not None:
        h = 50
        for i in range(len(out)):
            tag = tagI2W [out[i].argmax(1)]
            img = cv2.putText(img, tag, (20,h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
            h+=50
    else:
        # img = cv2.putText(img, "None", (20,300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)
        pass
    # img = np.array(img, dtype=np.uint8)[:, :, ::-1]
    cv2.imshow("AlphaPose Demo", img)
    k = cv2.waitKey(100) 

if __name__ == "__main__":
    try:
        pose_cfg = update_config(args.cfg)
        mode, input_source = check_input(args)
        # input_source='rtmp://58.200.131.2:1935/livetv/hunantv'
        input_source='rtmp://58.200.131.2:1935/livetv/gdtv'
        print('mode:',mode)
        print('input_source:',input_source)

        mainp = MainProcess(args,detector_cfg,pose_cfg,classifier_cfg)
        logger.info('inited')
        mainp.load_model()
        mainp.loadedEvent.wait()
        logger.info('loaded')
        mainp.start(input_source)
        logger.info('started')

        # logger.debug('try...')
        # time.sleep(5)
        logger.debug('read...')
        for i in range(300):
            (img,out) = mainp.read()
            # logger.debug('readed{}',i)
            if img is None:
                break
            showimg_muti(img,out,classifier_cfg.tagI2W)
        logger.debug('end')
        # time.sleep(30)
        # logger.info('hangUp')
        # mainp.hangUp()

        # time.sleep(20)
        # logger.info('restart')
        # mainp.start('rtmp://58.200.131.2:1935/livetv/gdtv')

        # time.sleep(30)
        mainp.stop()
    except KeyboardInterrupt:
        mainp.stop()
        logger.info('stop')
    except BaseException as e:
        logger.exception(e)
    finally:
        mainp.stop()


