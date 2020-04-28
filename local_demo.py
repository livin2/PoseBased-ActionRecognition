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
logger.add(sys.stdout, format='<g>{time:MM-DD HH:mm:ss}</> |\
{process.name}(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
:<c>{line}</> - {level}\n{message}', level="INFO")

if __name__ == "__main__":
    try:
        pose_cfg = update_config(args.cfg)
        mode, input_source = check_input(args)
        input_source='rtmp://58.200.131.2:1935/livetv/hunantv'
        print('mode:',mode)
        print('input_source:',input_source)

        mainp = MainProcess(args,detector_cfg,pose_cfg,classifier_cfg)
        logger.info('inited')
        mainp.load_model()
        mainp.loadedEvent.wait()
        logger.info('loaded')
        mainp.start(0)
        logger.info('started')

        # time.sleep(30)
        # logger.info('hangUp')
        # mainp.hangUp()

        # time.sleep(20)
        # logger.info('restart')
        # mainp.start('rtmp://58.200.131.2:1935/livetv/gdtv')

        time.sleep(30)
        mainp.stop()
    except KeyboardInterrupt:
        mainp.stop()
        logger.info('stop')
    finally:
        mainp.stop()


