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

from config.default_args import args
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

from server_process import MainProcess
logger.remove()
# logger.add(sys.stdout, format='<g>{time:MM-DD HH:mm:ss}</> |\
# {process.name}(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
# :<c>{line}</> - {level}\n{message}', level="INFO")

logger.add(sys.stdout, format='<y>{level}</>|\
{process.name}(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
:<c>{line}</> -  {message}', level="DEBUG")

import cv2
from config.apis import get_classifier_cfg
from utils.pickle import drawResultToImg_MutiPerson as drawImg
if __name__ == "__main__":
    try:
        # input_source='rtmp://58.200.131.2:1935/livetv/hunantv'
        # input_source='rtmp://58.200.131.2:1935/livetv/gdtv'
        input_source=0
        # mainp = MainProcess(args,drawImg)
        # stream = cv2.VideoCapture(input_source)
        # assert stream.isOpened(), 'Cannot capture source'
        # logger.info('input:{}',input_source)
        # for i in range(60):
        #     (grabbed, frame) = stream.read()
        #     if not grabbed: 
        #         logger.debug('no grab {}',i)
         
        # time.sleep(5)
        # stream.release()
        

        # # args.classmodel="fclstm9"
        args.localvis = True
        print('input_source:',input_source)
        classifier_cfg = get_classifier_cfg(args)
        mainp = MainProcess(args,drawImg)
        logger.info('inited')
        mainp.load_model()
        # time.sleep(10)
        mainp.loadedEvent.wait()
        logger.info('loaded')
        mainp.start(input_source)
        logger.info('started')

        # from utils.pickle import npImgToEncodeBytes
        # from utils.pickle import packResult
        # from utils.pickle import packResultMock
        # logger.debug('read...')
        # for i in range(60):
        #     (img,out,result) = mainp.read()
        #     if out is not None:
        #         # pack = packResult('imgimg',out,result,classifier_cfg.tagI2W)
        #         pack = packResultMock('imgimg',classifier_cfg.tagI2W)
        #         logger.debug('\n{} pack:\n{}',i,pack)
        #     if img is None:
        #         break
        #     showimg_muti(img,out,classifier_cfg.tagI2W)

        # time.sleep(30)
        # logger.info('hangUp')
        # mainp.hangUp()

        # time.sleep(20)
        # logger.info('restart')
        # mainp.start('rtmp://58.200.131.2:1935/livetv/gdtv')

        time.sleep(300)
        # logger.debug('end')
        # mainp.stop()
    except KeyboardInterrupt:
        mainp.stop()
        logger.info('stop')
    except BaseException as e:
        logger.exception(e)
    finally:
        mainp.stop()


