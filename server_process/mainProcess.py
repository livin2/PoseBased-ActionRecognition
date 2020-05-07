import torch.multiprocessing as mp
import os
import cv2
import numpy as np
import torch
import time
import sys

from easydict import EasyDict as edict
from tqdm import tqdm
import alphapose
posepath = os.path.dirname(os.path.dirname(
        os.path.abspath(alphapose.__file__)))
sys.path.append("..")
sys.path.append(posepath)

mp.set_start_method('forkserver', force=True)
mp.set_sharing_strategy('file_system')

from loguru import logger
logger.remove()
logger.add(sys.stdout, format='<y>{level}</>|{process.name}\
(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
:<c>{line}</> -  {message}', level="DEBUG")
# logger.add(sys.stdout, format='<g>{time:MM-DD HH:mm:ss}</> |\
# {process.name}(<m>{process.id}</>) |<c>{name}</>:<c>{function}</>\
# :<c>{line}</> - {level}\n{message}', level="INFO")

from alphapose.utils.config import update_config
from server_process import ActClassifier,PoseEstimator,WebcamDetector
from utils.F import loop
from utils import Profiler
from config.apis import get_classifier_cfg
class mainProcess():
    def __init__(self,opt,imgfn): #webcam queue= 2
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose_cfg = update_config(opt.cfg)
        classifier_cfg = get_classifier_cfg(opt)
        self.opt = opt
        self.poseEstim = PoseEstimator(pose_cfg,opt)
        self.actRecg = ActClassifier(classifier_cfg,opt,imgfn)
        self.det_loader = WebcamDetector(pose_cfg,opt)
        self.__toStartEvent = mp.Event()
        self.loadedEvent = mp.Event()
        self.stopped = mp.Event()
        self.__toKillEvent = mp.Event()
        
    def __load_model(self):
        self.det_worker = self.det_loader.start(self.__toStartEvent)
        self.actRecg.start(self.__toStartEvent)
        self.poseEstim.load_model()
        self.det_loader.loadedEvent.wait()
        self.actRecg.loadedEvent.wait()
        self.loadedEvent.set()
    
    def __output(self,img,out):
        self.showimg_all(img, out)
    
    @logger.catch
    def work(self):
        try:
            logger.info('Pose Process (%s)' % os.getpid())
            if(not self.loadedEvent.is_set()):self.__load_model()

            self.__toStartEvent.wait()
            #det time| pose time| post processing
            logger.info('Starting, press Ctrl + C to terminate...')
            sys.stdout.flush()
            #loop for webcam ##todo for vedio
            im_names_desc = tqdm(loop())
            #det time| pose time| post processing
            profiler = Profiler(['dt','pt','pn'],
                ['det time: {:.4f}','pose time: {:.4f}','post processing: {:.4f}'])
            args = self.opt
            batchSize = args.posebatch
            # if args.flip:batchSize = int(batchSize / 2)
            for i in im_names_desc:
                if self.stopped.is_set():
                    self.__stop()
                    return
                if self.__toKillEvent.is_set():
                    self.__kill()
                    return
                if args.profile:profiler.start()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.read()
                    if orig_img is None:break
                    if boxes is None or boxes.nelement() == 0:
                        self.actRecg.step(None, None, None, None, None, orig_img, os.path.basename(im_name))
                        continue
                    if args.profile:profiler.step('dt')
                    # Pose Estimation
                    hm = self.poseEstim.step(inps,self.det_loader.joint_pairs)
                    if args.profile:profiler.step('pt')
                    self.actRecg.step(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))
                    if args.profile:profiler.step('pn')

                if args.profile:im_names_desc.set_description(
                        'hm:{} |'.format(hm.shape[0]) + profiler.getResStr())
                    # 'hm:{hm}|det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                    #         hm=hm.shape[0],dt=profiler.getMean('dt'), pt=profiler.getMean('pt'), pn=profiler.getMean('pn'))
        except BaseException as e:
            logger.exception(e)
            self.__stop()
            

    def start_worker(self, target):
        p = mp.Process(target=target,name='PoseProcess', args=())
        p.start()
        return p

    def load_model(self):
        self.result_worker = self.start_worker(self.work)
        return self
    
    def wait_model_loaded(self):
        self.loadedEvent.wait()
    
    def is_model_loaded(self):
        return self.loadedEvent.is_set()

    @logger.catch
    def start(self,input_source):
        assert self.loadedEvent.is_set(),'model not loaded'
        self.__toStartEvent.set()
        logger.debug('stard WebCamDetector...')
        self.det_loader.run(input_source)
        logger.debug('WebCamDetector started')
        return self

    def hangUp(self):
        self.det_loader.hangUp()

    def running(self):
        return not self.stopped.is_set()

    def stop(self): #work on MainProcess
        self.stopped.set()
        self.__toStartEvent.set()
        self.result_worker.join()

    @logger.catch
    def __stop(self): #work on PoseProcess
        self.det_loader.stop() #put a None to det_loader?
        self.actRecg.stop()
        for p in self.det_worker:
            logger.debug('waiting WebCamDetector to stop...')
            p.join(self.opt.timeout)
            if(p.is_alive()):
                logger.debug('Timeout,kill WebCamDetector...')
                p.terminate()
            else:
                logger.debug('WebCamDetector stopped')
    
    def kill(self):
        self.__toKillEvent.set()
        self.result_worker.join()
        sys.exit(-1)

    @logger.catch
    def __kill(self):
        logger.debug('kill called')
        self.det_loader.stop() #put a None to det_loader?
        self.actRecg.kill()
        for p in self.det_worker:
            p.terminate()
        sys.exit(-1)

    
    def read(self,timeout=None):
        # logger.debug('reading:{}',self.actRecg.outqueue.qsize())
        return self.actRecg.read(timeout=timeout)
    


    
        

        

