import torch.multiprocessing as mp
import os
import cv2
import numpy as np
import torch
import time
import sys
    
from loguru import logger
from easydict import EasyDict as edict
from tqdm import tqdm

from server_process import ActClassifier,PoseEstimator,WebcamDetector
from utils.F import print_finish_info,loop
from utils import Profiler

import alphapose
posepath = os.path.dirname(os.path.dirname(
        os.path.abspath(alphapose.__file__)))
sys.path.append("..")
sys.path.append(posepath)
class mainProcess():
    def __init__(self,opt,detector_cfg,pose_cfg,classifier_cfg): #webcam queue= 2
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.poseEstim = PoseEstimator(pose_cfg,opt)
        self.actRecg = ActClassifier(classifier_cfg,opt)
        # self.detector = Tracker(detector_cfg, opt)
        self.det_loader = WebcamDetector(pose_cfg,detector_cfg,opt)
        self.__toStartEvent = mp.Event()
        self.loadedEvent = mp.Event()
        self.stopped = mp.Event()
        
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
        logger.info('Pose Process (%s)' % os.getpid())
        if(not self.loadedEvent.is_set()):self.__load_model()

        self.__toStartEvent.wait()
        #det time| pose time| post processing
        
        logger.info('Starting, press Ctrl + C to terminate...')
        sys.stdout.flush()
        #loop for webcam ##todo for vedio
        im_names_desc = tqdm(loop())
        #det time| pose time| post processing
        profiler = Profiler({'dt': [],'pt': [],'pn': []})
        args = self.opt
        batchSize = args.posebatch
        # if args.flip:batchSize = int(batchSize / 2)
        for i in im_names_desc:
            if self.stopped.is_set():
                self.__stop()
                return
            profiler.start()
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
                    'hm:{hm}|det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        hm=hm.shape[0],dt=profiler.getMean('dt'), pt=profiler.getMean('pt'), pn=profiler.getMean('pn'))
                )
        print_finish_info()

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target,name='PoseProcess', args=())
        p.start()
        return p

    def load_model(self):
        self.result_worker = self.start_worker(self.work)
        return self

    def start(self,input_source):
        assert self.loadedEvent.is_set(),'model not loaded'
        self.__toStartEvent.set()
        self.det_loader.run(input_source)
        return self

    def hangUp(self):
        self.det_loader.hangUp()

    def running(self):
        return not self.stopped.is_set()

    def stop(self): #work on MainProcess
        self.stopped.set()
        self.result_worker.join()
    
    def __stop(self): #work on PoseProcess
        self.det_loader.stop() #put a None to det_loader?
        self.actRecg.stop()
        for p in self.det_worker:
            p.terminate()
    
    
        

        

