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
    def __init__(self,opt,imgfn=None): 
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_cfg = update_config(opt.cfg)
        self.classifier_cfg = get_classifier_cfg(opt)
        self.opt = opt
        self.poseEstim = PoseEstimator(self.pose_cfg,opt)
        self.actRecg = ActClassifier(self.classifier_cfg,opt,imgfn)
        self.det_loader = WebcamDetector(self.pose_cfg,opt)
        self.__toStartEvent = mp.Event()
        self.loadedEvent = mp.Event()
        self.stopped = mp.Event()
        self.__toKillEvent = mp.Event()
        m = mp.Manager()
        self.mp_dict = m.dict()
        self.mp_dict['img'] = None
        self.mp_dict['fresh'] = False
        
    def __load_model(self):
        self.det_worker = self.det_loader.start(self.__toStartEvent)
        self.actRecg.start(self.__toStartEvent)
        self.poseEstim.load_model()
        self.det_loader.loadedEvent.wait()
        self.actRecg.loadedEvent.wait()
        self.loadedEvent.set()
    
    @logger.catch
    def work(self):
        try:
            logger.info('Pose Process (%s)' % os.getpid())
            if(not self.loadedEvent.is_set()):self.__load_model() #加载模型
            self.__toStartEvent.wait() #等待启动事件
            logger.info('Starting, press Ctrl + C to terminate...') #本地模式运行时通过KeyboardInterrupt终止应用
            sys.stdout.flush()
            im_names_desc = tqdm(loop()) #用1到无穷的生成器初始化进度条，用来统计图像帧
            #初始化Profiler用于统计时间 det time| pose time| post processing
            profiler = Profiler(['dt','pt','pn'],['det time: {:.4f}','pose time: {:.4f}','post processing: {:.4f}'])
            args = self.opt
            batchSize = args.posebatch
            # if args.flip:batchSize = int(batchSize / 2)
            for i in im_names_desc:
                if self.__toKillEvent.is_set(): #杀死进程
                    self.__kill()
                    return
                if self.stopped.is_set(): #停止进程
                    self.__stop()
                    return
                if args.profile:profiler.start() #开始统计时间
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.read()
                    if orig_img is None:break
                    if boxes is None or boxes.nelement() == 0: #目标检测结果为空，直接输出原图像
                        self.actRecg.step(None, None, None, None, None, orig_img, os.path.basename(im_name))
                        continue
                    if args.profile:profiler.step('dt') #目标检测计时
                    hm = self.poseEstim.step(inps,self.det_loader.joint_pairs) # 姿态估计
                    if args.profile:profiler.step('pt') #姿态估计计时
                    self.actRecg.step(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))
                    #todo self.actRecg.stepJoin() 
                    if args.profile:profiler.step('pn') #行为识别计时
                #输出图像中的人体个数+执行每个步骤的时间
                if args.profile:im_names_desc.set_description(
                        'hm:{} |'.format(hm.shape[0]) + profiler.getResStr())
        except BaseException as e: 
            logger.exception(e)
            self.__stop()

    def image_get(self):
        while True:
            if self.__toKillEvent.is_set() or self.stopped.is_set():
                return
            # frame,out,result = self.actRecg.read(timeout=self.opt.timeout)
            # if isinstance(frame, np.ndarray):
            frame,out,result = self.actRecg.read()
            self.mp_dict['img'] = (frame,out,result)
            self.mp_dict['fresh'] = True
            
    def start_worker(self, target,name):
        p = mp.Process(target=target,name=name, args=())
        p.start()
        return p

    def load_model(self):
        self.result_worker = self.start_worker(self.work,'PoseProcess')
        if(not self.opt.localvis):
            self.get_worker = self.start_worker(self.image_get,'getProcess')
        return self
    
    def wait_model_loaded(self):
        self.loadedEvent.wait()
    
    def is_model_loaded(self):
        return self.loadedEvent.is_set()

    @logger.catch
    def start(self,input_source):
        assert self.loadedEvent.is_set(),'model not loaded'
        self.__toStartEvent.set()
        logger.debug('started WebCamDetector...')
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
        if(not self.opt.localvis):
            self.get_worker.join()

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
        if(not self.opt.localvis):
            self.get_worker.terminate()
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
        # return self.actRecg.read(timeout=timeout)
        if (self.mp_dict['fresh']):
            return self.mp_dict['img']
        return None
    
    def count(self):
        return self.actRecg.count()


    
        

        

