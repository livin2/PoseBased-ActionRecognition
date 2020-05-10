import torch.multiprocessing as mp
import os
import cv2
import numpy as np
import torch
import time
import sys
sys.path.append("..")
    
from loguru import logger
from easydict import EasyDict as edict
from alphapose.utils.transforms import heatmap_to_coord_simple
from alphapose.utils.pPose_nms import pose_nms
   
from multiprocessing.synchronize import Event as EventType

from actRec.F import single_normalize_min_
from actRec import models
from actRec.models import get_model
EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
class classifier():
    def __init__(self,cfg,opt,imgfn=None): 
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.cfg = cfg
        self.imgfn = imgfn
        self.heatmap_to_coord = heatmap_to_coord_simple
        self.inqueue = mp.Queue(maxsize=opt.inqsize) #webcam queue= 2
        self.outqueue = mp.Queue(maxsize=opt.outqsize)
        self.model = None
        self.loadedEvent = mp.Event()
        self.holder = edict()
        self.localvis=opt.localvis
        # 

    def step(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        if not self.result_worker.is_alive():
            logger.info('classifier is no running')
            return
        self.wait_and_put(self.inqueue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))
    
    def __load_model(self):
        self.model = get_model(self.cfg.MODEL,self.cfg.tagI2W)
        logger.info(self.model)
        ckpt = torch.load(self.cfg.CHEKPT,map_location=self.opt.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        logger.info('epoch:{%.4f}' % ckpt['epoch'])
        logger.info('loss:{%.4f}' % ckpt['loss'])
        if len(self.opt.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model
                , device_ids=self.opt.gpus).to(self.opt.device)
        else:
            self.model.to(self.opt.device)
        self.model.eval()
    
    def __output(self,img,out):
        if self.localvis:
            self.showimg(img, out)
        else:
            self.wait_and_put(self.outqueue,(img,out))
    
    def read(self,timeout=None):
        if(self.outqueue.empty()):
            assert self.result_worker.is_alive(),'classifier is no running'
        return self.outqueue.get(timeout=timeout)
        
    def work(self):
        logger.info('Classifier Process (%s)' % os.getpid())
        if(self.model is None):self.__load_model() #加载模型
        self.loadedEvent.set() #模型加载完成
        self.model.exe_pre(self.opt.device,self.holder) #模型初始化
        if(isinstance(self.startEvent,EventType)):self.startEvent.wait() #等待开始事件
        while True:
            with torch.no_grad():
                (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.inqueue.get()
                if orig_img is None: #输入为空返回
                    self.__output(None, None)
                    return
                orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]  #改变图像通道排序 image channel RGB->BGR 

                if boxes is None: #目标检测与姿态估计结果为空 直接输出原图像
                    self.__output(orig_img, None)
                else:
                    # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                    pred = hm_data.cpu().data.numpy()
                    assert pred.ndim == 4
                    #姿态估计结果预处理
                    pose_coords = []
                    pose_scores = []
                    for i in range(hm_data.shape[0]):
                        bbox = cropped_boxes[i].tolist()
                        pose_coord, pose_score = self.heatmap_to_coord(pred[i][EVAL_JOINTS], bbox)
                        pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                        pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                    preds_img = torch.cat(pose_coords)
                    preds_scores = torch.cat(pose_scores)
                    result_orig = pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)
        
                    #在图像上可视化目标检测框/姿态估计点
                    result = {'imgname': im_name,'result': result_orig}
                    from alphapose.utils.vis import vis_frame_dense as vis_frame 
                    img = vis_frame(orig_img, result, add_bbox=True)

                    if(len(result_orig)<=0):#姿态估计结果为空
                        self.__output(orig_img,None)
                        continue 

                    #对图像中每个人体进行行为识别
                    out=[]
                    for i in range(len(result_orig)):
                        points = result_orig[i]['keypoints'].numpy()
                        points = single_normalize_min_(points)
                        points = points.reshape(1,34)
                        actres = self.model.exe(points,self.opt.device,self.holder)
                        # out.append(actres)
                        out.append(actres.cpu())

                    self.__output(img,out)
        self.clear_queues() #结束子进程

    def showimg(self,img,out):
        if self.imgfn is not None:
            img = self.imgfn(img,out,self.cfg.tagI2W)
        cv2.imshow("AlphaPose Demo",img)
        k = cv2.waitKey(30) 

    def start_worker(self, target):
        p = mp.Process(target=target,name='ActClassifier', args=())
        p.start()
        return p

    def start(self,startEvent=None):
        self.startEvent = startEvent
        self.result_worker = self.start_worker(self.work)
        return self

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        # return False
        return not self.inqueue.empty()

    def stop(self):
        self.step(None,None,None,None,None,None,None)
        self.result_worker.join()
        self.clear_queues()
        logger.info('classifier stop')
        cv2.destroyAllWindows()
    
    def kill(self):
        self.clear_queues()
        self.result_worker.terminate()
    
    def wait_and_put(self, queue, item):
        queue.put(item)
        queue.get() if self.opt.realtime and queue.qsize()>1 else time.sleep(0.01)
        
    def clear_queues(self):
        self.clear(self.inqueue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def count(self):
        return self.inqueue.qsize()
