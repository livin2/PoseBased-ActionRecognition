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
from alphapose.utils.vis import vis_frame_fast as vis_frame    
from multiprocessing.synchronize import Event as EventType

from actRec.F import single_normalize_min_
from actRec import models
from actRec.models import get_model

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
class classifier():
    def __init__(self,cfg,opt,queueSize=2): #webcam queue= 2
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.cfg = cfg
        self.heatmap_to_coord = heatmap_to_coord_simple
        self.inqueue = mp.Queue(maxsize=queueSize)
        self.model = None
        self.loadedEvent = mp.Event()
        self.holder = edict()
        # 

    def step(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.wait_and_put(self.inqueue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))
    
    def __load_model(self):
        self.model = get_model(self.cfg.MODEL,self.cfg.tagI2W)
        logger.info(self.model)
        ckpt = torch.load(self.cfg.CHEKPT,map_location=self.opt.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        logger.info('epoch:',ckpt['epoch'])
        logger.info('loss:',ckpt['loss'])
        if len(self.opt.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model
                , device_ids=self.opt.gpus).to(self.opt.device)
        else:
            self.model.to(self.opt.device)
        self.model.eval()
    
    def __output(self,img,out):
        self.showimg_all(img, out)
        
    def work(self):
        logger.info('Classifier Process (%s)' % os.getpid())
        if(self.model is None):self.__load_model()
        self.loadedEvent.set()
        self.model.exe_pre(self.opt.device,self.holder)
        
        if(isinstance(self.startEvent,EventType)):self.startEvent.wait()
        
        while True:
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.inqueue.get()
            if orig_img is None: return
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]  # image channel RGB->BGR

            if boxes is None:
                self.__output(orig_img, None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                pred = hm_data.cpu().data.numpy()
                assert pred.ndim == 4

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
                result = {
                    'imgname': im_name,
                    'result': result_orig
                }       
                img = vis_frame(orig_img, result, add_bbox=(self.opt.pose_track | self.opt.tracking))
                if(len(result['result'])<=0):continue
                points = result['result'][0]['keypoints'].numpy()

                points = single_normalize_min_(points)
                points = points.reshape(1,34)

                out = self.model.exe(points,self.opt.device,self.holder)
                self.__output(img,out)

        self.clear_queues()

    def showimg(self,img,tag):
        # height, width = img.shape[:2]
        img = cv2.putText(img, tag, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        img = np.array(img, dtype=np.uint8)[:, :, ::-1]
        cv2.imshow("AlphaPose Demo", img)
        k = cv2.waitKey(100) 

    def showimg_all(self,img,out):
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
        cv2.imshow("AlphaPose Demo", img)
        k = cv2.waitKey(100) 

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
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
        cv2.destroyAllWindows()
        self.result_worker.join()
        self.clear_queues()
        print('classifier stop')
        
    
    def wait_and_put(self, queue, item):
        queue.put(item)
        # queue.get() if queue is self.final_result_queue and queue.qsize()>1 else time.sleep(0.01)
        
    def clear_queues(self):
        self.clear(self.inqueue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def count(self):
        return self.inqueue.qsize()
