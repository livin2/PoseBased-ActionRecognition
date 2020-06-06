from itertools import count
from threading import Thread
from queue import Queue
import os
import cv2
import numpy as np
import time
import torch
import torch.multiprocessing as mp

from loguru import logger
from alphapose.utils.presets import SimpleTransform
from multiprocessing.synchronize import Event as EventType
from config.apis import get_detector

## fork from alphapose.util.webcam_detector

class WebCamDetectionLoader():
    def __init__(self,pose_cfg, opt):
        self.cfg = pose_cfg
        self.opt = opt

        self._input_size = pose_cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = pose_cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = pose_cfg.DATA_PRESET.SIGMA
        if pose_cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False)

        self._stopped = mp.Value('b', False)

        if(opt.realtime==True):self.opt.inqsize=2
        self.pose_queue = mp.Queue(maxsize=self.opt.inqsize)

        self.loadedEvent = mp.Event()
        self.runningEvent = mp.Event()
        self.detector = None
        # self.__set_input(input_source)
        # self.path = mp.Value('i',-1)
        self.path = mp.Queue(maxsize=1)

    def __set_input(self,input_source):
        stream = cv2.VideoCapture(input_source)
        assert stream.isOpened(), 'Cannot capture source'
        # self.path.value = int(input_source)
        logger.info('input:{}',input_source)
        self.path.put(input_source)
        stream.release()

    def start_worker(self, target):
        p = mp.Process(target=target,name='WebCamDetector',args=())
        p.start()
        return p

    def start(self,startEvent=None):
        # start a thread to pre process images for object detection
        self.startEvent = startEvent
        logger.info('start:')
        print(self.startEvent)
        image_preprocess_worker = self.start_worker(self.frame_preprocess)
        # self.image_preprocess_worker = image_preprocess_worker
        return [image_preprocess_worker]
    
    def run(self,input_source):
        self.__set_input(input_source)
        self.runningEvent.set()

    @logger.catch
    def stop(self):
        # end threads
        self._stopped.value = True
        self.runningEvent.set()
        self.clear_queues()
        self.pose_queue.put((None, None, None, None, None, None,None))
        # self.image_preprocess_worker.join()
        # clear queues
        

    def terminate(self):
        self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)
            queue.get() if self.opt.realtime and queue.qsize()>1 else time.sleep(0.01)

    def wait_and_get(self, queue):
        if not self.stopped:
            return queue.get()

    def __load_model(self):
        self.detector = get_detector(self.opt)
        self.detector.load_model() ##

    def hangUp(self):
        self.runningEvent.clear()
        self.clear_queues()

    def onStop(self):
        self.clear_queues()
        logger.debug('on stop')
        self.pose_queue.put((None, None, None, None, None, None,None)) 


    @logger.catch
    def frame_preprocess(self):
        logger.info('%s Process (%s)' % (self.__class__,os.getpid()))
        if (self.detector is None):self.__load_model()
        self.loadedEvent.set()
        if(isinstance(self.startEvent,EventType)):self.startEvent.wait()
        while True:
            assert self.startEvent.is_set(),'Detector not started'
            self.runningEvent.wait()
            if self.stopped:
                self.onStop()
                return
            inputpath = self.path.get()
            logger.info('input:{}',inputpath)
            stream = cv2.VideoCapture(inputpath)
            assert stream.isOpened(), 'Cannot capture source'
            for i in count():
                if self.stopped: #停止
                    stream.release()
                    self.onStop()
                    return
                if not self.runningEvent.is_set(): #暂停
                    stream.release()
                    self.hangUp()
                    break
                if not self.pose_queue.full():
                    (grabbed, frame) = stream.read()
                    if not grabbed: #往输出队列放入空对象，continue
                        logger.debug('not grabbed')
                        self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                        stream.release()
                        return
                    #预处理
                    # expected frame shape like (1,3,h,w) or (3,h,w)
                    img_k = self.detector.image_preprocess(frame) 
                    if isinstance(img_k, np.ndarray):
                        img_k = torch.from_numpy(img_k)
                    # add one dimension at the front for batch if image shape (3,h,w)
                    if img_k.dim() == 3:
                        img_k = img_k.unsqueeze(0)
                    im_dim_list_k = frame.shape[1], frame.shape[0]
                    orig_img = frame[:, :, ::-1]
                    im_name = str(i) + '.jpg'
                    with torch.no_grad():
                        # Record original image resolution
                        im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
                    img_det = self.image_detection((img_k, orig_img, im_name, im_dim_list_k)) #目标检测
                    self.image_postprocess(img_det) #后处理

    def image_detection(self, inputs):
        img, orig_img, im_name, im_dim_list = inputs
        if img is None or self.stopped:
            return (None, None, None, None, None, None, None)

        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return (orig_img, im_name, None, None, None, None, None)
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            if self.opt.tracking:
                ids = dets[:, 6:7]
            else:
                ids = torch.zeros(scores.shape)

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return (orig_img, im_name, None, None, None, None, None)
        inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes_k.size(0), 4)
        return (orig_img, im_name, boxes_k, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self, inputs):
        with torch.no_grad():
            (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = inputs
            if orig_img is None or self.stopped:
                logger.debug('not grabbed')
                self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                return
            if boxes is None or boxes.nelement() == 0:
                self.wait_and_put(self.pose_queue, (None, orig_img, im_name, boxes, scores, ids, None))
                return
            # imght = orig_img.shape[0]
            # imgwidth = orig_img.shape[1]                print(type(box))
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                # if not hasattr(self,'checksize'):
                #     print(orig_img.shape)
                #     print(inps[i].shape)
                #     self.checksize = True

                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)

            self.wait_and_put(self.pose_queue, (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes))

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        return self._stopped.value

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
