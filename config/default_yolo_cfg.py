from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'model/detector/yolo/yolov3-spp.cfg'
cfg.WEIGHTS = 'model/detector/yolo/yolov3-spp.weights'
cfg.INP_DIM =  608
cfg.NMS_THRES =  0.1
cfg.CONFIDENCE = 0.9
cfg.NUM_CLASSES = 80