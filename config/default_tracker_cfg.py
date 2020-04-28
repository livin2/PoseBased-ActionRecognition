from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'model/detector/tracker/yolov3.cfg'
cfg.WEIGHTS = 'model/detector/tracker/jde.1088x608.uncertainty.pt'
cfg.IMG_SIZE =  (1088, 608)
cfg.NMS_THRES =  0.6
cfg.CONFIDENCE = 0.4
cfg.BUFFER_SIZE = 30 # frame buffer