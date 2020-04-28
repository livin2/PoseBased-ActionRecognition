
def get_detector(opt=None):
    if opt.detector == 'yolo':
        from detector.yolo_api import YOLODetector
        from .default_yolo_cfg import cfg
        return YOLODetector(cfg, opt)
    elif opt.detector == 'tracker':
        from detector.tracker_api import Tracker
        from .default_tracker_cfg import cfg
        return Tracker(cfg, opt)
    elif opt.detector.startswith('efficientdet_d'):
        from detector.effdet_api import EffDetDetector
        from detector.effdet_cfg import cfg
        return EffDetDetector(cfg, opt)
    else:
        raise NotImplementedError