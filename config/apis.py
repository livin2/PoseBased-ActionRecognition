
def set_posemodel_cfg(args):
    if args.posemodel.lower() == 'resnet50':
        args.cfg = 'model/pose_res50/256x192_res50_lr1e-3_1x.yaml'
        args.checkpoint = 'model/pose_res50/fast_res50_256x192.pth'
        return args
    elif args.posemodel.lower() == 'resnet50-dcn':
        args.cfg = 'model/pose_res50_dcn/256x192_res50_lr1e-3_2x-dcn.yaml'
        args.checkpoint = 'model/pose_res50_dcn/fast_dcn_res50_256x192.pth'
        return args
    elif args.posemodel.lower() == 'resnet152-duc':
        args.cfg = 'model/pose_res152_dcu/256x192_res152_lr1e-3_1x-duc.yaml'
        args.checkpoint = 'model/pose_res152_dcu/fast_421_res152_256x192.pth'
        return args
    raise NotImplementedError

def get_classifier_cfg(args):
    if args.classmodel.lower() == 'dnnsingle9':
        from config.act_dnnsingle_cfg import cfg9 
        return cfg9
    elif args.classmodel.lower() == 'fclstm9':
        from config.act_fclstm_cfg import cfg9 
        return cfg9
    raise NotImplementedError

def get_detector_cfg(args):
    pass

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