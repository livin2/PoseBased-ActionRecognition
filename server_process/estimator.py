import torch
import torch.multiprocessing as mp
import numpy as np
import sys
sys.path.append("..")
    
from loguru import logger
from easydict import EasyDict as edict
from alphapose.models import builder
from alphapose.utils.transforms import flip, flip_heatmap

class PoseEstimator():
    def __init__(self,cfg,opt):
        self.opt = opt
        self.cfg = cfg
        self.pose_model = None
    
    def load_model(self):
        cfg = self.cfg
        args = self.opt
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        logger.info(f'Loading pose model from {args.checkpoint}..')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        if len(args.gpus) > 1:
            self.pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
        else:
            self.pose_model.to(args.device)
        self.pose_model.eval()
        
    @logger.catch
    def step(self,inps,joint_pairs):
        args = self.opt
        batchSize = args.posebatch
        with torch.no_grad():
            inps = inps.to(args.device)
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                if args.flip:
                    inps_j = torch.cat((inps_j, flip(inps_j)))
                hm_j = self.pose_model(inps_j)
                if args.flip:
                    hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):],joint_pairs, shift=True)
                    hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                hm.append(hm_j)
            hm = torch.cat(hm)
            hm = hm.cpu()
        return hm

