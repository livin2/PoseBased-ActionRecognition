#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

import pysnooper
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import json

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader

from actRec.F import check_input,print_finish_info,loop
from config.default_args import args

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
cfg = update_config(args.cfg)
# estimator_cfg = update_config(args.cfg)
mode, input_source = check_input(args)
print('mode:',mode)
print('input_source:',input_source)

from detector.tracker_api import Tracker
from config.default_tracker_cfg import cfg as detector_cfg
from config.default_classifier_cfg import cfg as classifier_cfg
from server_process import ActClassifier,PoseEstimator

def main():
    actrecg.start()
    # actrecg.loadedEvent.wait()
    logger.info('Main Process (%s)' % os.getpid())

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    detector = Tracker(detector_cfg, args)
    det_loader = WebCamDetectionLoader(input_source, detector, cfg, args).start()

    # Load pose model
    poseEstim.load_model()

    #det time| pose time| post processing
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
        # 'cn': []
    }

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())

    # print('outer:',det_loader,det_loader.pose_queue.qsize())
    # # main process
    batchSize = args.posebatch
    # if args.flip:
    #     batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    # print('orig_img is None')
                    break
                if boxes is None or boxes.nelement() == 0:
                    # print('boxes is None')
                    actrecg.step(None, None, None, None, None, orig_img, os.path.basename(im_name))
                    # actrecg.put_img(None)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                
                hm = poseEstim.step(inps,det_loader.joint_pairs)

                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)

                actrecg.step(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()
        det_loader.stop()
        while(actrecg.running()):
                time.sleep(1)
                print('===========================> remaining ' + str(actrecg.count()) + ' images...')
        # actrecg.clear_queues()
        print('before call stop')
        actrecg.stop()
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            # det_loader.clear_queues()

if __name__ == "__main__":
    main()


