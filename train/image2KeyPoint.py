import os
import torch
import sys
from loguru import logger
from tqdm import tqdm

# print(os.getcwd())
torch.multiprocessing.set_start_method('forkserver', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

# IN_PATH = 'train/data_orig'
EXEC = "find -maxdepth 5 -ipath '*.avi' -a -ipath '*/data_orig/*'"
def getVideoList():
    f = os.popen(EXEC)
    l = f.read().split('\n')
    return l
    # lslist = os.listdir(IN_PATH)
    # for it in lslist:
    #     if(not os.path.isdir(it)):
    #         logger.info('{} not dir',it)
    #         continue

def getActTagFromPath(path):
    p = os.path.split(path)[0]
    return os.path.split(p)[1]

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

@logger.catch
def process(idx,input_source,pose_cfg,poseEstim,detector):
    print('processing %s...'%input_source)
    det_loader = DetectionLoader(input_source,detector,pose_cfg, args, 
        batchSize=args.detbatch, mode='video', queueSize=args.qsize)
    det_worker = det_loader.start()

    writer = DataWriter(pose_cfg, args, save_video=False, queueSize=args.qsize).start()

    batchSize = args.posebatch
    if args.flip:batchSize = int(batchSize / 2)
    det_loader.loadEvent.wait()
    try:
        profiler = Profiler(['dt','pt','pn'],['det time: {:.4f}','pose time: {:.4f}','post processing: {:.4f}'])
        im_names_desc = tqdm(range(det_loader.length-1), dynamic_ncols=True)
        sys.stdout.flush()
        for i in im_names_desc:
            if args.profile:profiler.start() 
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:break
                if boxes is None or boxes.nelement() == 0: 
                    writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                    continue
                if args.profile:profiler.step('dt') 
                hm = poseEstim.step(inps,det_loader.joint_pairs) 
                if args.profile:profiler.step('pt') 
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))
                if args.profile:profiler.step('pn') 
            sys.stdout.flush()
            # print()
            if args.profile:im_names_desc.set_description(
                    'hm:{} |'.format(hm.shape[0]) + profiler.getResStr())
        while(writer.running()):
            time.sleep(1)
            print(str(writer.count()) + ' images remain...')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        logger.error(repr(e))
        logger.error('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        for p in det_worker:p.terminate()
        writer.commit()
        writer.clear_queues()
    final_result = writer.results()
    write_json(final_result, args.outputpath, form=args.format, for_eval=args.eval)
    logger.info("Results have been written to {}",args.outputpath)
    return 'v%d save to '+args.outputpath

import cv2
if __name__ == "__main__":
    path = os.path.dirname(os.getcwd())
    sys.path.append(path)

    # from alphapose.utils.detector import DetectionLoader
    from utils.detector import DetectionLoader
    from server_process import PoseEstimator
    from alphapose.utils.writer import DataWriter
    from utils import Profiler
    from alphapose.utils.pPose_nms import write_json
    from alphapose.utils.config import update_config
    from config.default_args import args
    from config.apis import get_detector
    OUTROOTPATH='train/out' #output-directory
    args.save_img=True

    offset = 0
    os.chdir("..")
    mkdirs(OUTROOTPATH)
    vlist = getVideoList()
    print(vlist[1365:1368])
    inps = tqdm(vlist)
    # inps = vlist
    # logger.info(vlist)
    pose_cfg = update_config(args.cfg)
    poseEstim = PoseEstimator(pose_cfg,args)
    logger.debug('loading pose model...')
    poseEstim.load_model()
    logger.debug('pose model loaded')
    logger.debug('loading det model...')
    detector = get_detector(args)
    detector.load_model()
    logger.debug('pose det loaded')
    
    for idx,inp in enumerate(inps):
        idx += offset
        if idx<offset:continue
        args.outputpath = OUTROOTPATH+'/'+getActTagFromPath(inp)+'_%d'%idx
        mkdirs(args.outputpath)
        if(os.path.splitext(fpath)[1] == '.avi'):
            des = process(idx,inp,pose_cfg,poseEstim,detector)
        if des is not None:inps.set_description(des%idx)