import torch
import platform
from easydict import EasyDict as edict

args = edict()
args.cfg='model/pose_res50/256x192_res50_lr1e-3_1x.yaml'
args.checkpoint='model/pose_res50/fast_res50_256x192.pth'
args.inputpath='' #image-directory
args.inputlist=''   #image-list
args.inputimg='' #image-name
args.outputpath='res/out' #output-directory
args.save_img=False
args.vis=False
args.showbox=True
args.profile = True #'add speed profiling at screen output'
args.format = None

args.min_box_area=0 #min box area to filter out
args.eval=False  #save the result json as coco format, using image index(int) instead of image name(str)
args.qsize=1024 #the length of result buffer, where reducing it will lower requirement of cpu memory
args.flip=True #enable flip testing
args.debug = False #print detail information

args.video="" #video-name
args.webcam=-1 #webcam number
args.save_video = False
args.vis_fast = True
args.pose_track = False

args.gpus = "0"
args.detbatch = 5 #detection batch size PER GPU
args.posebatch = 80 #pose estimation maximum batch size PER GPU
args.detector = 'yolo' #yolo/tracker
args.sp = False #single process

args.tracking = (args.detector == 'tracker')
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#mutiprocess trigger
# if platform.system() == 'Windows': args.sp = True

#mutigpus
args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
# args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
# args.detbatch = args.detbatch * len(args.gpus)
# args.posebatch = args.posebatch * len(args.gpus)

#addition
args.localvis = True
args.realtime = True
args.inqsize = 2 
args.outqsize = 2 
args.posemodel = 'resnet50'
args.classmodel = 'dnnsingle9'
args.timeout = 5 #timeout in ? seconds
#### ----------------------edit------------------------------
args.webcam = 0
# args.video="res_data/run_3.avi"
# args.inputpath='res_data/sit_img'
args.detbatch = 1
args.posebatch = 5
# args.qsize=2
args.detector = 'yolo' #yolo/tracker

