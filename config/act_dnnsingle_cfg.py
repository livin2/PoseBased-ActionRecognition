from easydict import EasyDict as edict

cfg9 = edict()
cfg9.MODEL = 'dnnSingle'
cfg9.CHEKPT = 'model/act_dnnSingle_9/epoch_1000.pth'
cfg9.tagI2W = ["jump","kick","punch","run","sit","squat","stand","walk","wave"]

# cfg.CHEKPT = 'model/act_dnnSingle_5/epoch_1000.pth'
# cfg.tagI2W = ["jump","run","sit","stand","walk"]