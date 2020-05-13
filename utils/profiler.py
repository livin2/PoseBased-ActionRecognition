from alphapose.utils.vis import getTime
import numpy as np
class profiler():
    def __init__(self,keys,describes=None):
        if(describes is None):
            self.context_dict = {k:([],k+': {}') for k in keys}
        else:
            # self.context_dict = {k:([],d) for k in keys for d in describes}
            self.context_dict = {k:([],describes[i]) for i,k in enumerate(keys)}
    def start(self):
        self.ckpt_time = getTime()
    def step(self,key):
        self.ckpt_time, self.duration = getTime(self.ckpt_time)
        self.context_dict[key][0].append(self.duration)
    def getMean(self,key):
        return np.mean(self.context_dict[key][0])
    def getResStr(self):
        res = ''
        for k,(l,des) in self.context_dict.items():
            res += des.format(np.mean(l))
            res += ' |'
        return res
            