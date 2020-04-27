from alphapose.utils.vis import getTime
import numpy as np
class profiler():
    def __init__(self,profile_dict):
        self.runtime_profile = profile_dict
    def start(self):
        self.ckpt_time = getTime()
    def step(self,key):
        self.ckpt_time, self.duration = getTime(self.ckpt_time)
        self.runtime_profile[key].append(self.duration)
    def getMean(self,key):
        return np.mean(self.runtime_profile[key])