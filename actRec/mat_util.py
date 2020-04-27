import scipy.io as sio
import numpy as np
import pandas as pd

def mat2dict(mio):
    keys = mio._fieldnames
    dtmp = {}
    for i in keys:
        item = mio.__dict__[i]
        if(type(item)!= np.ndarray):
            raise Exception(i,type(item))
        item = item.reshape(1,-1)[0]
        if(len(item)==1):
            dtmp[i] = item2dict(item[0])
            continue
        l = []
        for it in item:
            l.append(item2dict(it))
        dtmp[i]=l
    return dtmp
    
def item2dict(item):
    # if(type(item)==np.uint8):
    #     return item
    # if(type(item)==np.uint16):
    #     return item
    # if(type(item)==np.str_):
    #     return item
    # if(type(item)==np.float64):
    #     return item
    if(type(item)==sio.matlab.mio5_params.mat_struct):
        return mat2dict(item)
    return item
    # raise Exception(type(item)) 

class MatNpBuild:
    def __init__(self,matnp):
        assert type(matnp) == np.ndarray
        self.data = matnp.reshape(-1) #ndarray.shape = (-1,)
        self.i = 0
    def pos(self,i):
        self.i = i
        return self
    def get(self,key):
        return MatNpBuild(self.data[self.i].__dict__[key].reshape(-1))
    def val(self):
        if(len(self.data)<=self.i):
            return None
        return self.data[self.i]
    def len(self):
        return len(self.data)
    def keys(self):
        assert type(self.data[self.i]) == sio.matlab.mio5_params.mat_struct
        return self.data[self.i]._fieldnames

class MATBuild:
    def __init__(self,mat):
        self.data = mat #matlab.mio5_params.mat_struct
    def get(self,key,i=0):
        arr = self.data.__dict__[key].reshape(-1)
        if(len(arr)<=i):return MATBuild(None)
        return MATBuild(arr[i])
    def val(self):
        return self.data

# #@return ndarray.shape = (-1,)
# def matget(mat,key):
#     return mat.__dict__[key].reshape(-1)

def anno_nopoint(matlist):
    retl = []
    idx = 0
    for mat in matlist:
        mat = MATBuild(mat)
        tmpd = {}
        tmpd['imgidx'] = idx
        idx+=1
        tmpd['image.name'] = mat.get('image').get('name').val()
        tmpd['vididx'] = mat.get('vididx').val()
        tmpd['frame_sec'] = mat.get('frame_sec').val()
        retl.append(tmpd)
    return np.array(retl)

def anno_act_vid_getter(objnp,i):
    anl = MatNpBuild(objnp).get('annolist')
    act = MatNpBuild(objnp).get('act')
    vid = MatNpBuild(objnp).get('video_list')
    ret = {}
    ret['image.name'] = anl.pos(i).get('image').get('name').val()
    ret['act_name'] = act.pos(i).get('act_name').val()
    ret['cat_name'] = act.pos(i).get('cat_name').val()
    ret['act_id'] = act.pos(i).get('act_id').val()
    ret['vididx'] = anl.pos(i).get('vididx').val()
    ret['frame_sec'] = anl.pos(i).get('frame_sec').val()
    vidx = ret['vididx']
    if(vidx!=None):
        ret['YoutubeUrl'] = vid.data[vidx-1][0]
    return ret

def anno_act_vid(objnp,num = -1):
    anl = MatNpBuild(objnp).get('annolist')
    act = MatNpBuild(objnp).get('act')
    vid = MatNpBuild(objnp).get('video_list')
    retl = []
    if(num==-1):num = anl.len()
    for i in range(num):
        ret = {}
        ret['image.name'] = anl.pos(i).get('image').get('name').val()
        ret['act_name'] = act.pos(i).get('act_name').val()
        ret['cat_name'] = act.pos(i).get('cat_name').val()
        ret['act_id'] = act.pos(i).get('act_id').val()
        ret['vididx'] = anl.pos(i).get('vididx').val()
        ret['frame_sec'] = anl.pos(i).get('frame_sec').val()
        vidx = ret['vididx']
        if(vidx!=None):
            ret['YoutubeUrl'] = vid.data[vidx-1][0]
        retl.append(ret)
    return retl
mpiiidx = ['RAnkle','Rknee','RHip','LHip','Lknee','LAnkle'
,'Pelvis','Thorax','UpNeck','HeadTop'
,'RWrist','RElbow','RShoulder','LShoulder','LElbow','LWrist']
def anno_act_pos(objnp,num = -1):
    anl = MatNpBuild(objnp).get('annolist')
    act = MatNpBuild(objnp).get('act')
    retl = []
    if(num==-1):num = anl.len()
    for i in range(num):
        personnum = anl.pos(i).get('annorect').len()
        for j in range(personnum):
            if('annopoints' not in anl.pos(i).get('annorect').pos(j).keys()):
                continue
            ret = {}
            ret['image.name'] = anl.pos(i).get('image').get('name').val()
            ret['act_name'] = act.pos(i).get('act_name').val()
            ret['cat_name'] = act.pos(i).get('cat_name').val()
            ret['act_id'] = act.pos(i).get('act_id').val()
            
            perpos = anl.pos(i).get('annorect').pos(j).get('annopoints').get('point')
            for k in range(perpos.len()):
                id = perpos.pos(k).get('id').val()
                keystr = mpiiidx[id]
                ret[keystr+'_x'] = perpos.pos(k).get('x').val()
                ret[keystr+'_y'] = perpos.pos(k).get('y').val()
            retl.append(ret)
    return retl

def main():
    path = 'mpii.mat'
    obj = sio.loadmat(path,struct_as_record=False)['RELEASE']
    df = pd.DataFrame(anno_act_vid(obj))
    print(df)
    df.to_csv('anno_act_pos.csv')
    # df.to_csv('data/mpii_imganno_act_vid.csv')
    # df.dropna(how='any').to_csv('data/mpii_imganno_act_vid_noNa.csv')

if __name__ == '__main__':
    main()
# path = 'data/mpii.mat'
# obj = sio.loadmat(path,struct_as_record=False)['RELEASE'][0,0]
# obj = sio.loadmat('data/mpii.mat',struct_as_record=False)['RELEASE']

# anl = obj.__dict__['annolist']
# act = obj.__dict__['act']
# vid = obj.__dict__['video_list']