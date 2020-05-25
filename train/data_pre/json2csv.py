import os
import shutil
import json
import pandas as pd
import numpy as np
from loguru import logger

execResjson = "find -type f -iname 'alphapose-results.json'"

alphaI2W = [ "nose","LEye","REye","LEar","REar","LShoulder","RShoulder", "LElbow","RElbow",\
"LWrist", "RWrist","LHip","RHip", "LKnee","Rknee", "LAnkle","RAnkle"]# neck is addtion

@logger.catch
def cleanJson(jslist:list):
    # jsitem = jslist[0]
    dicts =[]
    for jsitem in jslist:
        itKeypoints = np.array(jsitem['keypoints']).reshape(-1,3)
        idx = jsitem['idx']
        imgid= jsitem['image_id']
        if(len(idx)>1):
            # logger.debug('muti-idx: %s | %s'%(imgid,idx.__str__()))
            idx = idx[0][0]
        else:
            idx= idx[0]
        d={'image_id': imgid,'idx':idx}
        for i,xys in enumerate(itKeypoints):
            d[alphaI2W[i]+'_x'] = xys[0]
            d[alphaI2W[i]+'_y'] = xys[1]
        dicts.append(d)
    return dicts

def getParDirFromPath(path):
    p = os.path.split(path)[0]
    return os.path.split(p)[1]

@logger.catch    
def Jsons2Csv(filelist:list):
    for fpath in filelist:
        if(os.path.splitext(fpath)[1] != '.json'):
            logger.info('{} isn\'t json,skip',fpath)
            continue
        with open(fpath,'r') as f:
            jslist = json.load(f)
            df = pd.DataFrame(cleanJson(jslist))
            if(df['image_id'].duplicated().sum()>0):
                mode = df['idx'].mode().item()
                df = df.loc[df['idx'] == mode]
            _fpath =os.path.join(outpath,'%s.csv'%getParDirFromPath(fpath)) 
            df.to_csv(_fpath)
            logger.info('%d rows to %s'%(df.shape[0],_fpath))
            # os.remove(fpath)
            
if __name__ == "__main__":
    os.chdir("..")
    logger.add('data_pre/json2csv_{time}.log') 
    outpath = 'data/hdmb'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with os.popen(execResjson) as p:
        jsonlist = p.read().splitlines()
        Jsons2Csv(jsonlist)

    



