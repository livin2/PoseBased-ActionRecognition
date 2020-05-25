import os
import shutil
import json
import pandas as pd
import numpy as np
from loguru import logger

execResjson = "find -type f -iname 'alphapose-results.json'"

def getVisDir(path):
    p = os.path.split(path)[0]
    return os.path.join(p,'vis')

def getImgToDel(jsit):
    with open(jsit) as jsf:
        js = json.load(jsf)
        resls = {it['image_id'] for it in js}
        visdir = getVisDir(jsit)
        imgls = os.listdir(visdir)
        todel = [os.path.join(visdir,im) for im in imgls if im not in resls]
        return todel

def delImgToDel(todel):
    for jsf in todel:
        os.remove(jsf)
        logger.info('{} deleted.',jsf)

if __name__ == "__main__":
    os.chdir("..")
    logger.add('data_pre/nanFilter_{time}.log') 
    try:
        with os.popen(execResjson) as p:
            jsl = p.read().splitlines()
            for jsit in jsl:
                delImgToDel(getImgToDel(jsit))
    except BaseException as e:
        logger.warning('script only work in linux')
        logger.error(e)
