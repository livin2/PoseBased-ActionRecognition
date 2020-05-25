import os
import shutil
import json
import pandas as pd
import numpy as np
from loguru import logger

execResjson = "find -type f -iname 'alphapose-results.json'"
execVis = "viewnior %s"

def visAndChoose(pardir):
    vispath = os.path.join(pardir,'vis')
    name = os.path.split(pardir)[1]
    os.system(execVis%vispath)
    print('remove %s?(Y/N):'%name,end='')
    x = input()
    if(x.lower()=='y'):
        logger.info('{} remove to {}',name,dumppath)
        shutil.move(pardir,dumppath)

if __name__ == "__main__":
    os.chdir("..")
    logger.add('data_pre/KeyPointFilter_{time}.log')
    dumppath = 'dump'
    if not os.path.exists(dumppath):os.makedirs(dumppath)
    try:
        with os.popen(execResjson) as p:
            jsl = p.read().splitlines()
            for jsit in jsl:
                pardir = os.path.split(jsit)[0]
                visAndChoose(pardir)
                # delImgToDel(getImgToDel(jsit))
    except BaseException as e:
        logger.warning('script only work in linux')
        logger.error(e)