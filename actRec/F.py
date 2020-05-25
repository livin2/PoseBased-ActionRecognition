import numpy as np
def  referto_skeleton_(pose:np.ndarray,ref_idx:int):
    #inplace opt
    pose = pose.reshape(len(pose),-1,2)
    ref_x = pose[:,ref_idx,0] #.copy()
    ref_y = pose[:,ref_idx,1]
    for i in range(len(pose)):
        pose[i,:,0]-= ref_x[i]
        pose[i,:,1]-= ref_y[i]
    return pose
def  normalize_referto_skeleton_(pose:np.ndarray,ref_idx:int):
    #inplace opt
    pose = pose.reshape(len(pose),-1,2)
    ref_x = pose[:,ref_idx,0] #.copy()
    ref_y = pose[:,ref_idx,1]
    for i in range(len(pose)):
        pose[i,:,0]-= ref_x[i]
        pose[i,:,1]-= ref_y[i]
        
        xlen = np.max(pose[i,:,0]) - np.min(pose[i,:,0]) 
        ylen = np.max(pose[i,:,1]) - np.min(pose[i,:,1])
        
        if(xlen==0): pose[i,:,0]=0
        else:pose[i,:,0] /= xlen 
        if(ylen==0): pose[i,:,1]=0
        else:pose[i,:,1] /= ylen
    return pose

def normalize_min_(pose:np.ndarray):
    pose = pose.reshape(len(pose),-1,2)
    for i in range(len(pose)):
        xmin = np.min(pose[i,:,0]) 
        ymin = np.min(pose[i,:,1])
        xlen = np.max(pose[i,:,0]) - xmin
        ylen = np.max(pose[i,:,1]) - ymin
        
        if(xlen==0): pose[i,:,0]=0
        else:
            pose[i,:,0] -= xmin 
            pose[i,:,0] /= xlen 
        if(ylen==0): pose[i,:,1]=0
        else:
            pose[i,:,1] -= ymin
            pose[i,:,1] /= ylen
    return pose

def normalize_min_seq_(pose:np.ndarray):
    pose = pose.reshape(len(pose),-1,2)
    xmin = np.min(pose[:,:,0]) 
    ymin = np.min(pose[:,:,1])
    xlen = np.max(pose[:,:,0]) - xmin
    ylen = np.max(pose[:,:,1]) - ymin
    if(xlen==0): pose[:,:,0]=0
    else:
        pose[:,:,0] -= xmin 
        pose[:,:,0] /= xlen 
    if(ylen==0): pose[:,:,1]=0
    else:
        pose[:,:,1] -= ymin
        pose[:,:,1] /= ylen
    return pose
    
def single_normalize_min_(pose:np.ndarray):
    pose = pose.reshape(-1,2)
    xmin = np.min(pose[:,0]) 
    ymin = np.min(pose[:,1])
    xlen = np.max(pose[:,0]) - xmin
    ylen = np.max(pose[:,1]) - ymin
    if(xlen==0): pose[:,0]=0
    else:
        pose[:,0] -= xmin 
        pose[:,0] /= xlen 
    if(ylen==0): pose[:,:,1]=0
    else:
        pose[:,1] -= ymin
        pose[:,1] /= ylen
    return pose
