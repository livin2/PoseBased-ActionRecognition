import numpy as np
import torch
import cv2

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

#fork from alphapose.utils.vis
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
def vis_frame_fast(frame, im_res, add_bbox=False, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        NotImplementedError

    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    for index,human in enumerate(im_res['result']):
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        # Draw bboxes
        if add_bbox:
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]
            else:
                from PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)
            # color = get_color_fast(int(abs(human['idx'][0])))
            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), BLUE, 2)
            # Draw indexes of humans
            if 'idx' in human.keys():
                cv2.putText(img, ''.join(str(index)), (int(bbox[0])+26, int((bbox[2] + 46))), DEFAULT_FONT, 2, RED, 2)
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.35:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2 * (kp_scores[start_p] + kp_scores[end_p]) + 1)
    return img