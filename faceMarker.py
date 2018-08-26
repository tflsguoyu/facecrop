import os
import sys
import glob
import cv2
import math
import numpy as np
import face_recognition as fr

def _find_center_pt(points):
    '''
    find centroid point by several points that given
    '''
    x = 0
    y = 0
    num = len(points)
    for pt in points:
        x += pt[0]
        y += pt[1]
    x //= num
    y //= num
    return (x,y)

def _angle_between_2_pt(p1, p2):
    '''
    to calculate the angle rad by two points
    '''
    x1, y1 = p1
    x2, y2 = p2
    tan_angle = (y2 - y1) / (x2 - x1)
    return (np.degrees(np.arctan(tan_angle)))

def _get_rotation_matrix(left_eye_pt, right_eye_pt, nose_center):
    '''
    to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
    '''
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    M = cv2.getRotationMatrix2D((nose_center[0]/2, nose_center[1]/2), eye_angle, 1)
    return M

def _crop_face(img, face_loc, padding_size=1):
    '''
    crop face into small image, face only, but the size is not the same
    '''
    H, W, C = img.shape
    top = face_loc[0]
    right = face_loc[1]
    down = face_loc[2]
    left = face_loc[3]
    h = down - top
    w = right - left
    top = top - int(h * padding_size)
    down = down + int(h * padding_size)
    left = left - int(w * padding_size)
    right = right + int(w * padding_size)

    if top < 0:
        top = 0
    if down > H+1:
        down = H+1
    if left < 0:
        left = 0
    if right > W+1:
        right = W+1
    img_crop = img[top:down, left:right]
    return img_crop

def _crop_face_o(img, face_lm, scale=1, adjust=0):
    '''
    crop face into small image, face only, but the size is not the same
    '''
    lm = []
    for facial_feature in face_lm.keys():
        lm.extend(face_lm[facial_feature])
    
    top = 99999
    down = -99999
    left = 99999
    right = -99999
    for lm_this in lm:
        if lm_this[1] < top:
            top = lm_this[1]
        if lm_this[1] > down:
            down = lm_this[1]
        if lm_this[0] < left:
            left = lm_this[0]
        if lm_this[0] > right:
            right = lm_this[0]
    
    diff_TD = down - top
    diff_LR = right - left
    if diff_TD < diff_LR:
        shift = diff_LR
    else:
        shift = diff_TD

    shift2 = int(scale*shift)
    down = down + int(shift2*adjust)
    top = down - shift2
    left = left - int((scale-1)/2*shift)
    right = left + shift2

    H, W, C = img.shape
    if top < 0:
        top = 0
    if down > H+1:
        down = H+1
    if left < 0:
        left = 0
    if right > W+1:
        right = W+1

    img_crop = img[top:down, left:right]
    return img_crop

def _padding(img_o):
    H, W, C = img_o.shape
    D = int(math.sqrt(H*H+W*W))
    img = np.zeros((D, D, 3), dtype='uint8')
    off_h = int((D - H)/2)
    off_w = int((D - W)/2)
    img[off_h:(off_h + H), off_w:(off_w + W), :] = img_o
    return img

def _draw_kp(img, face_lm, size):
    lm = []
    for facial_feature in face_lm.keys():
        lm.extend(face_lm[facial_feature])

    for lm_this in lm:
        img[lm_this[1]-size:lm_this[1]+size,lm_this[0]-size:lm_this[0]+size,0] = 0
        img[lm_this[1]-size:lm_this[1]+size,lm_this[0]-size:lm_this[0]+size,1] = 255
        img[lm_this[1]-size:lm_this[1]+size,lm_this[0]-size:lm_this[0]+size,2] = 0
    return img

def _save_lm_to_file(fn, face_lm, h, w):
    lm = []
    for facial_feature in face_lm.keys():
        lm.extend(face_lm[facial_feature])

    f = open(fn, 'w+')
    for lm_this in lm:
        f.write('%.5f %.5f\n' % (lm_this[0]/w,lm_this[1]/h))
    f.close()
    

# Main: 
dir_input = sys.argv[1]
dir_faces = dir_input + '_faces/'
dir_markers = dir_input + '_markers/'
dir_facesMarkers = dir_input + '_facesMarkers/'
os.makedirs(dir_faces, exist_ok=True)
os.makedirs(dir_markers, exist_ok=True)
os.makedirs(dir_facesMarkers, exist_ok=True)
os.makedirs(dir_faces + dir_faces, exist_ok=True)
os.makedirs(dir_markers + dir_markers, exist_ok=True)
os.makedirs(dir_facesMarkers + dir_facesMarkers, exist_ok=True)

fn_list = os.listdir(dir_input)
for fn in fn_list:
    # print('Processing ' + fn + ' ...')

    img = cv2.imread(dir_input + '/' + fn)
    try:
        img.shape    
    except:
        print('image can not be read')
        continue

    H,W,C = img.shape
    
    dir1 = dir_faces
    # imvar = cv2.Laplacian(img[int(H/4):int(3*H/4),int(W/4):int(3*W/4),:], cv2.CV_64F).var()
    imvar=np.mean(img)
    
    # face_lm_list = fr.face_landmarks(img)
    # if len(face_lm_list) == 0:
    #     print(fn + ': %d face found' % (len(face_lm_list)))
    #     continue
    # if len(face_lm_list) == 1:
    #     dir1 = dir_faces
    #     dir2 = dir_markers
    #     dir3 = dir_facesMarkers
    # elif len(face_lm_list) > 1:
    #     dir1 = dir_faces + dir_faces
    #     dir2 = dir_markers + dir_markers
    #     dir3 = dir_facesMarkers + dir_facesMarkers
    #     print(fn + ': %d face found' % (len(face_lm_list)))

        
    # img_kp = img.copy()
    # face_lm = face_lm_list[0]
    
    # img_kp = _draw_kp(img_kp, face_lm, size=max(2,int(min(H,W)/400)))
    
    # _save_lm_to_file(dir2 + '%.5f.txt' % imvar, face_lm, H, W)
    # cv2.imwrite(dir3 + '%.5f_kp.jpg' % imvar, img_kp);

    cv2.imwrite(dir1 + '%.5f.jpg' % imvar, img);
    
