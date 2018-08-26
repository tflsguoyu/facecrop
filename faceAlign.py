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


# Main: 
dir_input = sys.argv[1]
dir_faces = dir_input + '_faces/'
os.makedirs(dir_faces, exist_ok=True)

fn_list = os.listdir(dir_input)
numOfImg = len(fn_list)
for idxfn, fn in enumerate(fn_list):
    print('Processing %d of %d: ' % (idxfn,numOfImg) + fn + ' ...' )

    img = cv2.imread(dir_input + '/' + fn)
    try:
        img.shape
        print(img.shape)    
    except:
        print('image can not be read')
        continue 
 
    if img.shape[2] > 3:
        img = img[:,:,0:3]    

    face_lm_list = fr.face_landmarks(img)
    print('%d face found' % (len(face_lm_list)))
    if len(face_lm_list) == 0:
        print('no face detected')
        continue

    H,W,C = img.shape
    for idx, face_lm in enumerate(face_lm_list):     
        img_this = _crop_face_o(img, face_lm, scale=2.5, adjust=0.2)
        img_this = _padding(img_this)
        
        h,w,c = img_this.shape
        
        face_lm_list_this = fr.face_landmarks(img_this)
        if len(face_lm_list_this) == 0:
            print('face #%d detected but not cropped 1' % (idx+1))
            continue
        face_lm_this = face_lm_list_this[0]

        left_eye_center = _find_center_pt(face_lm_this['left_eye'])
        right_eye_center = _find_center_pt(face_lm_this['right_eye'])
        nose_center = _find_center_pt(face_lm_this['nose_tip'])
        trotate = _get_rotation_matrix(left_eye_center, right_eye_center, nose_center)
        img_this_warped = cv2.warpAffine(img_this, trotate, (h,w)) 
        
        
        face_lm_list_this = fr.face_landmarks(img_this_warped)
        if len(face_lm_list_this) == 0:
            print('face #%d detected but not cropped 2' % (idx+1))
            continue
        face_lm_this = face_lm_list_this[0]
        img_this_warped_cropped = _crop_face_o(img_this_warped, face_lm_this, scale=1.65, adjust=0.05)
                
        hh,ww,cc = img_this_warped_cropped.shape
        if hh != ww:
            print('Cropped image is not a square')
            continue
        # if ww < 200:
        #     print('face too small')
        #     continue
        # if ww < 256:
        #     img_this_warped_cropped = cv2.resize(img_this_warped_cropped,(256,256))

        if ww >= 512:
            img_this_warped_cropped = cv2.resize(img_this_warped_cropped,(512,512))
        else:
            print('face too small')
            continue
        imvar = cv2.Laplacian(img_this_warped_cropped[int(hh/3):int(2*hh/3),int(ww/3):int(2*ww/3),:], cv2.CV_64F).var()
        if imvar < 20:
            print('face too blur')
            continue
        imvar = np.mean(img_this_warped_cropped)
        # mask = np.zeros((hh,ww),np.uint8)
        # bgdModel = np.zeros((1,65),np.float64)
        # fgdModel = np.zeros((1,65),np.float64)
        # mask, bgdModel, fgdModel = cv2.grabCut( img_this_warped_cropped, mask, (2,1,ww-2,hh), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        # img_this_warped_cropped = img_this_warped_cropped*mask2[:,:,np.newaxis]

        cv2.imwrite(dir_faces + '%.5f.jpg' % imvar, img_this_warped_cropped);
        print('face #%d cropped' % (idx+1))
