import glob
import numpy as np
import cyvlfeat as vlfeat
import cv2

# use SIFT to extract features from image desired
def feature_matching(img1, min_len, target='waldo'):
    max_len = 0
    number_keypoints = 0
    kp2_len = 0
    
    # get a list of paths for sift templates
    persons = glob.glob("datasets/Templates/sift_templates/" + target + "/*")
        
    # extract feaures from img1
    kp1, des1 = vlfeat.sift.dsift(cv2.resize(img1, (50,50)), step=2, fast=True)
    for i, person in enumerate(persons):
        img2 = cv2.imread(person)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # extract features from img2
        kp2, des2 = vlfeat.sift.dsift(cv2.resize(img2, (50,50)), step=2, fast=True)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict()   # or pass empty dictionary
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(np.float32(des1), np.float32(des2), 2)
        good_points = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good_points.append(m)
        
        # get the max number of good points
        if len(good_points) >= 0 and len(good_points) > max_len:
            max_len = len(good_points)
            kp2_len = kp2.shape[0]
    
    # get the number of keypoints
    if kp1.shape[0] >= kp2_len:
        number_keypoints = kp1.shape[0]
    else:
        number_keypoints = kp2_len
    
    return max_len, number_keypoints
