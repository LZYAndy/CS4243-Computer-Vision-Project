import numpy as np
import cv2
import time

# use Haar Cascade model to find waldo
def evaluate_waldo(img_num):
    result = []
    
    # import two cascade models for waldo
    waldo_cascade_rec = cv2.CascadeClassifier('datasets/Templates/haar_templates/data-1-2-bigger/cascade.xml')
    waldo_cascade_square = cv2.CascadeClassifier('datasets/Templates/haar_templates/data-square-bigger/cascade.xml')
    
    # read in test image
    img = cv2.imread('datasets/JPEGImages/' + img_num + '.jpg')
    # change the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use the cascade models to detect waldo objects
    waldo_rec = waldo_cascade_rec.detectMultiScale3(gray, 2, 2, cv2.CASCADE_SCALE_IMAGE, (50, 80), outputRejectLevels=True)
    waldo_square = waldo_cascade_square.detectMultiScale3(gray, 1.5, 3, cv2.CASCADE_SCALE_IMAGE, (50, 50), outputRejectLevels=True)

    temp = ""
    # record the locations for bounding boxes found
    for i, (x,y,w,h) in (enumerate(waldo_rec[0])):
            temp = img_num + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str(y+h)
            result.append(temp)

    # record the locations for bounding boxes found
    for i, (x,y,w,h) in (enumerate(waldo_square[0])):
            temp = img_num + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str(y+h)
            result.append(temp)

    return result

# use Haar Cascade model to find wenda
def evaluate_wenda(img_num):
    result = []
    
    # import two cascade models for wenda
    wenda_cascade_rec = cv2.CascadeClassifier('datasets/Templates/haar_templates/data-wenda-1-2/cascade.xml')
    wenda_cascade_square = cv2.CascadeClassifier('datasets/Templates/haar_templates/data-wenda-square/cascade.xml')
    
    # read in test image
    img = cv2.imread('datasets/JPEGImages/' + img_num + '.jpg')
    # change the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use the cascade models to detect wenda objects
    wenda_rec = wenda_cascade_rec.detectMultiScale3(gray, 1.1, 4, cv2.CASCADE_SCALE_IMAGE, (50, 80), outputRejectLevels=True)
    wenda_square = wenda_cascade_square.detectMultiScale3(gray, 2, 2, cv2.CASCADE_SCALE_IMAGE, (50, 50), outputRejectLevels=True)
 
    temp = ""
    # record the locations for bounding boxes found
    for i, (x,y,w,h) in (enumerate(wenda_rec[0])):
            temp = img_num + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str(y+h)
            result.append(temp)        
    
    # record the locations for bounding boxes found
    for i, (x,y,w,h) in (enumerate(wenda_square[0])):
            temp = img_num + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str(y+h) 
            result.append(temp)

    return result

# use Haar Cascade model to find wizard
def evaluate_wizard(img_num):
    result = []
    
    # import two cascade models for wenda
    wizard_cascade_rec = cv2.CascadeClassifier('datasets/Templates/haar_templates/data-wizard-1-2/cascade.xml')
    
    # read in test image
    img = cv2.imread('datasets/JPEGImages/' + img_num + '.jpg')
    # change the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use the cascade models to detect wizard objects
    wizard_rec = wizard_cascade_rec.detectMultiScale3(gray, 1.1, 3, cv2.CASCADE_SCALE_IMAGE, (50, 80), outputRejectLevels=True)
    
    temp = ""
    # record the locations for bounding boxes found
    for i, (x,y,w,h) in (enumerate(wizard_rec[0])):
            temp = img_num + " " + str(x) + " " + str(y) + " " + str(x+w) + " " + str(y+h)
            result.append(temp)
            
    return result
