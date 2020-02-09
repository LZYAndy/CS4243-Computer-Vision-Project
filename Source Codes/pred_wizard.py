import os
import cv2
import math
import numpy as np 
from string import digits
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
import xml.etree.ElementTree as ET
import pickle
from haar import evaluate_wizard
from sift_match import feature_matching
from svm import SVM


def detect_wizard(visual=False, img_loc='datasets/ImageSets/val.txt'):
	dir_name = 'templates/'
	bboxes = []
	threshold = 0.2
	open('results/eval/wizard.txt', 'w').close()
	txt = open('results/eval/wizard.txt', 'a')
	svm = SVM('wizard')

	# Run Wizard detection on each target image
	val_im_num = open(img_loc, "r")
	for x in val_im_num:
		res_list = []
		svm_res = []
		bboxes = []

		image_num = x[:3]
		img_name = 'datasets/JPEGImages/' + image_num + '.jpg'
		print("Reading " + img_name)
		img = cv2.imread(img_name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		haar_results = evaluate_wizard(image_num) # returns a list of bbox
		for haar_res in haar_results:
			x1,y1,x2,y2 = haar_res.split()[1:5]
			x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
			patch = gray[y1:y2, x1:x2]
			res = svm.predict(patch)
			svm_res.append(res)
			# Check if the image patch passes the initial threshold
			if res[0,1] > threshold:
				res_list.append(res[0,1])
		
		if len(res_list) > 0:
			# Recalculate threshold
			f_t = min(res_list)+((max(res_list) - min(res_list)) * 0.9)
			for i in range(len(haar_results)):
				x1,y1,x2,y2 = haar_results[i].split()[1:5]
				x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
				if svm_res[i][0,1] > math.floor(f_t * 100)/100.0:
					bboxes.append((x1,y1,x2,y2))
		
		# Removing subsets
		final_bbox = []
		for x1,y1,x2,y2 in bboxes:
			subset = False
			for x11,y11,x22,y22 in bboxes:
				if x1 > x11 and x2 < x22 and y1 > y11 and y2 < y22:
					subset = True
					break
			if not subset:
				final_bbox.append((x1,y1,x2,y2))

		for x1,y1,x2,y2 in final_bbox:
			cv2.rectangle(img, (x1, y1) ,(x2, y2), (0,0,255), 10)
			txt.write(image_num + ' 0.000 ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
		
		cv2.imwrite("results/wizard/result_" + image_num + '.jpg',img)
		if visual:
			cv2.imshow("result", cv2.resize(img, (1600, 900)))
			cv2.waitKey(0)
