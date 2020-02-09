import os
import cv2
import math
import pickle
import numpy as np
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog


class SVM:
	def __init__(self, target='waldo', ortn=10, ppc=6):
		#hog parameters
		self.ortn = ortn	 # orientations
		self.ppc = ppc  # pixels per cell
		self.x_window = 26
		self.y_window = 50
		self.target = target
		try :
			self.model = pickle.load(open('svm_' + target + '.sav', 'rb'))
		except FileNotFoundError:
			print(target + " svm not found. Please train the model using train()")
			self.model = None

	def predict(self, patch):
		assert self.model is not None, "Model not found!!"
		
		# Resize image patch since they can come in varying sizes
		patch = cv2.resize(patch, (self.x_window, self.y_window))

		# Generate the HOG feature vector for this image patch
		features = hog(patch, orientations=self.ortn, pixels_per_cell=(self.ppc,self.ppc), cells_per_block=(2,2))
		features = features.flatten()

		# Generate confidence score using trained SVM
		res = self.model.predict_proba(features.reshape(1, -1))
		return res

	def train(self):
		# Grab all templates
		dir_name = 'datasets/Templates/svm_templates/'
		templates = []
		train_labels = []
		train_features = []
		template_dir = os.listdir(dir_name)

		# Grab SVM training template data
		for folder_name in template_dir:
		    folder_dir = os.listdir(dir_name + folder_name)
		    for file_name in folder_dir:
		        templates.append(dir_name + folder_name + '/' + file_name)

		for file in templates:
			print("Reading file " + file)
			
			# Read in template and resize it since all templates might not be the same size
			img = cv2.imread(file)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized_gray = cv2.resize(gray, (self.x_window, self.y_window))

			# Generate the HOG feature vector for the template
			features = hog(resized_gray, orientations=self.ortn, pixels_per_cell=(self.ppc, self.ppc), cells_per_block=(2,2))
			train_features.append(features.flatten())
			if file[33:33+len(self.target)] == self.target:
				label = self.target
			else:
				label = 'not_' + self.target
			if label == self.target:
				train_labels.append(label)
			else:
				train_labels.append('not_' + self.target)

		train_features = np.asarray(train_features)

		# Train the SVM model using the generated labels and corresponding HOG feature vectors
		# Model will be saved as svm_<target>.sav
		model = SVC(gamma='scale', decision_function_shape='ovo', probability=True)
		model.fit(train_features, train_labels)
		pickle.dump(model, open('svm_' + self.target + '.sav', 'wb'))
		print("saved model to " + 'svm_' + self.target + '.sav')
