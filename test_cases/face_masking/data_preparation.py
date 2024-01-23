import pandas as pd
import numpy as np
import os
import random
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import cv2
from test_cases.face_masking.helper import getJSON

annotations = r"test_cases\face_masking\annotations"
images = r"test_cases\face_masking\images"
train_csv_path = r"test_cases\face_masking\train.csv"


def load_and_preprocess_data():

	# Attempt to load the train.csv file
	try:
		df = pd.read_csv(train_csv_path)
	except FileNotFoundError:
		print("train.csv not found in the current directory.")
		return None
	
	train = []
	img_size = 124
	mask = ['face_with_mask']
	non_mask = ["face_no_mask"]
	labels = {'mask': 0, 'without mask': 1}
	
	for i in df["name"].unique():
		f = i + ".json"
		for j in getJSON(os.path.join(annotations, f)).get("Annotations"):
			x, y, w, h = map(int, j["BoundingBox"])  # Convert to int in case they are float
			img = cv2.imread(os.path.join(images, i), 1)
			
			# Ensure that the img is not None
			if img is not None:
				# Correct the slicing here
				img_cropped = img[y:y + h, x:x + w]
				
				# Check if the cropped image is empty
				if img_cropped.size != 0:
					img_resized = cv2.resize(img_cropped, (img_size, img_size))
					if j["classname"] in mask:
						train.append([img_resized, labels["mask"]])
					elif j["classname"] in non_mask:
						train.append([img_resized, labels["without mask"]])
				else:
					print(f"Empty crop detected for image: {i}")
			else:
				print(f"Image not found or cannot be opened: {i}")
	
	# Shuffle the dataset
	random.shuffle(train)
	
	return train


def prepare_data(train):
	X = []
	Y = []
	for features, label in train:
		X.append(features)
		Y.append(label)
	
	X = np.array(X) / 255.0
	X = X.reshape(-1, 124, 124, 3)
	Y = np.array(Y)
	return X, Y


def generate_tensor_data(xtrain):
	tensordata = ImageDataGenerator(
		featurewise_center=False,
		samplewise_center=False,
		featurewise_std_normalization=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		rotation_range=15,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True,
		vertical_flip=False)
	tensordata.fit(xtrain)
	return tensordata
