import keras
import numpy as np
import os
import cv2
import open3d as opd
from Network.facedetect import find_faces

def point_cloud_normalization(pc):
	center = np.mean(pc, axis=0)
	pc -= center
	furthest = np.max(np.sqrt(np.sum(abs(pc)**2, axis=1)))
	pc /= furthest

	return pc

def origin_translation(pc):
	p = np.array([np.min(pc[:, i]) for i in [0, 1, 2]])
	pc -= p
	return pc

class LSTMDataset_V3(keras.utils.Sequence):
	def __init__(self, path, images=5, batch_size=1, norm=True, N=27508):
		self.N = N
		self.func = point_cloud_normalization if norm else origin_translation
		self.path = path
		self.batch_size = batch_size
		self.n_img = images
		self.__find_images__()
		self.on_epoch_end()

	# Initial method for loading data. Load one and one batch to memory. Slow training.
	def __len__(self):
		return int(np.floor(len(self.indicies)/self.batch_size))

	def __getitem__(self, index):
		X, y = self.__data_generation(index)
		return X, y

	# Finds all jpgs as X and single ply file as ground truth.
	# All different faces in its own sunfolder.
	# Subfolder for subject x contains all images of subject x
	# as well as a single ply file used as ground truth.
	
	def __find_images__(self):
		print("Loading dataset.")
		self.item_pairs = []
		for folder in os.listdir(self.path):
			images = [
				'/'.join([self.path, folder, file]) for file 
				in os.listdir(self.path+'/'+folder) if file.endswith('.jpg')
				]

			# Finds the ground truth .ply file
			ground_truth_file = [
				'/'.join([self.path, folder, file]) for file 
				in os.listdir('/'.join([self.path, folder])) if file.endswith('.ply')
				]
			
			gt = self.func(np.array(opd.io.read_point_cloud(*ground_truth_file).points))
			
			self.item_pairs.append((images, gt))

	def on_epoch_end(self):
		inputs = []
		gts = []

		for Xs, Y in self.item_pairs:
			Y = np.expand_dims(Y, axis=0)
		# For each subject, shuffle all images of subject and create batches of size n_image
			for _ in range(5):
				np.random.shuffle(Xs)
			batches = [
				Xs[i:i+self.n_img] for i in range(0, len(Xs), self.n_img) 
				if len(Xs[i:i+self.n_img]) == self.n_img
				]
			
			for batch in batches:
				inputs.append(batch)
				gts.append(Y)

		self.inputs = inputs
		self.truths = gts

		indicies = [i for i in range(len(inputs))]
		for _ in range(np.random.randint(3, 10)):
			np.random.shuffle(indicies)

		self.indicies = indicies

	def __data_generation(self, index):
		X = np.zeros((self.batch_size, self.n_img, 256, 256, 3), dtype=np.float32)
		Y = np.zeros((self.batch_size, self.N, 3))
		for batch in range(self.batch_size):
			images = self.inputs[self.indicies[index + batch]]		# Images
			Y[batch] = self.truths[self.indicies[index + batch]]	# Ground truth point cloud
			
			for i, entry in enumerate(images):
				image = cv2.imread(entry).astype(np.float32)
				cv2.normalize(image, X[batch, i], 0.0, 1.0, cv2.NORM_MINMAX)
		return X, Y
