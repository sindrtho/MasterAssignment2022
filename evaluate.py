from sys import argv
import tensorflow as tf
import numpy as np

from Datagenerator.dataset import LSTMDataset_V3

import keras as k
from keras.layers import Input
from keras.models import Model
from Network.mobile3model import get_keras_model, relu6
import tensorflow as tf
import keras
import tqdm
from datetime import datetime as t
import time

from Evaluation.evaluate_predictions import prediction_evaluater, apply_homogenous_tform
from Evaluation.icp import icp

if __name__=="__main__":
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
	config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

	if len(argv) < 5:
		print("please run script with parameters: batch_size, n_images, data_dir, model_weights, [result_file]")
		exit(1)

	BATCH_SIZE = int(argv[1])
	N_IMAGES = int(argv[2])
	PATH = argv[3]
	MODEL = argv[4]
	
	
	filename = None
	if len(argv) > 5:
		filename = argv[5]
		print(filename)
		if filename:
			print("YES FILENAME!")

	model = k.models.load_model(MODEL, custom_objects={'relu6':relu6})
	model.compile()

	loss = keras.losses.MeanSquaredError()
	evaluation_metric = prediction_evaluater()

	testing_generator = LSTMDataset_V3(PATH, images=N_IMAGES, batch_size=BATCH_SIZE, N=2**14)

	total_loss, metric = 0.0, 0.0

	for index in tqdm.trange(0, len(testing_generator)):
		X, Y = testing_generator.__getitem__(index)
		logits = model(X, training=False)
		loss_value = loss(Y, logits)

		total_loss += loss_value

		metric += evaluation_metric(np.array(logits[0]), np.array(Y[0]))

	total_loss /= len(testing_generator)
	metric /= len(testing_generator)

	print(f"Average evaluation loss: {float(total_loss)}\nAverage NME: {metric}\n\n")
	
	if filename:
		with open(filename, 'w') as file:
			file.write(f'Average loss: {float(total_loss)}\nAverage NME: {metric}')
