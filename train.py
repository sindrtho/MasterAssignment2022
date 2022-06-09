import tensorflow as tf
import numpy as np

from Datagenerator.dataset import LSTMDataset_new, LSTMDataset_V3, LSTMDataset

from keras.layers import Input
from keras.models import Model
from Network.mobile3model import get_keras_model
import tensorflow as tf
import keras
import tqdm
from datetime import datetime as t
import time

from Evaluation.evaluate_predictions import prediction_evaluater, apply_homogenous_tform
from Evaluation.icp import icp

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 5
N_IMAGES = 8
SHAPE = (N_IMAGES, 256, 256, 3)

training_generator = LSTMDataset_V3('Dataset/BU4DFE/Downsampled/Training', images=N_IMAGES, batch_size=BATCH_SIZE, N=2**14)
testing_generator = LSTMDataset_V3('Dataset/BU4DFE/Downsampled/Testing', images=N_IMAGES, batch_size=BATCH_SIZE, N=2**14)

print("Initializing model")

SQUEEZE_AND_EXCITE = True

model = get_keras_model(SHAPE, N=2**14, squeeze=SQUEEZE_AND_EXCITE)

print("Model initialized")

EPOCHS = 300

lr = 0.0001

opt = keras.optimizers.Adam(learning_rate=lr)
loss = keras.losses.MeanSquaredError()

evaluation_metric = prediction_evaluater()

best_loss = float('inf')
best_train = float('inf')
savepath = 'model.h5'
counter = 0
THRESHOLD = 30

for epoch in range(EPOCHS):
	total_training_loss = 0.0
	training_NME = 0.0

	start = time.time()
	print("Start epoch %d at %s" % (epoch+1, t.now().strftime('%H:%M:%S')))
	for index in tqdm.trange(0, len(training_generator)):
		X, Y = training_generator.__getitem__(index)
	
		with tf.GradientTape() as tape:
			logits = model(X, training=True)

			loss_value = loss(logits, Y)
			total_training_loss += float(loss_value)
		grads = tape.gradient(loss_value, model.trainable_weights)
		opt.apply_gradients(zip(grads, model.trainable_weights))
		
		print(f"Training step {index+1}:\n\tLoss: {float(loss_value)}\n")

	stop = time.time()
	d = stop-start
	total_training_loss /= len(training_generator)
	print(f"Finished epoch {epoch+1}\nFinal loss: {float(total_training_loss)}\nFinished at {t.now().strftime('%H:%M:%S')}")
	
	training_generator.on_epoch_end()


	##### Validating model on validation set #####

	print("Validating:\n")
	total_validation_loss = 0
	# metric = 0.0

	for index in tqdm.trange(0, len(testing_generator)):
		X, Y = testing_generator.__getitem__(index)
		logits = model(X, training=False)
		loss_value = loss(Y, logits)

		total_validation_loss += loss_value

		# if epoch != 0 and epoch % 10 == 0:
		# 	metric += evaluation_metric(np.array(logits[0]), np.array(Y[0]))

	# if epoch != 0 and epoch % 10 == 0:
	# 	metric /= len(testing_generator)

	total_validation_loss /= len(testing_generator)
	
	print(f"Average validation loss epoch {epoch+1}: {float(total_validation_loss)}")

	testing_generator.on_epoch_end()
	##### Writing information to file #####

	with open(f'results.txt', 'a') as file:
		file.write(f'Epoch {epoch+1}\n')
		file.write(f'Finished after {d} seconds\n')
		file.write(f'Average training loss: {total_training_loss}\n')
		file.write(f'Average validation loss: {total_validation_loss}\n\n')
		# file.write(f'Average validation NME: {float(metric)}\n\n' if epoch != 0 and epoch % 10 == 0 else '\n')

	
	if epoch > 0 and epoch % 10 == 0:
		print(f"Halving learning rate from {lr} to {lr/2.0}")
		lr /= 2.0
		opt.learning_rate = lr
		
	if total_validation_loss < best_loss:
		print(f'Loss improved from {best_loss} to {total_validation_loss}. Saving model to {savepath}')
		best_loss = total_validation_loss
		model.save(savepath)
		counter = 0
	elif total_training_loss < best_train:
		print("Training loss improved. Saving train model")
		counter = 0
		best_train = total_training_loss
		model.save("trainmodel.h5")
	else:	# Early stopping if loss does not improve in 5 epochs
		counter += 1
		if counter >= THRESHOLD:
			print(f"Early stopping at epoch {epoch+1}. Loss not improved in {THRESHOLD} epochs.")
			exit(0)
	

