import pprint

from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, utils

import numpy as np

def main():
	# Prepare datasets (using keras.datasets.mnist)
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape(-1, 28, 28, 1)
	test_images = test_images.reshape(-1, 28, 28, 1)
	train_labels = train_labels.astype(np.float32)
	test_labels = test_labels.astype(np.float32)

	'''
	# Inspect integrity of training images and labels
	for i in range(20):
		img = train_images[i]
		print(train_labels[i])
		cv2.imshow('a', img)
		if cv2.waitKey(0) == 27:
			sys.exit(0)
	'''

	# Normalize dataset (data range = 0.0-1.0)
	train_images = train_images.astype(np.float32) / 255.0
	test_images  = test_images.astype(np.float32) / 255.0

	# Convert the class label to one-hot vector   e.g. 3->[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	train_labels_v = utils.to_categorical(train_labels, 10)
	test_labels_v  = utils.to_categorical(test_labels , 10)

	# Build MNIST CNN model
	model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='relu', name='target_conv_layer'),
		layers.Flatten(),
		layers.Dense(64, activation='relu'),
		layers.Dense(10, activation='softmax')
	])

	# Compiling the model
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
	model.summary()

	# Training the model
	print('\n*** Start training...')
	model.fit(train_images, train_labels_v, epochs=5)

	tgt_layer = model.get_layer('target_conv_layer')
	weights = tgt_layer.get_weights()        # Obtain weights and bias
	np.save('weights.npy', weights)
	print('\n*** Weight and bias data of "target_conv_layer" saved')

	# Validate the model
	print('\n*** Start validation...')
	test_loss, test_acc = model.evaluate(test_images, test_labels_v)
	print('\nTest accuracy:', test_acc)

	# Saving entire model data (model+weight) in TF SavedModel format
	model.save('mnist-savedmodel', save_format='tf')
	print('*** TF SavedModel saved')

if __name__ == '__main__':
	main()
	print('\n\n*** Training completed.\n\n')
