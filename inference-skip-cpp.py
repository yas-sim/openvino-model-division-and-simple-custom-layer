import sys
import time

import numpy as np
from openvino.inference_engine import IECore
from tensorflow.keras.datasets import mnist

#from myLayers_py import conv2d		# Python version
from myLayers import conv2d		# C++ version - You need to build myLayer.[so|pyd] first

model1 = './models/mnist_skip_1'
model2 = './models/mnist_skip_2'

def main():
	ie = IECore()

	net1 = ie.read_network(model=model1+'.xml', weights=model1+'.bin')
	input_name1    = next(iter(net1.input_info))
	output_name1   = next(iter(net1.outputs))
	print('Input node name1=', input_name1, ' Output node name1=', output_name1)
	input_shape1 = net1.input_info[input_name1].tensor_desc.dims
	print('Input shape1 = ', input_shape1)
	exec_net1 = ie.load_network(network=net1, device_name='CPU', num_requests=1)
	b, c, h, w = input_shape1

	net2 = ie.read_network(model=model2+'.xml', weights=model2+'.bin')
	input_name2    = next(iter(net2.input_info))
	output_name2   = next(iter(net2.outputs))
	print('Input node name2=', input_name2, ' Output node name2=', output_name2)
	input_shape2 = net2.input_info[input_name2].tensor_desc.dims
	print('Input shape2 = ', input_shape2)
	exec_net2 = ie.load_network(network=net2, device_name='CPU', num_requests=1)

	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape(-1, 1, 28, 28, 1)
	test_images = test_images.reshape(-1, 1, 28, 28, 1)

	weights, bias = np.load('weights.npy', allow_pickle=True) 					# [(3,3,64,64),(64,)]  (fx,fy,fc,fn), (fn)

	right=0
	num=0
	stime = time.time()
	for label, img in zip(test_labels, test_images):
		img = img.astype(np.float32).reshape(1, 1, 28, 28)
		img /= 255.0
		# Cascade 2 models
		result1 = exec_net1.infer(inputs={input_name1 : img})		# 1st half of the original model ('target_conv_layer' is excluded)
		result = conv2d(result1[output_name1], weights, bias)		# Do Conv2D+BiasAdd+Relu on behalf of the excluded 'target_conv_layer'
		result2 = exec_net2.infer(inputs={input_name2 : result})	# 2nd half of the original model

		correct=label												# correct answer (label)
		infered=np.argmax(result2[output_name2])					# infered answer

		if correct == infered:
			print('.', end='', flush=True)
			right+=1
		else:
			print('X', end='', flush=True)
		if num % 50==49:
			print()
		num+=1
	print('{} / {} : {} %'.format(right, num, (right/num)*100))
	etime = time.time()
	print('Inference time =', etime-stime, "sec")
	return 0

if __name__ == '__main__':
	sys.exit(main())
