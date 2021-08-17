import sys
import time

import numpy as np
from openvino.inference_engine import IECore
from tensorflow.keras.datasets import mnist

model1 = './models/mnist_skip_1'
model2 = './models/mnist_skip_2'
#model2 = './models/mnist_div_2' # This model should work fine as the 2nd model ('mnist_skip_2' and 'mnist_div_2' are the identical model)


# Naive implementation of Conv2D + BiasAdd + ReLU -> VERY VERY VERY SLOW but works
# Param : strides=(1,1), padding='valid', kernel_size=(3,3), filters=64
def myConv2d(input, weights, bias):
	n,c,h,w     = input.shape   # (1,64,5,5)
	kernel_size = 3
	filters     = 64
	strides     = 1

	output = np.zeros((n,filters,h-kernel_size+1, w-kernel_size+1), np.float32)	# [1,64,3,3]

	for fc in range(filters):
		for dy in range(0, h-kernel_size+1, strides):
			for dx in range(0, w-kernel_size+1, strides):
				cnv = 0
				for cc in range(c):
					for fy in range(kernel_size):
						for fx in range(kernel_size):
							flt = weights[fy, fx, cc, fc]		# fx and fy are swapped (matmul)
							dt  = input[0, cc, dy+fy, dx+fx]
							cnv += flt * dt						# Convolution
				output[0, fc, dy, dx] = cnv + bias[fc]			# Bias addition
	output = np.where(output<0, 0, output)						# ReLU
	return output


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
		result = myConv2d(result1[output_name1], weights, bias)		# Do Conv2D+BiasAdd+Relu on behalf of the excluded 'target_conv_layer'
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
