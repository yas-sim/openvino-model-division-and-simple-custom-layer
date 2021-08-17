import sys
import time

import numpy as np
from openvino.inference_engine import IECore
from tensorflow.keras.datasets import mnist

model = './models/mnist'

def main():
	ie = IECore()
	net = ie.read_network(model=model+'.xml', weights=model+'.bin')
	input_name    = next(iter(net.input_info))
	output_name   = next(iter(net.outputs))
	print('Input node name=', input_name, ' Output node name=', output_name)
	batch, c, h, w = net.input_info[input_name].tensor_desc.dims
	print('Input shape = ', net.input_info[input_name].tensor_desc.dims)
	exec_net = ie.load_network(network=net, device_name='CPU', num_requests=1)

	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape(-1, 1, 28, 28, 1)
	test_images = test_images.reshape(-1, 1, 28, 28, 1)

	right=0
	num=0
	stime = time.time()
	for label, img in zip(test_labels, test_images):
		img = img.astype(np.float32).reshape(1, 1, 28, 28)
		img /= 255.0
		result = exec_net.infer(inputs={input_name: img})
		correct=label                                  # correct answer (label)
		infered=np.argmax(result[output_name])         # infered answer
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
