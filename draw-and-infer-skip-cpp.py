import sys
import time

import cv2
import numpy as np

from myLayers import conv2d

from openvino.inference_engine import IENetwork, IECore

win_name = 'Draw and Infer'
ratio = 20
size = 28*ratio
caption_size = ratio * 4
frame = np.zeros((size, size, 3), dtype=np.uint8)

last_infer_time = 0

def onMouse(event, x, y, flags, param):
	global frame, last_infer_time
	pen_img = np.zeros(frame.shape, dtype=np.uint8)
	cv2.circle(pen_img, (x,y), ratio, (255,255,255), -1)
	if event==cv2.EVENT_MOUSEMOVE:							# Mouse move event 
		if flags and cv2.EVENT_FLAG_LBUTTON:				# Left button is pressing down
			frame |= pen_img								# Draw a filled circle
	elif event==cv2.EVENT_RBUTTONDOWN:						# Right button down event
		frame = np.zeros((size, size, 3), dtype=np.uint8)	# Frame clear
	tmpimg = frame | pen_img
	cv2.imshow(win_name, tmpimg)

def main():
	global frame
	global last_infer_time

	model1 = './models/mnist_skip_1'
	model2 = './models/mnist_skip_2'

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

	cv2.namedWindow(win_name)
	cv2.setMouseCallback(win_name, onMouse, param=None)
	cv2.imshow(win_name, frame)

	weights, bias = np.load('weights.npy', allow_pickle=True) 					# [(3,3,64,64),(64,)]  (fy,fx,fc,fn), (fn)

	while(cv2.waitKey(100)!=27):   # 27==ESC key
		# Image preprocess - shrink and convert to single channel image (monochrome)
		shrank_img = cv2.resize(frame, (28, 28))   # 28x28
		input_img, _, _ = cv2.split(shrank_img)

		stime = time.time()
		# Cascade 2 models
		result1 = exec_net1.infer(inputs={input_name1: input_img})
		result = conv2d(result1[output_name1], weights, bias)		# Do Conv2D+BiasAdd+Relu on behalf of the excluded 'target_conv_layer'
		result2 = exec_net2.infer(inputs={input_name2: result})
		etime = time.time()
		last_infer_time = etime - stime
		result = result2[output_name2][0]

		# Draw inference score bar chart
		caption = np.zeros((caption_size, size, 3), dtype=np.uint8)
		for i in range(10):
			prob = result[i]
			x1 = int((i+1)*(size/12))
			y1 = int(caption_size-prob*caption_size)
			x2 = int((i+1.5)*(size/12))
			y2 = int(caption_size)
			cv2.rectangle(caption, (x1, y1), (x2, y2), (255,0,0), -1)
			cv2.putText(caption, str(i), (x1, caption_size-ratio), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
		caption[:28,:28,:]=shrank_img
		cv2.putText(caption, '{:6.3f}ms'.format(last_infer_time*1000), (30, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,128), 1)
		cv2.imshow('score', caption)

if __name__ == '__main__':
	main()
