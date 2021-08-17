import numpy as np

# Naive implementation of Conv2D + BiasAdd + ReLU -> VERY VERY VERY SLOW but works
# Param : strides=(1,1), padding='valid', kernel_size=(3,3), filters=64
def conv2d(input, weights, bias):
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
