#add pycu to path
import pathlib, sys
path = pathlib.Path(__file__)
path = path.parent.absolute() #.parent
sys.path.append(str(path))

import pycu
import os
import time
import numpy as np

def compile_kernels():
	global surface_levelset, count_occupied, quad_size

	file = pycu.utils.open_file("kernel.cu")
	entry = ["root", "dense"]
	nvrtc_options = {}

	#if the C header files are in a location other than where the main.py file is located
	#their directories will need to be included so that the compiler (NVRTC) can find them
	I = {"I":['vector_extensions', 'interval']}

	#macros defined in the kernel.cu file. This allows kernel parameters to be defined/modified
	#within the python file. Each also has a default macro value defined within the kernel
	#file if no value is provided from Python.
	macros = {'define-macro':[f'SURFACE={surface_levelset}',
							  f'COUNT_OCCUPIED={str(count_occupied).lower()}', #C/C++ boolean values are lowercase
							  f'QUAD_SIZE={quad_size}']}
	nvrtc_options.update(I)
	nvrtc_options.update(macros)

	#libcudadert is the device runtime and must be included to compile code that uses dynamic parallelism
	return pycu.compile_source(file, entry, nvrtc_options = nvrtc_options, libcudadert = True)

def launch_sparse(box, h):
	global root
	def set_pending_kernel_limit():
		#setting the pending kernel launch count to a high value helps prevent
		#the buffer from needing to be resized during the integration process.
		attr = pycu.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
		pycu.ctx_set_limit(attr, 1 << 15)

	set_pending_kernel_limit()

	#quad is multiple entries (which are then summed at the end) to help prevent
	#floating point round-off
	quad = np.zeros(quad_size, dtype = np.float32)
	#tracks the number of leaf nodes run (must be enabled within kernel.cu)
	occupancy = np.zeros(1, dtype = np.uint64)
	#the functions used in the examples are centered on (0,0,0) so an
	#offset is used to shift the grid such that the geometry is centered
	#and not clipped
	offset = np.array(-box/2., dtype = np.float32)

	d_quad = pycu.to_device_buffer(quad)
	d_occupancy = pycu.to_device_buffer(occupancy)
	d_box = pycu.to_device_buffer(box)
	d_offset = pycu.to_device_buffer(offset)
	d_h = h

	blocks, threads = 1,1

	print('sparse')
	start = time.time()
	root<<[blocks, threads]>>(d_quad, d_occupancy, d_box, d_h, d_offset)
	quad = sorted(pycu.to_host(d_quad))
	print('time: ', time.time() - start)
	print('quad: ', np.sum(quad))
	if count_occupied:
		occupancy = pycu.to_host(d_occupancy)
		print('leaf nodes evaluated: ', occupancy[0])
	print()

def launch_dense(box, h):
	def get_kernel_launch_parameters(gpu = 0):
		dev = pycu.device_get(gpu)

		attr = pycu.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
		SMs = pycu.device_get_attribute(attr, dev)
		blocks = 32*SMs

		attr = pycu.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
		threads = pycu.device_get_attribute(attr, dev) #512

		return blocks, threads

	quad = np.zeros(1, dtype = np.float32)
	shape = (np.ceil(box/h - 0.00001) + 1).astype(np.uint32)
	offset = np.array(-box/2., dtype = np.float32)

	d_quad = pycu.to_device_buffer(quad)
	d_shape = pycu.to_device_buffer(shape)
	d_offset = pycu.to_device_buffer(offset)
	d_h = h

	blocks, threads = get_kernel_launch_parameters()

	print('dense')
	start = time.time()
	dense<<[blocks, threads]>>(d_quad, d_shape, d_offset, d_h)
	quad = pycu.to_host(d_quad)
	print('time: ', time.time() - start)
	print('quad: ', quad[0])

surface_levelset = 0
count_occupied = False
quad_size = 10

box = np.array([4.5,4.5,4.5], dtype = np.float32)
shape = 1 << 11
h = np.float32(box[0]/((shape) - 1))

print([shape]*3,'\n')
root, dense = compile_kernels()
# launch_sparse(box, h)
# launch_dense(box, h)
