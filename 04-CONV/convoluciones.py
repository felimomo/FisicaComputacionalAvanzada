from itertools import product
import numpy as np
from scipy.signal import convolve2d
from PIL import Image

N = 128
m1 = 50
dm = 25
img1 = np.zeros((N,N))
img1[m1:m1+dm, m1:m1+dm] = 150 * np.ones((dm,dm))

Image.fromarray(
	img1.astype(np.uint8), 
	mode='L'
).save('images/original.png')

n = 32

kernel_lib = {
	'horiz': np.array([
		[1, 1, 1],
		[0, 0, 0],
		[-1, -1, -1]
	]),
	'vert': np.array([
		[1, 0, -1],
		[1, 0, -1],
		[1, 0, -1]
	]),
	'L': np.array([
		[1, 0, 0],
		[-1, 0, 0],
		[1, -1, 1]
	]),
	'sq': np.array([
		[1, 1, 1],
		[1, 1, 1],
		[1, 1, 1]
	]),
	'diag1': np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
	]),
	'diag2': np.array([
		[0, 0, 1],
		[0, 1, 0],
		[1, 0, 0]
	]),
	'lor1': np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, -1]
	]),
	'lor2': np.array([
		[0, 0, 1],
		[0, 1, 0],
		[-1, 0, 0]
	]),
	'grad': np.array([
		[m / n for m in range(n)]
		for _ in range(n)
	]),
}

convolutions = {}
for name, kernel in kernel_lib.items():
	convolutions[name] = convolve2d(
		img1, 
		kernel, 
		mode="valid" # sin padding
	)
	convolutions[name] = np.clip(
		convolutions[name], 0 , 255
	).astype(np.uint8)

# print([np.unique(c) for name, c in convolutions.items()])

conv_images = {
	name: Image.fromarray(conv_arr, mode='L')
	for name, conv_arr in convolutions.items()
}

for name, img in conv_images.items():
	img.save(f'images/out/{name}-conv.png')