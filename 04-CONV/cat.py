import numpy as np
from PIL import Image
from scipy.signal import convolve2d

pellas1 = Image.open(
	'images/pellas1.jpg'
).convert('L') # -> to grayscale
# print(np.asarray(pellas1.convert('L')).shape)
pellas1.save('images/pellas1-grayscale.jpg')

pellas2 = Image.open(
	'images/pellas2.jpeg'
).convert('L') 
pellas2.save('images/pellas2-grayscale.jpg')

pellas3 = Image.open(
	'images/pellas3.jpeg'
).convert('L') 
pellas3.save('images/pellas3-grayscale.jpg')

def apply_ker_lib(img: Image, ker_lib: dict):
	img_np = np.asarray(img)
	return {
		name: convolve2d(
			img_np, 
			kernel, 
			mode="valid"
		).clip(0, 255).astype(np.uint8)
		for name, kernel in ker_lib.items()
	}

def save_convs(conv_dict: dict, img_name: str) -> None:
	""" conv_dict = {name : np.arr(np.uint8)} """
	for name, arr in conv_dict.items():
		Image.fromarray(arr).save(
			f"images/out/{name}-{img_name}-conv.jpg"
		)


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
	'gauss': np.array([
		[0,   0.1, 0.2, 0.1, 0],
		[0.1, 0.2, 0.3, 0.2, 0.1],
		[0.2, 0.3, 0.4, 0.3, 0.2],
		[0.1, 0.2, 0.3, 0.2, 0.1],
		[0,   0.1, 0.2, 0.1, 0],
	])
}

conv_pellas1 = apply_ker_lib(
	img = pellas1, 
	ker_lib = kernel_lib
)
save_convs(conv_pellas1, "pellas1")

conv_pellas2 = apply_ker_lib(
	img = pellas2, 
	ker_lib = kernel_lib
)
save_convs(conv_pellas2, "pellas2")

conv_pellas3 = apply_ker_lib(
	img = pellas3, 
	ker_lib = kernel_lib
)
save_convs(conv_pellas3, "pellas3")


