import argparse

import numpy as np
from scipy.integrate import solve_ivp
import scipy.fft
from scipy.fft import fft2 as FFT
from scipy.fft import ifft2 as IFFT

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

##
# NOTA IMPORTANTE
##
# Cambios respecto a la 1D: la rfft2 produce una
# matriz no cuadrada por eficiencia. Esto hace todo
# un poco más confuso y esa eficiencia no la
# necesitamos a este nivel (no necesitamos demasiadas
# frecuencias en neustra expansión).
#
# Por esto decidí usar la fft2 en cambio, esta es la
# transformada compleja. Para contrarrestar el hecho
# de que aparecen componentes complejas de error,
# tomo la parte real en todos los pasos que involucren
# una fft.
##

parser = argparse.ArgumentParser()
parser.add_argument(
	'-hm', '--homogenea', 
	action='store_true', default=False, 
	help='Incluir este argumento si la ec es homogenea.'
)
args = parser.parse_args()

# versión de la ecuación
HOMOGENEA = args.homogenea

# discretización
N = (2 ** 7)

# condiciones iniciales
T = 10
N_t = 50
t_eval = [(T / N_t) * t for t in range(N_t+1)]
X = np.linspace(0, 2 * np.pi, N)
xx, yy = np.meshgrid(X, X)
k = scipy.fft.fftfreq(N, 1./N)
kx, ky = np.meshgrid(k,k)


# f0 = FFT(np.sin(X))
f0_unhat = np.exp(-(xx-np.pi) ** 2 - (yy-np.pi) ** 2)
f0 = FFT(f0_unhat)
natural_shape = f0.shape
print(f'k shape = {k.shape}')
print(f'kx shape = {kx.shape}')
print(f'ky shape = {ky.shape}')
print(f'xx shape = {xx.shape}')
print(f'yy shape = {yy.shape}')
print(f'f0 hat shape = {natural_shape}')
print(f'f0 shape = {f0_unhat.shape}')
f0 = f0.flatten()
VISC = 0.02

def G(t, u: np.ndarray, homogenea: bool = False):
	u = u.reshape(natural_shape)
	if homogenea:
		sk=0
	else:
		st = 2 * np.array([
			np.cos(2 * np.pi * t / 20),
			np.sin(2 * np.pi * t / 20),
		])
		sk = FFT(
			np.exp(
				- 10 * (xx-st[0]) ** 2 
				- 10 * (yy-st[1]) ** 2
			)
		)

	u = np.real(u)
	sk=np.real(sk)

	r = (
		- FFT(
			IFFT(u) * IFFT(1j * (kx * u + ky * u))
		)
		- VISC * (kx ** 2 + ky ** 2) * u
		+ sk
	).flatten()
	r=np.real(r)
	print(f't={t:.2f}, | sk | / | r | = {np.linalg.norm(sk)/np.linalg.norm(r):.2f}', end='\r')
	return r

def main():
	print('here')
	fourier_result = solve_ivp(
		fun = G, 
		t_span = [t_eval[0],t_eval[-1]], 
		y0=f0, 
		t_eval=t_eval,
		args=[HOMOGENEA]
	)
	# print('\nfourier result shape')
	# print(fourier_result.y.shape)

	f = np.array(
		[
			IFFT(fourier_f.reshape(natural_shape)).flatten()
			for fourier_f in fourier_result.y.T
		]
	)

	print('f shape:')
	print(f.shape)

	real_f = np.real(f).reshape(len(t_eval), N, N)
	zmin, zmax = real_f.min(), real_f.max()

	fig = px.imshow(
		real_f,
		animation_frame=0,
		color_continuous_scale='viridis',
		origin='lower',
		labels=dict(animation_frame='t'),
		zmin=zmin,
		zmax=zmax,
	)
	for step, t in zip(fig.layout.sliders[0].steps, t_eval):
		step['label'] = f'{t:.2f}'
	if HOMOGENEA:
		filename = 'homoge_peli.html'
	else:
		filename = 'fuente_peli.html'
	fig.write_html(filename)
	print(f'Animacion guardada en: {filename}')

	# for i,t in enumerate(t_eval):
	# 	df = pd.DataFrame({
	# 		'x': xx.flatten(),
	# 		'y': yy.flatten(),
	# 		'f(t)': np.real(f[i])
	# 	}).pivot(index='y', columns='x', values='f(t)')
	# 	# print(df.head())
	# 	ax = sns.heatmap(df, cmap='viridis')
	# 	ax.set_xlabel('')
	# 	ax.set_ylabel('')
	# 	plt.tight_layout()
	# 	plt.savefig(f'f(t={t:.2f}).png')
	# 	plt.clf()

	# if HOMOGENEA:
	# 	plt.savefig('new_homogenea.png')
	# else:
	# 	plt.savefig('new_fuente.png')


if __name__=='__main__':
	main()
