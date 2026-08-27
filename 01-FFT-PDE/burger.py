import argparse

import numpy as np
from scipy.integrate import solve_ivp
import scipy.fft
from scipy.fft import rfft as FFT
from scipy.fft import dst as DST
from scipy.fft import irfft as IFFT

import pandas as pd
import matplotlib.pyplot as plt

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
N = (2 ** 8)

# condiciones iniciales
T = 10
N_t = 10
t_eval = [(T / N_t) * t for t in range(N_t+1)]
X = np.linspace(0, 2 * np.pi, N)
k = scipy.fft.rfftfreq(N, 1./N)

print('Frecuencias k usadas:')
print(k)

# f0 = FFT(np.sin(X))
f0 = FFT(np.exp(-(X-np.pi) ** 2))
VISC = 0.1

def G(t, u: np.ndarray, homogenea: bool = False):
	if homogenea:
		sk=0
	else:
		st = 0.2 * np.sin(2 * 10 * np.pi * t)
		sk = (
			5 * np.sqrt(np.pi)
			* np.exp(1j * k * (np.pi+st))
			* np.exp(
				- np.pi ** 2
				* k ** 2
			)
		)

	r = (
		- FFT(
			IFFT(u) * IFFT(1j * k * u)
		)
		- VISC * (k ** 2) * u
		+ sk
	)
	# print(f't={t:.2f}, | sk | / | r | = {np.linalg.norm(sk)/np.linalg.norm(r):.2f}', end='\r')
	return r

def main():
	fourier_result = solve_ivp(
		fun = G, 
		t_span = [t_eval[0],t_eval[-1]], 
		y0=f0, 
		t_eval=t_eval,
		args=[HOMOGENEA]
	)
	f = np.array(
		[
			IFFT(fourier_f)
			for fourier_f in fourier_result.y.T
		]
	)

	df = pd.DataFrame({ 
		**{
			f't_{t}': f[i]
			for i, t in enumerate(t_eval)
		},
		**{
			'x': [2 * np.pi * l/len(f[0]) for l in range(len(f[0]))]
		}
		})

	axes = df.plot(x='x')
	plt.legend(
		bbox_to_anchor=(1.05, 0.5), 
		loc='center left'
	)
	plt.tight_layout()

	if HOMOGENEA:
		plt.savefig('new_homogenea.png')
	else:
		plt.savefig('new_fuente.png')


if __name__=='__main__':
	main()
