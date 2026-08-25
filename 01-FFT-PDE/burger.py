import numpy as np
from scipy.integrate import solve_ivp
import scipy.fft
import scipy.fft.fft as FFT
import scipy.fft.ifft as IFFT

# discretización
N = 128

# condiciones iniciales
f0 = np.zeros(N)
t_eval = [0.1 * t for t in range(11)]
k = scipy.fft.fftfreq(N)
X = np.linspace(0, 1, N)

def S(x,t):
	st = np.cos(2 * np.pi * t)
	return np.exp(-(x - st) ** 2 / 2)

def G(t, u: np.ndarray):
	sk = FFT(np.array([S(x, t) for x in X]))
	return (
		- FFT(
			IFFT(u) * IFFT(1j * k * u)
		)
		- (k ** 2) * u
		+ sk
	)

def main():
	fourier_result = solve_ivp(fun = G, t_span = [0,1], y0=f0, t_eval=t_eval)
	f = np.array(
		[
			IFFT(fourier_f for fourier_f in fourier_result.y)
		]
	)

	# plottear soluciones para cada t:
	...