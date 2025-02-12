import scipy
import numpy as np

def nintegrate(f, a, b, epsrel=0.005, **kwargs):
    return scipy.integrate.quad(f, a, b, epsabs=0, epsrel=epsrel, **kwargs)[0]

def nintegrate_fast(f, a, b, N=5, **kwargs):
    x = np.linspace(a,b,N)
    y = np.array([f(X) for X in x])
    f_interp = scipy.interpolate.interp1d(x, y, kind='cubic')
    x_fine = np.linspace(a,b,N*4)
    y_interp = np.array([f_interp(X) for X in x_fine])
    return np.trapz(y_interp, x=x_fine)

def nintegrate_complex(func, a, b, epsrel=0.005, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = scipy.integrate.quad(real_func, a, b, epsrel=epsrel, epsabs=0, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, a, b, epsrel=epsrel, epsabs=0, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]
