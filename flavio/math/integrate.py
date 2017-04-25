import scipy
import numpy as np
import warnings
from scipy.integrate.quadrature import AccuracyWarning

def nintegrate(f, a, b, epsrel=0.005, **kwargs):
    with warnings.catch_warnings():
        # ignore AccuracyWarning that is issued when an integral is zero
        warnings.filterwarnings("ignore", category=AccuracyWarning)
        return scipy.integrate.quadrature(f, a, b, rtol=epsrel, tol=0, vec_func=False, **kwargs)[0]

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
