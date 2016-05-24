import scipy
import numpy as np

def nintegrate(f, a, b, epsrel=0.01, **kwargs):
    return scipy.integrate.quad(f, a, b, epsrel=epsrel, epsabs=0, **kwargs)[0]


def nintegrate_complex(func, a, b, epsrel=0.01, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = scipy.integrate.quad(real_func, a, b, epsrel=epsrel, epsabs=0, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, a, b, epsrel=epsrel, epsabs=0, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]
