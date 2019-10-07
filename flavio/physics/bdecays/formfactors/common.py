from math import sqrt
import numpy as np
from flavio.config import config
from functools import lru_cache

@lru_cache(maxsize=config['settings']['cache size'])
def z(mB, mM, q2, t0=None):
    r"""Form factor expansion parameter $z$.

    Parameters
    ----------

    - `mB`:
        initial pseudoscalar meson mass
    - `mM`:
        final meson meson mass
    - `q2`:
        momentum transfer squared $q^2$
    - `t0` (optional):
        parameter $t_0$.
        If not given, chosen as $t_0 = t_+ (1-\sqrt{1-t_-/t_+})$ where
        $t_\pm = (m_B \pm m_M)^2$.
        If equal to `'tm'`, set to $t_0=t_-$
    """
    tm = (mB-mM)**2
    tp = (mB+mM)**2
    if t0 is None:
        t0 = tp*(1-sqrt(1-tm/tp))
    elif t0 == 'tm':
        t0 = tm
    sq2 = sqrt(tp-q2)
    st0 = sqrt(tp-t0)
    return (sq2-st0)/(sq2+st0)


def w_minus_1_pow_n(z, n, order_z):
    r"""Monomial $(w-1)^n$ where $w$ is expressed in terms of $z$ as $w(z)$ and
    the whole expression is expanded around $z=0$ until (including) power $z^m$,
    where $m=$`order_z`.

    The exact expression is $w(z)=(1 + 6 z + z^2)/(-1 + z)^2$.
    """
    zs = np.array([1, z, z**2, z**3])  # zs[i] = z**i
    if order_z > 3:
        raise ValueError("(w-1)^n monomial only implemented until order_z=3.")
    if n > 3:
        raise ValueError("(w-1)^n monomial only implemented until n=3.")
    if n == 0:
        return 1
    if n == 1:
        a = np.array([0, 8, 16, 24])
    if n == 2:
        a = np.array([0, 0, 64, 256])
    if n == 3:
        a = np.array([0, 0, 0, 512])
    return np.sum((a * zs)[:order_z + 1])  # sum_i=0^order_z a[i] * z**i
