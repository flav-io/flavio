from math import sqrt
import numpy as np

def z(mB, mM, q2, t0=None):
    r"""Form factor expansion parameter $z$.

    Parameters
    ----------
    mB : float
        initial pseudoscalar meson mass
    mM : float
        final meson meson mass
    q2 : float
        momentum transfer squared $q^2$
    t0 : float, optional
        parameter $t_0$.
        If not given, chosen as $t_0 = t_+ (1-\sqrt{1-t_-/t_+})$ where
        $t_\pm = (m_B \pm m_M)^2$.
    """
    tm = (mB-mM)**2
    tp = (mB+mM)**2
    if t0 is None:
        t0 = tp*(1-sqrt(1-tm/tp))
    sq2 = sqrt(tp-q2)
    st0 = sqrt(tp-t0)
    return (sq2-st0)/(sq2+st0)
