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
