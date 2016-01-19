from math import pi

meson_quark = { 'B0': 'bd', 'Bs': 'bs', 'K0': 'sd', 'D0': 'cu' }

def bag_msbar2rgi(alpha_s, meson):
    """Conversion factor between renormalization group invariant (RGI) defintion
    $\hat B$ of the bag parameter and the $\overline{\mathrm{MS}}$ definition
    $B(mu)$:
    $$\hat B = b_B^{(n_f)} B(mu)$$

    See e.g. eq. (84) in arXiv:1011.4408.
    """
    J={}
    if meson in ['B0', 'Bs']: # nf=5
        J = 5165/3174.
        g = 6/23
    elif meson == 'K0': # nf=3
        J = 307/162.
        g = 2/9.
    return alpha_s**(-g) * (1 + alpha_s/(4*pi) * J)
