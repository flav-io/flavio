r"""Functions for $K\to pi$ hadronic form factors."""

import flavio
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config
from math import exp

def fp0_dispersive(q2, par):
    mK = par['m_KL']
    mpi = par['m_pi+']
    DeltaKpi = mK**2 - mpi**2
    t0 = (mK - mpi)**2
    # specific parameters
    fp_0 = par['K->pi f+(0)']
    lnC =    par['K->pi ln(C)']
    Lp =   par['K->pi Lambda_+']
    D =    par['K->pi D']
    d =    par['K->pi d']
    k =    par['K->pi k']
    H1 =    par['K->pi H1']
    H2 =    par['K->pi H2']
    # (A.1) of 0903.1654
    flavio.citations.register("Bernard:2009zm")
    x = q2/t0
    G = x*D + (1-x)*d + x*(1-x)*k
    # (A.3) of 0903.1654
    flavio.citations.register("Bernard:2009zm")
    H = H1*x + H2*x**2
    # (33) of 1005.2323
    flavio.citations.register("Antonelli:2010yf")
    f0_bar = exp(q2/DeltaKpi * (lnC - G))
    # (37) of 1005.2323
    flavio.citations.register("Antonelli:2010yf")
    fp_bar = exp(q2/mpi**2 * (Lp + H))
    ff = {}
    ff['f+'] = fp_0 * fp_bar
    ff['f0'] = fp_0 * f0_bar
    return ff

def fT_pole(q2, par):
    # specific parameters
    fT_0 = par['K->pi fT(0)']
    sT = par['K->pi sT']
    ff = {}
    # (4) of 1108.1021
    ff['fT'] = fT_0 / (1- sT * q2)
    return ff

def ff_dispersive_pole(wc_obj, par_dict, q2):
    ff = {}
    ff.update( fp0_dispersive(q2, par_dict) )
    ff.update( fT_pole(q2, par_dict) )
    return ff

quantity = 'K->pi form factor'
a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
a.set_description(r'Hadronic form factor for the $K\to\pi$ transition')

iname = 'K->pi dispersive + pole'
i = Implementation(name=iname, quantity=quantity,
               function=ff_dispersive_pole)
i.set_description(r"Dispersive parametrization (see arXiv:hep-ph/0603202) for "
                  r"$f_+$ and $f_0$ and simple pole for $f_T$.")
