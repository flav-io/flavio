r"""Functions for the branching ratio of the leptonic decay
$B_q\to\ell^+\ell^-\gamma$, where $q=d$ or $s$ and $\ell=e$, $\mu$ or
$\tau$. The branching ratio is taken from hep-ph/0410146"""

import flavio
from math import sqrt,pi,atanh
from flavio.physics.bdecays.common import meson_quark
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays.wilsoncoefficients import get_wceff, wctot_dict
from flavio.classes import Observable, Prediction
import warnings

def _Re(z):
    return z.real
def _Im(z):
    return z.imag
def _Co(z):
    return complex(z).conjugate()

##################
# hep-ph/0410146:
#   - Uses the C7>0, C9<0, C10>0 convention
#   - has a global minus sign typo in dG12dsMN
#   - defines f_V instead of f_V^{.e.m} (see 1712.07926 and fix in the code)
#   - Defines an effective C9 (as for B->P/Vll), for the moment C9 is used,
#     which means that charmonium is not implemented!!
##################

# form factors
def get_ff(q2, par, B):
    return AuxiliaryQuantity['B->gamma form factor'].prediction(par_dict=par, wc_obj=None, q2=q2, B=B)


def prefactor(s, par, B, ff, lep, wc):
    GF = par['GF']
    scale = config['renormalization scale']['bllgamma']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    bq = meson_quark[B]
    xi_t = ckm.xi('t',bq)(par)
    return GF**2/(2**10*pi**4)*abs(xi_t)**2*alphaem**3*par['m_'+B]**5

def getfft(s, par, B, ff, ff0, lep, wc):
    scale = config['renormalization scale']['bllgamma']
    mb = running.get_mb(par, scale, nf_out=5)
    bq = meson_quark[B]
    a1 = -wc['C1_'+bq]-wc['C2_'+bq]/3 #minus sign comes from WC sign convention
    fftv = ff['tv']+ff0['tv']+16/3*a1/wc['C7_'+bq]*par['f_'+B]/mb
    ffta = ff['ta']+ff0['ta']

    #Add light meson resonances
    resonances = {'Bs': ['phi'], 'B0': ['rho0', 'omega']}
    flavio.citations.register("Kozachuk:2017mdk")
    fVtofVemfactors = {'phi': -1/3, 'rho0': 1/sqrt(2), 'omega': 1/(3*sqrt(2))} # Given in Sec. 8.A.3 of 1712.07926
    #We use the general B->rho and B->omega parameters, hence the isotopic factors
    resgV = {('Bs','phi'): -par['Bs->phi BSZ a0_T1'],
             ('B0','rho0'): 1/sqrt(2)*par['B->rho BSZ a0_T1'],
             ('B0','omega'): -1/sqrt(2)*par['B->omega BSZ a0_T1'],}
    for V in resonances[B]:
        gV = resgV[(B, V)]
        fV = fVtofVemfactors[V]*par['f_'+V]
        fftv -= 2*fV*gV*par['m_'+B]**2*s/par['m_'+V]/(par['m_'+B]**2*s-par['m_'+V]**2+1j*par['m_'+V]/par['tau_'+V])
        ffta -= 2*fV*gV*par['m_'+B]**2*s/par['m_'+V]/(par['m_'+B]**2*s-par['m_'+V]**2+1j*par['m_'+V]/par['tau_'+V])

    return (fftv, ffta)

def getF1(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    scale = config['renormalization scale']['bllgamma']
    mb = running.get_mb(par, scale, nf_out=5)
    mbh = mb/par['m_'+B]
    bq = meson_quark[B]
    label = bq+lep+lep
    fftv, ffta = getfft(s, par, B, ff, ff0, lep, wc)
    return (abs(wc['C9_'+label])**2 + abs(wc['C10_'+label])**2)*ff['v']**2 + 4*mbh**2/s**2*abs(wc['C7_'+bq]*fftv)**2 + 4*mbh/s*ff['v']*_Re(wc['C7_'+bq]*ffta*_Co(wc['C9_'+label]))

def getF2(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    scale = config['renormalization scale']['bllgamma']
    mb = running.get_mb(par, scale, nf_out=5)
    mbh = mb/par['m_'+B]
    bq = meson_quark[B]
    label = bq+lep+lep
    fftv, ffta = getfft(s, par, B, ff, ff0, lep, wc)
    return (abs(wc['C9_'+label])**2 + abs(wc['C10_'+label])**2)*ff['a']**2 + 4*mbh**2/s**2*abs(wc['C7_'+bq]*ffta)**2 + 4*mbh/s*ff['a']*_Re(wc['C7_'+bq]*ffta*_Co(wc['C9_'+label]))


def B10(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    flavio.citations.register("Guadagnoli:2016erb")
    F1 = getF1(s, par, B, ff, ff0, lep, wc)
    F2 = getF2(s, par, B, ff, ff0, lep, wc)
    mlh = par['m_'+lep]/par['m_'+B]
    label = meson_quark[B]+lep+lep
    return (s + 4*mlh**2)*(F1 + F2) - 8*mlh**2*abs(wc['C10_'+label])**2*(ff['v']**2+ff['a']**2)

# B11 vanishes for the BR estimation

def B12(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    flavio.citations.register("Guadagnoli:2016erb")
    F1 = getF1(s, par, B, ff, ff0, lep, wc)
    F2 = getF2(s, par, B, ff, ff0, lep, wc)
    return s*(F1+F2)

def B120(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    flavio.citations.register("Guadagnoli:2016erb")
    scale = config['renormalization scale']['bllgamma']
    mb = running.get_mb(par, scale, nf_out=5)
    mbh = mb/par['m_'+B]
    mlh = par['m_'+lep]/par['m_'+B]
    bq = meson_quark[B]
    label = bq+lep+lep
    fftv, ffta = getfft(s, par, B, ff, ff0, lep, wc)
    return -16*mlh**2*(1-s)*(ff['v']*_Re(wc['C9_'+label]*_Co(wc['C10_'+label])) + 2*mbh/s*_Re(_Co(wc['C10_'+label])*fftv*wc['C7_'+bq]))


def dG1dsMN(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    mlh = par['m_'+lep]/par['m_'+B]
    pref = prefactor(s, par, B, ff, lep, wc)*(1-s)**3*sqrt(1-4*mlh**2/s)
    return pref*(B10(s, par, B, ff, ff0, lep, wc) + (s - 4*mlh**2)/(3*s)*B12(s, par, B, ff, ff0, lep, wc))

def dG2dsMN(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    mlh = par['m_'+lep]/par['m_'+B]
    bq = meson_quark[B]
    label = bq+lep+lep
    pref = 16*prefactor(s, par, B, ff, lep, wc)*(par['f_'+B]/par['m_'+B]*abs(wc['C10_'+label])*mlh)**2/(1-s)
    return pref*(-8*sqrt(s-4*mlh**2)*sqrt(s)+4*atanh(sqrt(1-4*mlh**2/s))*(1+s-mlh**2*(1-s)**2))

def dG12dsMN(s, par, B, ff, ff0, lep, wc):
    flavio.citations.register("Melikhov:2004mk")
    mlh = par['m_'+lep]/par['m_'+B]
    pref = 4*prefactor(s, par, B, ff, lep, wc)*par['f_'+B]/par['m_'+B]*(1-s)
    return pref*atanh(sqrt(1-4*mlh**2/s))*B120(s, par, B, ff, ff0, lep, wc)

def dGdsMN(s, par, B, ff, ff0, lep, wc):
    return dG1dsMN(s, par, B, ff, ff0, lep, wc) + dG2dsMN(s, par, B, ff, ff0, lep, wc) + dG12dsMN(s, par, B, ff, ff0, lep, wc)


def bllg_dbrdq2(q2, wc_obj, par, B, lep):
    if q2 >= 8.7 and q2 < 14 and lep != 'tau':
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    tauB = par['tau_'+B]
    ml = par['m_'+lep]
    mB = par['m_'+B]
    scale = config['renormalization scale']['bllgamma']
    ff = get_ff(q2, par, B)
    ff0 = get_ff(0, par, B)
    label = meson_quark[B]+lep+lep
    wc = wctot_dict(wc_obj, label, scale, par)
    if q2 <= 4*ml**2 or q2 > mB**2:
        return 0
    else:
        dBR = tauB * dGdsMN(q2/mB**2, par, B, ff, ff0, lep, wc)
        return dBR

def bllg_dbrdq2_func(B, lep):
    def fct(wc_obj, par, q2):
        return bllg_dbrdq2(q2, wc_obj, par, B, lep)
    return fct

def bllg_dbrdq2_int(q2min, q2max, wc_obj, par, B, lep, epsrel=0.005):
    def obs(q2):
        return bllg_dbrdq2(q2, wc_obj, par, B, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max, epsrel=epsrel)/(q2max-q2min)

def bllg_dbrdq2_int_func(B, lep):
    def fct(wc_obj, par, q2min, q2max):
        return bllg_dbrdq2_int(q2min, q2max, wc_obj, par, B, lep)
    return fct

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_tex_B = {'B0': r'\bar B^0', 'Bs': r'\bar B_s'}

_process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to \ell^+\ell^-\gamma$ :: $'

for l in ['e', 'mu']:
    for B in ['Bs', 'B0']:

        _process_tex =  _tex_B[B]+r"\to" +_tex[l]+r"^+"+_tex[l]+r"^-\gamma"


        # binned branching ratio
        _obs_name = "<dBR/dq2>("+B+"->"+l+l+"gamma)"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Binned differential branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
        Prediction(_obs_name, bllg_dbrdq2_int_func(B, l))

        # differential branching ratio
        _obs_name = "dBR/dq2("+B+"->"+l+l+"gamma)"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Differential branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
        Prediction(_obs_name, bllg_dbrdq2_func(B, l))
