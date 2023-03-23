r"""$S\to ll^\prime$ branching ratio"""

from flavio.classes import Observable, Prediction
import flavio
from flavio.physics.running import running
from flavio.physics.common import lambda_K
from flavio.physics.quarkonium.Vll import wc_sector
import numpy as np

meson_quark = { 'chi_c0(1P)' : 'cc', 
                'chi_b0(1P)': 'bb',
                'chi_b0(2P)': 'bb',
                }


def getS_lfv(wc_obj,par,S,l1,l2,CeGGij,CeGGji):
    # renormalization scale
    scale = flavio.config['renormalization scale'][S]
    alphas = running.get_alpha_s(par, scale)
    # Wilson coefficients
    wc = wc_obj.get_wc(wc_sector[(l1,l2)], scale, par)

    mS = par['m_'+S]   
    # The form factor in the parameters_uncorrelated.yml file follows 1607.00815. Thus we have to include an additional factor of "-i" which drops out in the absence of an anomaly term.
    fS=par['f_'+S]*(-1j) 
    aS=par['a_'+S]
    qq=meson_quark[S]
    aPfac=1j*aS*4.*np.pi/alphas
    SR= aPfac * CeGGij +  (mS*fS/2.)*(wc['CSRR_'+l1+l2+qq]+wc['CSRL_'+l1+l2+qq])  
    SL= aPfac * CeGGji + (mS*fS/2.)*(wc['CSRL_'+l2+l1+qq]+wc['CSRR_'+l2+l1+qq]).conjugate() 
    return SL,SR


def Sll_br(wc_obj, par,S, l1,l2,CeGGij,CeGGji):
    r"""Branching ratio for the lepton-flavour violating leptonic decay P -> l l' """
    #####branching ratio obtained from 2207.10913#####
    flavio.citations.register("Calibbi:2022ddo")
    mP = par['m_'+S]   
    tauP = par['tau_'+S]  
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    y1=ml1/mP
    y2=ml2/mP
    y1s=y1**2
    y2s=y2**2
    SL,SR = getS_lfv(wc_obj,par,S,l1,l2,CeGGij,CeGGji)
    return  tauP*mP/(16.*np.pi) * np.sqrt(lambda_K(1,y1s,y2s)) * ((1-y1s-y2s)*(np.abs(SL)**2+np.abs(SR)**2) -4*y1*y2 *(SL*SR.conjugate()).real)


def Sll_br_func(S,  l1, l2):
    def fct(wc_obj, par,CeGGij=0,CeGGji=0):
        return Sll_br(wc_obj, par, S,  l1, l2,CeGGij,CeGGji)
    return fct


def Sll_br_comb_func(S,  l1, l2):
    def fct(wc_obj, par,CeGGij=0,CeGGji=0):
        return Sll_br(wc_obj, par, S,  l1, l2,CeGGij,CeGGji)+ Sll_br(wc_obj, par, S,  l2, l1,CeGGij,CeGGji)
    return fct


# Observable and Prediction instances
_hadr = { 
    'chi_c0(1P)': {'tex': r"\chi_{c0}(1P)\to", 'S': 'chi_c0(1P)', 'Q': 2./3., },
    'chi_b0(1P)': {'tex': r"\chi_{b0}(1P)\to", 'S': 'chi_b0(1P)', 'Q': -1./3., },
    'chi_b0(2P)': {'tex': r"\chi_{b0}(2P)\to", 'S': 'chi_b0(2P)', 'Q': -1./3., },
 }

_tex = {'ee': r'e^+e^-', 'mumu': r'\mu^+\mu^-', 'tautau': r'\tau^+\tau^-', 
    'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-',
    'emu,mue': r'e^\pm\mu^\mp', 'etau,taue': r'e^\pm\tau^\mp','mutau,taumu': r'\mu^\pm\tau^\mp',
    'mue,emu': r'e^\pm\mu^\mp', 'taue,etau': r'e^\pm\tau^\mp','taumu,mutau': r'\mu^\pm\tau^\mp'
    }


def _define_obs_S_ll(M, ll):
    _process_tex = _hadr[M]['tex']+_tex[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $S\to \ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+_hadr[M]['S']+"->"+''.join(ll)+")"
    _obs = Observable(_obs_name,arguments=["CeGGij","CeGGji"])
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name


for M in _hadr:
    for (l1,l2) in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
        _obs_name = _define_obs_S_ll(M, (l1,l2))
        Prediction(_obs_name, Sll_br_func(_hadr[M]['S'], l1,l2))
    for ll in [('e','mu'), ('e','tau'), ('mu','tau')]:
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_S_ll(M, ('{0}{1},{1}{0}'.format(l1,l2),))
        Prediction(_obs_name, Sll_br_comb_func(_hadr[M]['S'],  l1,l2))
        _obs_name = _define_obs_S_ll(M, ('{1}{0},{0}{1}'.format(l1,l2),))
        Prediction(_obs_name, Sll_br_comb_func(_hadr[M]['S'],  l1,l2))
 
