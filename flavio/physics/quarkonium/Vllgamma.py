r"""$V\to ll^\prime\gamma$ branching ratio"""

from flavio.classes import Observable, Prediction
from flavio.physics.running import running
import flavio
from flavio.physics.quarkonium.Vll import Vll_br, wc_sector
import numpy as np

meson_quark = { 'J/psi' : 'cc', 
                'psi(2S)': 'cc',
                'Upsilon(1S)': 'bb',
                'Upsilon(2S)': 'bb',
                'Upsilon(3S)': 'bb',
                }

def getWC_lfv(wc_obj,par,V,Q,l1,l2,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji):
    # renormalization scale
    scale = flavio.config['renormalization scale'][V]
    # Wilson coefficients
    wc = wc_obj.get_wc(wc_sector[(l1,l2)], scale, par)

    alphaem = running.get_alpha_e(par, scale)
    ee=np.sqrt(4.*np.pi*alphaem) 

    mV = par['m_'+V]   
    fV=par['f_'+V] 
    fV_T=par['fT_'+V]
    qq=meson_quark[V]
    # for emu and taue the name of the Wilson coefficient sector agrees with the ordering of leptons in the vector bilinear
    # This is not the case for mutau. Thus distinguish between the two cases here.
    if wc_sector[(l1,l2)]=="mutau":
        ll="taumu"
    else:
        ll=wc_sector[(l1,l2)]
    VR=fV*mV*(wc['CVRR_'+ll+qq] + wc['CVLR_'+qq+ll]) 
    VL=fV*mV*(wc['CVLL_'+ll+qq] + wc['CVLR_'+ll+qq]) 
    TR=fV_T*mV*wc['CTRR_'+l1+l2+qq] - ee*Q *fV*wc['Cgamma_'+l2+l1] 
    TL=(fV_T*mV*wc['CTRR_'+l2+l1+qq] - ee*Q *fV*wc['Cgamma_'+l1+l2]).conjugate()
    SR=2.*mV*fV*(wc['CSRR_'+l1+l2+qq]+wc['CSRL_'+l1+l2+qq])
    SL=2.*mV*fV*(wc['CSRL_'+l2+l1+qq]+wc['CSRR_'+l2+l1+qq]).conjugate()   
    PR=2.*mV*fV*(wc['CSRR_'+l1+l2+qq]-wc['CSRL_'+l1+l2+qq])
    PL=2.*mV*fV*(wc['CSRL_'+l2+l1+qq]-wc['CSRR_'+l2+l1+qq]).conjugate()   
    AR=2.*fV*mV*(wc['CVLR_'+qq+ll]- wc['CVRR_'+ll+qq]) 
    AL=2.*fV*mV*(wc['CVLL_'+ll+qq] - wc['CVLR_'+ll+qq]) 

    StildeR=4*mV**2*fV*CeFFij
    StildeL=4*mV**2*fV*CeFFji.conjugate()
    PtildeR=4*mV**2*fV*CeFFtildeij
    PtildeL=4*mV**2*fV*CeFFtildeji.conjugate()

    if ll==l2+l1: 
    # As we use ll instead of l1+l2 or l2+l1 above when constructing the effective parameters, we have to complex conjugate it, in case indices are reversed.
    # This is not necessary for the others, because there we explicitly defined the flavour indices.
        VL=VL.conjugate()
        VR=VR.conjugate()
        AL=AL.conjugate()
        AR=AR.conjugate()
    return VL,VR,TL,TR,SR,SL,PR,PL,AR,AL,StildeR,StildeL,PtildeR,PtildeL

# we use \mu=y^2 inside the kinematic functions and denote them by F_ instead of G_ in the paper.

def F_A(y):
    mu=y**2
    if mu==0:
        return 2./9
    return (8.-45*mu+36*mu**2+mu**3+6*(mu-6)*mu**2*np.log(mu))/36
def F_S(y):
    mu=y**2
    if mu==0:
        return 1./12
    return (1-6*mu+3*mu**2*(1-2*np.log(mu))+2*mu**3)/12.
def Ftilde_S(y):
    mu=y**2
    if mu==0:
        return 1./40
    return (3.-30.*mu-20.*mu**2+60.*mu**3-15*mu**4+2*mu**5-60.*mu**2*np.log(mu))/120.
def Fhat_P(y):
    mu=y**2
    if mu==0:
        return 0.
    return mu*(-8+8*mu**2-mu**3 - 12*mu*np.log(mu))/12.
def Fhat_S(y):
    mu=y**2
    if mu==0:
        return 1./12
    return Fhat_P(y) + 1./12
def F_PA(y):
    if y==0:
        return 0.
    mu=y**2
    return y*(1.+4*mu-5*mu**2+2*mu*(2+mu)*np.log(mu))/2.
def Ftilde_PA(y):
    if y==0:
        return 0.
    mu=y**2
    return y*(1.+9*mu-9*mu**2-mu**3+6*mu*(1+mu)*np.log(mu))/3.


def Vllgamma_br(wc_obj, par,V,Q, l1,l2,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji):
    r"""Branching ratio for the lepton-flavour violating leptonic decay J/psi-> l l' \gamma"""
    #####branching ratio obtained from 2207.10913#####
    flavio.citations.register("Calibbi:2022ddo")
    # renormalization scale
    scale = flavio.config['renormalization scale'][V]
    alphaem = running.get_alpha_e(par, scale)
    mV = par['m_'+V]   
    tauV = par['tau_'+V]  
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]

    VL,VR,TL,TR,SR,SL,PR,PL,AR,AL,StildeR,StildeL,PtildeR,PtildeL = getWC_lfv(wc_obj,par,V,Q,l1,l2,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji)
 
    AV=np.abs(AL)**2+np.abs(AR)**2
    SP=np.abs(SL)**2+np.abs(PL)**2+np.abs(SR)**2+np.abs(PR)**2
    SPtilde=np.abs(StildeL)**2+ np.abs(PtildeL)**2 +np.abs(StildeR)**2  +np.abs(PtildeR)**2
    SStilde= (SL*StildeL.conjugate() + SR*StildeR.conjugate()).real
    PPtilde= (PL*PtildeL.conjugate() + PR*PtildeR.conjugate()).imag

   
    if ml1<ml2:
        y=ml2/mV
        I_AP=-(AL*PR.conjugate()+AR*PL.conjugate()).real
        I_APtilde=-(AL*PtildeR.conjugate()+AR*PtildeL.conjugate()).imag
    elif ml2<ml1:
        y=ml1/mV
        I_AP=(AL*PL.conjugate()+AR*PR.conjugate()).real
        I_APtilde=(AL*PtildeL.conjugate()+AR*PtildeR.conjugate()).imag
    else:
        print("The case of non-hierarchical masses is not implemented.")

    prefactor=alphaem*Q**2*tauV*mV/(192*np.pi**2)
    
    return prefactor*(AV * F_A(y) + SP*F_S(y) + I_AP*F_PA(y) + SPtilde *Ftilde_S(y) + SStilde * Fhat_S(y) + PPtilde * Fhat_P(y) +I_APtilde * Ftilde_PA(y))


def Vllgamma_br_func(V, Q, l1, l2):
    def fct(wc_obj, par,CeFFij=0,CeFFji=0,CeFFtildeij=0,CeFFtildeji=0):
        return Vllgamma_br(wc_obj, par, V, Q, l1, l2,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji)
    return fct


def Vllgamma_br_comb_func(V, Q, l1, l2):
    def fct(wc_obj, par,CeFFij=0,CeFFji=0,CeFFtildeij=0,CeFFtildeji=0):
        return Vllgamma_br(wc_obj, par, V, Q, l1, l2,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji)+ Vllgamma_br(wc_obj, par, V, Q, l2, l1,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji)
    return fct

def Vllgamma_ratio_func(V, Q, l1, l2):
    def fct(wc_obj, par,CeFFij=0,CeFFji=0,CeFFtildeij=0,CeFFtildeji=0):
        BRee=Vll_br(wc_obj,par,V,Q,'e','e')
        return Vllgamma_br(wc_obj, par, V, Q, l1, l2,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji)/BRee
    return fct


def Vllgamma_ratio_comb_func(V, Q, l1, l2):
    def fct(wc_obj, par,CeFFij=0,CeFFji=0,CeFFtildeij=0,CeFFtildeji=0):
        BRee=Vll_br(wc_obj,par,V,Q,'e','e')
        return (Vllgamma_br(wc_obj, par, V, Q, l1, l2,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji)+ Vllgamma_br(wc_obj, par, V, Q, l2, l1,CeFFij,CeFFji,CeFFtildeij,CeFFtildeji))/BRee
    return fct



# Observable and Prediction instances
_hadr = { 
    'J/psi': {'tex': r"J/\psi\to", 'V': 'J/psi', 'Q': 2./3., },
    'psi(2S)': {'tex': r"\psi(2S)\to", 'V': 'psi(2S)', 'Q': 2./3., }, 
    'Upsilon(1S)': {'tex': r"\Upsilon(1S)\to", 'V': 'Upsilon(1S)', 'Q': -1./3., },
    'Upsilon(2S)': {'tex': r"\Upsilon(2S)\to", 'V': 'Upsilon(2S)', 'Q': -1./3., },
    'Upsilon(3S)': {'tex': r"\Upsilon(3S)\to", 'V': 'Upsilon(3S)', 'Q': -1./3., },
 }

_tex = {'ee': r'e^+e^-', 'mumu': r'\mu^+\mu^-', 'tautau': r'\tau^+\tau^-', 
    'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-',
    'emu,mue': r'e^\pm\mu^\mp', 'etau,taue': r'e^\pm\tau^\mp','mutau,taumu': r'\mu^\pm\tau^\mp',
    'mue,emu': r'e^\pm\mu^\mp', 'taue,etau': r'e^\pm\tau^\mp','taumu,mutau': r'\mu^\pm\tau^\mp'
    }


def _define_obs_V_ll_gamma(M, ll):
    _process_tex = _hadr[M]['tex']+_tex[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $V\to \ell^+\ell^-\gamma$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+_hadr[M]['V']+"->"+''.join(ll)+"gamma)"
    _obs = Observable(_obs_name,arguments=["CeFFij","CeFFji","CeFFtildeij","CeFFtildeji"])
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name

def _define_obs_V_ll_gamma_ratio(M, ll):
    _process_tex = _hadr[M]['tex']+_tex[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $V\to \ell^+\ell^-\gamma$ :: $' + _process_tex + r"$"
    _obs_name = "R("+_hadr[M]['V']+"->"+''.join(ll)+"gamma)"
    _obs = Observable(_obs_name,arguments=["CeFFij","CeFFji","CeFFtildeij","CeFFtildeji"])
    _obs.set_description(r"Ratio of $"+_process_tex+r"$ over BR$(V\to ee)$")
    _obs.tex = r"$\text{R}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name


for M in _hadr:
    for (l1,l2) in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
        _obs_name = _define_obs_V_ll_gamma(M, (l1,l2))
        Prediction(_obs_name, Vllgamma_br_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
        _obs_name = _define_obs_V_ll_gamma_ratio(M, (l1,l2))
        Prediction(_obs_name, Vllgamma_ratio_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))

    for (l1,l2) in [('e','mu'), ('e','tau'), ('mu','tau')]:
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_V_ll_gamma(M, ('{0}{1},{1}{0}'.format(l1,l2),))
        Prediction(_obs_name, Vllgamma_br_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
        _obs_name = _define_obs_V_ll_gamma_ratio(M, ('{0}{1},{1}{0}'.format(l1,l2),))
        Prediction(_obs_name, Vllgamma_ratio_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
        
        _obs_name = _define_obs_V_ll_gamma(M, ('{1}{0},{0}{1}'.format(l1,l2),))
        Prediction(_obs_name, Vllgamma_br_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
        _obs_name = _define_obs_V_ll_gamma_ratio(M, ('{1}{0},{0}{1}'.format(l1,l2),))
        Prediction(_obs_name, Vllgamma_ratio_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
 
