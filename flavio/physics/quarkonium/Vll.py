r"""$V\to ll^\prime$ branching ratio"""

from flavio.classes import Observable, Prediction
from flavio.physics.running import running
import flavio
from flavio.physics.common import lambda_K
import numpy as np

meson_quark = { 'J/psi' : 'cc', 
                'psi(2S)': 'cc',
                'Upsilon(1S)': 'bb',
                'Upsilon(2S)': 'bb',
                'Upsilon(3S)': 'bb',
                }

wc_sector = { ('e','e') : "dF=0", ('mu','mu'): "dF=0", ('tau','tau'): "dF=0", ('e','mu') : 'mue',  ('mu','e') :'mue', ('e','tau') : 'taue', ('tau','e') : 'taue', ('mu','tau') : 'mutau', ('tau','mu') : 'mutau'}

def getVT_lfv(wc_obj,par,V,Q,l1,l2):
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
    VL=fV*mV*(wc['CVLL_'+ll+qq] + wc['CVLR_'+ll+qq]) 
    VR=fV*mV*(wc['CVRR_'+ll+qq] + wc['CVLR_'+qq+ll]) 
    TR=fV_T*mV*wc['CTRR_'+l1+l2+qq] - ee*Q *fV*wc['Cgamma_'+l2+l1] 
    TL=(fV_T*mV*wc['CTRR_'+l2+l1+qq] - ee*Q *fV*wc['Cgamma_'+l1+l2]).conjugate()
    if ll==l2+l1: 
        VL=VL.conjugate()
        VR=VR.conjugate()
    return VL,VR,TL,TR

def getVT_lfc(wc_obj,par,V,Q,l):
# add contribution from photon exchange and then activate LFC channels

    # renormalization scale
    scale = flavio.config['renormalization scale'][V]
    # Wilson coefficients
    wc = wc_obj.get_wc(wc_sector[(l,l)], scale, par)
    ll=l+l
    alphaem = running.get_alpha_e(par, scale)
    ee2=4.*np.pi*alphaem
    ee=np.sqrt(ee2) 

    norm=4*par['GF']/np.sqrt(2)
    normDipole=norm*ee/(16*np.pi**2)*par['m_'+l]

    mV = par['m_'+V]   
    fV=par['f_'+V] 
    fV_T=par['fT_'+V]
    qq=meson_quark[V]

    VL=fV*mV*(wc['CVLL_'+ll+qq]*norm + wc['CVLR_'+ll+qq]*norm - 2*Q* ee2/mV**2 ) 
    VR=fV*mV*(wc['CVRR_'+ll+qq]*norm + wc['CVLR_'+qq+ll]*norm  - 2*Q* ee2/mV**2 )
    TR=fV_T*mV*wc['CTRR_'+ll+qq]*norm - ee*Q *fV*wc['C7_'+ll]*normDipole 
    TL=TR.conjugate()
    
    return VL,VR,TL,TR


def Vll_br(wc_obj, par,V,Q, l1,l2):
    r"""Branching ratio for the lepton-flavour violating leptonic decay J/psi-> l l'"""
    #####branching ratio obtained from 2207.10913#####
    flavio.citations.register("Calibbi:2022ddo")
    mV = par['m_'+V]   
    tauV = par['tau_'+V]  
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    y1=ml1/mV
    y2=ml2/mV
    y1s=y1**2
    y2s=y2**2
    if l1==l2:
        VL,VR,TL,TR = getVT_lfc(wc_obj,par,V,Q,l1)
    else:
        VL,VR,TL,TR = getVT_lfv(wc_obj,par,V,Q,l1,l2)


    ampSquared_V = (np.abs(VL)**2+np.abs(VR)**2 )/12. *(2-y1s-y2s-(y1s-y2s)**2)+y1*y2*(VL*VR.conjugate()).real
    ampSquared_T= 4./3.*(np.abs(TL)**2+np.abs(TR)**2) * (1+y1s+y2s-2*(y1s-y2s)**2) +16.*y1*y2*(TR*TL.conjugate()).real
    ampSquared_VT = 2*y1*(1+y2s-y1s)*(VR*TR.conjugate()+VL*TL.conjugate()).real + 2*y2*(1+y1s-y2s)*(VL*TR.conjugate()+VR*TL.conjugate()).real
    
    return tauV*mV/(16.*np.pi) * np.sqrt(lambda_K(1,y1s,y2s)) * (ampSquared_V+ampSquared_T+ampSquared_VT)


def Vll_br_func(V, Q, l1, l2):
    def fct(wc_obj, par):
        return Vll_br(wc_obj, par, V, Q, l1, l2)
    return fct


def Vll_br_comb_func(V, Q, l1, l2):
    def fct(wc_obj, par):
        return Vll_br(wc_obj, par, V, Q, l1, l2)+ Vll_br(wc_obj, par, V, Q, l2, l1)
    return fct

def Vll_ratio_func(V, Q, l1, l2):
    def fct(wc_obj, par):
        BRee=Vll_br(wc_obj,par,V,Q,'e','e')
        return Vll_br(wc_obj, par, V, Q, l1, l2)/BRee
    return fct


def Vll_ratio_comb_func(V, Q, l1, l2):
    def fct(wc_obj, par):
        BRee=Vll_br(wc_obj,par,V,Q,'e','e')
        return (Vll_br(wc_obj, par, V, Q, l1, l2)+ Vll_br(wc_obj, par, V, Q, l2, l1))/BRee
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


def _define_obs_V_ll(M, ll):
    _process_tex = _hadr[M]['tex']+_tex[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $V\to \ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+_hadr[M]['V']+"->"+''.join(ll)+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name

def _define_obs_V_ll_ratio(M, ll):
    _process_tex = _hadr[M]['tex']+_tex[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $V\to \ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "R("+_hadr[M]['V']+"->"+''.join(ll)+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Ratio of $"+_process_tex+r"$ over BR$(V\to ee)$")
    _obs.tex = r"$\text{R}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name


for M in _hadr:
    for (l1,l2) in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
        _obs_name = _define_obs_V_ll(M, (l1,l2))
        Prediction(_obs_name, Vll_br_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
        _obs_name = _define_obs_V_ll_ratio(M, (l1,l2))
        Prediction(_obs_name, Vll_ratio_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
    for (l1,l2) in [('e','mu'), ('e','tau'), ('mu','tau')]:
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_V_ll(M, ('{0}{1},{1}{0}'.format(l1,l2),))
        Prediction(_obs_name, Vll_br_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
        _obs_name = _define_obs_V_ll_ratio(M, ('{0}{1},{1}{0}'.format(l1,l2),))
        Prediction(_obs_name, Vll_ratio_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))

        _obs_name = _define_obs_V_ll(M, ('{1}{0},{0}{1}'.format(l1,l2),))
        Prediction(_obs_name, Vll_br_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
        _obs_name = _define_obs_V_ll_ratio(M, ('{1}{0},{0}{1}'.format(l1,l2),))
        Prediction(_obs_name, Vll_ratio_comb_func(_hadr[M]['V'], _hadr[M]['Q'], l1,l2))
    for ll in ['e', 'mu', 'tau']:
        _obs_name = _define_obs_V_ll(M, (ll,ll))
        Prediction(_obs_name, Vll_br_func(_hadr[M]['V'], _hadr[M]['Q'], ll,ll))
 
