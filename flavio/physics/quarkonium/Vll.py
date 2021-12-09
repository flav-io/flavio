r"""$V\to ll^\prime$ branching ratio"""

#from math import pi,sqrt
from flavio.classes import Observable, Prediction
from flavio.physics.running import running
import flavio
import numpy as np

meson_quark = { 'J/psi' : 'cc'}

def kaellen(x,y,z):
    return x**2+y**2+z**2-2*(x*y+x*z+y*z)

def Vll_br(wc_obj, par,V,Q, l1,l2,label):
    r"""Branching ratio for the lepton-flavour violating leptonic decay J/psi-> l l' based on XXXX.XXXXX"""
    mV = par['m_'+V]   
    GammaV = par['Gamma_'+V]  
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    y1=ml1/mV
    y2=ml2/mV
    y1s=y1**2
    y2s=y2**2

    # renormalization scale
    scale = flavio.config['renormalization scale'][V]
    # Wilson coefficients
    wc = wc_obj.get_wc(label, scale, par)


    alphaem = running.get_alpha(par, scale)['alpha_e']
    ee=np.sqrt(4.*np.pi*alphaem) 

    fV=par['f_'+V] 
    fV_T=par['fT_'+V]
    qq=meson_quark[V]
    ll=l1+l2 
    llqq =ll+qq
    qqll=qq +ll
    VL=fV*mV*(wc['CVLL_'+llqq] + wc['CVLR_'+llqq]) 
    VR=fV*mV*(wc['CVRR_'+llqq] + wc['CVLR_'+qqll]) 
    TR=fV_T*mV*wc['CTRR_'+llqq] - ee*Q *fV*wc['Cgamma_'+l2+l1] 
    TL=(fV_T*mV*wc['CTRR_'+l1+l2+qq] - ee*Q *fV*wc['Cgamma_'+ll]).conjugate()

    ampSquared_V = (np.abs(VL)**2+np.abs(VR)**2 )/12. *(2-y1s-y2s-(y1s-y2s)**2)+y1*y2*(VL*VR.conjugate()).real
    ampSquared_T= 4./3.*(np.abs(TL)**2+np.abs(TR)**2) * (1+y1s+y2s-2*(y1s-y2s)**2) +16.*y1*y2*(TR*TL.conjugate()).real
    ampSquared_VT = 2*y1*(1+y2s-y1s)*(VR*TR.conjugate()+VL*TL.conjugate()).real + 2*y2*(1+y1s-y2s)*(VL*TR.conjugate()+VR*TL.conjugate()).real
    
    return mV/(16.*np.pi*GammaV) * np.sqrt(kaellen(1,y1s,y2s)) * (ampSquared_V+ampSquared_T+ampSquared_VT)


def Vll_br_func(V, Q, l1, l2,label):
    def fct(wc_obj, par):
        return Vll_br(wc_obj, par, V, Q, l1, l2,label)
    return fct


def Vll_br_comb_func(V, Q, l1, l2,label):
    def fct(wc_obj, par):
        return Vll_br(wc_obj, par, V, Q, l1, l2,label)+ Vll_br(wc_obj, par, V, Q, l2, l1,label)
    return fct

# Observable and Prediction instances
_hadr = { 'J/psi': {'tex': r"J/\psi\to", 'V': 'J/psi', 'Q': 2./3., }, }

_tex = {'ee': r'e^+e^-', 'mumu': r'\mu^+\mu^-', 'tautau': r'\tau^+\tau^-', 
    'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-',
    'emu,mue': r'e^\pm\mu^\mp', 'etau,taue': r'e^\pm\tau^\mp',
    'mutau,taumu': r'\mu^\pm\tau^\mp'}


def _define_obs_V_ll(M, ll):
    _process_tex = _hadr[M]['tex']+_tex[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $V\to \ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+_hadr[M]['V']+"->"+''.join(ll)+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name

for M in _hadr:
    for ll0 in [('e','mu','mue'), ('mu','e','mue'), ('e','tau','taue'), ('tau','e','taue'), ('mu','tau','mutau'), ('tau','mu','mutau')]:
        ll=(ll0[0],ll0[1])
        label=ll0[2]
        _obs_name = _define_obs_V_ll(M, ll)
        Prediction(_obs_name, Vll_br_func(_hadr[M]['V'], _hadr[M]['Q'], ll[0], ll[1],label))
    for ll0 in [('e','mu','mue'), ('e','tau','taue'), ('mu','tau','mutau')]:
        ll=(ll0[0],ll0[1])
        label=ll0[2]
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_V_ll(M, ('{0}{1},{1}{0}'.format(*ll),))
        Prediction(_obs_name, Vll_br_comb_func(_hadr[M]['V'], _hadr[M]['Q'], ll[0], ll[1],label))
