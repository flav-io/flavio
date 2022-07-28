r"""$P\to ll^\prime$ branching ratio"""
# TODO add reference

#from math import pi,sqrt
from flavio.classes import Observable, Prediction
import flavio
from flavio.physics.running import running
import numpy as np

meson_quark = { 'eta_c' : 'cc', 
                'eta_b': 'bb',
                }

def kaellen(x,y,z):
    return x**2+y**2+z**2-2*(x*y+x*z+y*z)

def getS_lfv(wc_obj,par,P,l1,l2,wc_sector,CeGGij,CeGGji):
    # renormalization scale
    scale = flavio.config['renormalization scale'][P]
    alphas = running.get_alpha_s(par, scale)
    # Wilson coefficients
    wc = wc_obj.get_wc(wc_sector, scale, par)

    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mP = par['m_'+P]   
    fP=par['f_'+P] 
    aP=par['a_'+P]
    hP= mP**2 * fP -aP
    qq=meson_quark[P]
    mq=par['m_'+qq[0]]
    # for emu and taue the name of the Wilson coefficient sector agrees with the ordering of leptons in the vector bilinear
    # This is not the case for mutau. Thus distinguish between the two cases here.
    if wc_sector=="mutau":
        ll="taumu"
    else:
        ll=wc_sector
    # CHECK print([ml1,ml2,mP,aP,hP,mq])
    aPfac=1j*aP*4.*np.pi/alphas
    SR= aPfac * CeGGij +  (hP/(4*mq))*(wc['CSRR_'+l1+l2+qq]-wc['CSRL_'+l1+l2+qq])  -(fP/2.)*(ml1*(wc['CVRR_'+ll+qq] - wc['CVLR_'+qq+ll])+ml2* (wc['CVLR_'+ll+qq] -wc['CVLL_'+ll+qq] ))  
    SL= aPfac * CeGGji + (hP/(4*mq))*(wc['CSRL_'+l2+l1+qq]-wc['CSRR_'+l2+l1+qq]).conjugate() -(fP/2.)*(ml2*(wc['CVRR_'+ll+qq] - wc['CVLR_'+qq+ll])+ml1* (wc['CVLR_'+ll+qq] -wc['CVLL_'+ll+qq] )) 
#    print("[SR,SL,CSRR] = ",[SR,SL,hP,mq,(hP/(4*mq))*(wc['CSRR_'+l1+l2+qq]-wc['CSRL_'+l1+l2+qq])])
    return SL,SR


def Pll_br(wc_obj, par,P,Q, l1,l2,wc_sector,CeGGij,CeGGji):
    r"""Branching ratio for the lepton-flavour violating leptonic decay P -> l l' """
    #####branching ratio obtained from 2207.10913#####
    flavio.citations.register("Calibbi:2022ddo")
    mP = par['m_'+P]   
    GammaP = par['Gamma_'+P]  
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    y1=ml1/mP
    y2=ml2/mP
    y1s=y1**2
    y2s=y2**2
    SL,SR = getS_lfv(wc_obj,par,P,l1,l2,wc_sector,CeGGij,CeGGji)
    return  mP/(16.*np.pi*GammaP) * np.sqrt(kaellen(1,y1s,y2s)) * ((1-y1s-y2s)*(np.abs(SL)**2+np.abs(SR)**2) -4*y1*y2 *(SL*SR.conjugate()).real)


def Pll_br_func(P, Q, l1, l2,wc_sector):
    def fct(wc_obj, par,CeGGij=0,CeGGji=0):
        return Pll_br(wc_obj, par, P, Q, l1, l2,wc_sector,CeGGij,CeGGji)
    return fct


def Pll_br_comb_func(P, Q, l1, l2,wc_sector):
    def fct(wc_obj, par,CeGGij=0,CeGGji=0):
        return Pll_br(wc_obj, par, P, Q, l1, l2,wc_sector,CeGGij,CeGGji)+ Pll_br(wc_obj, par, P, Q, l2, l1,wc_sector,CeGGij,CeGGji)
    return fct


# Observable and Prediction instances
_hadr = { 
    'eta_c': {'tex': r"\eta_c\to", 'P': 'eta_c', 'Q': 2./3., },
    'eta_b': {'tex': r"\eta_b\to", 'P': 'eta_b', 'Q': -1./3., },
 }

_tex = {'ee': r'e^+e^-', 'mumu': r'\mu^+\mu^-', 'tautau': r'\tau^+\tau^-', 
    'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-',
    'emu,mue': r'e^\pm\mu^\mp', 'etau,taue': r'e^\pm\tau^\mp','mutau,taumu': r'\mu^\pm\tau^\mp',
    'mue,emu': r'e^\pm\mu^\mp', 'taue,etau': r'e^\pm\tau^\mp','taumu,mutau': r'\mu^\pm\tau^\mp'
    }


def _define_obs_P_ll(M, ll):
    _process_tex = _hadr[M]['tex']+_tex[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $P\to \ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+_hadr[M]['P']+"->"+''.join(ll)+")"
    _obs = Observable(_obs_name,arguments=["CeGGij","CeGGji"])
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name


for M in _hadr:
    for ll0 in [('e','mu','mue'), ('mu','e','mue'), ('e','tau','taue'), ('tau','e','taue'), ('mu','tau','mutau'), ('tau','mu','mutau')]:
        ll=(ll0[0],ll0[1])
        wc_sector=ll0[2]
        _obs_name = _define_obs_P_ll(M, ll)
        Prediction(_obs_name, Pll_br_func(_hadr[M]['P'], _hadr[M]['Q'], ll[0], ll[1],wc_sector))
    for ll0 in [('e','mu','mue'), ('e','tau','taue'), ('mu','tau','mutau')]:
        ll=(ll0[0],ll0[1])
        wc_sector=ll0[2]
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_P_ll(M, ('{0}{1},{1}{0}'.format(*ll),))
        Prediction(_obs_name, Pll_br_comb_func(_hadr[M]['P'], _hadr[M]['Q'], ll[0], ll[1],wc_sector))
        _obs_name = _define_obs_P_ll(M, ('{1}{0},{0}{1}'.format(*ll),))
        Prediction(_obs_name, Pll_br_comb_func(_hadr[M]['P'], _hadr[M]['Q'], ll[0], ll[1],wc_sector))
 