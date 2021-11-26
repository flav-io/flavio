r"""$B_c$ lifetime"""

import flavio

def kaellen(x,y,z):
    return x**2+y**2+z**2-2*(x*y+x*z+y*z)

def Vll_br(wc_obj, par,V,l1,l2):
    r"""Branching ratio for the lepton-flavour violating leptonic decay J/psi-> l l' based on XXXX.XXXXX"""
    mV = par['m_'+V]   # where is the J/psi mass stored 
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    y1=ml1/mV
    y2=ml2/mV

    y1s=y1**2
    y2s=y2**2

    fV=1.  # form factors still have to be implemented
    fV_T=fV # NRCSM

    VL=fV*mV*1 # wc_obj[''] # Wilson coefficients have to be implemented
    VR=fV*mV*1 # wc_obj['']
    TL=fV_T*mV*1 # wc_obj['']
    TR=fV_T*mV*1 # wc_obj['']

    ampSquared_V = (np.abs(VL)**2+np.abs(VR)**2 )/24. *(3-2*y1s-2*y2s+2*y1s*y2s-y1s**2-y2s**2)+y1*y2*(VL*VR.conjugate()).real
    ampSquared_T= 4./3.*(np.abs(TL)**2+np.abs(TR)**2) * (1+y1s+y2s-2*y1s**2-2*y2s**2+4*y1s*y2s) +16.*(TR*TL.conjugate()).real
    ampSquared_VT = -2*y1*(1+y2s-y1s)*(VR*TR.conjugate()+VL*TL.conjugate()).real - 2*y2*(1+y1s-y2s)*(VL*TR.conjugate()+VR*TL.conjugate()).real
    
    return mV/(16.*np.pi) * np.sqrt(kaellen(1,y1s,y2s)) * (ampSquared_V+ampSquared_T+ampSquared_VT)


def Vll_br_func(V, l1, l2):
    def fct(wc_obj, par):
        return Vll_br(wc_obj, par, V, l1, l2)
    return fct


def Vll_br_comb_func(V, l1, l2):
    def fct(wc_obj, par):
        return Vll_br(wc_obj, par, V, l1, l2)+ Vll_br(wc_obj, par, V, l2, l1)
    return fct

# Observable and Prediction instances
_hadr_lfv = {
'J/psi': {'tex': r"J/\psi\to", 'V': 'J/psi', },
}
_tex_lfv = {'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-',
    'emu,mue': r'e^\pm\mu^\mp', 'etau,taue': r'e^\pm\tau^\mp',
    'mutau,taumu': r'\mu^\pm\tau^\mp'}


def _define_obs_V_ll(M, ll):
    _process_tex = _hadr_lfv[M]['tex']+' '+_tex_lfv[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $V\to \ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+M+''.join(ll)+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name

for M in _hadr_lfv:
    for ll in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
        _obs_name = _define_obs_V_ll(M, ll)
        Prediction(_obs_name, Vll_br_func(_hadr_lfv[M]['V'], ll[0], ll[1]))
    for ll in [('e','mu'), ('e','tau'), ('mu','tau')]:
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_V_ll(M, ('{0}{1},{1}{0}'.format(*ll),))
        Prediction(_obs_name, VLL_br_comb_func(_hadr_lfv[M]['V'], ll[0], ll[1]))