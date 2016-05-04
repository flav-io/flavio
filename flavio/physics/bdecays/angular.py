r"""Generic $B\to V \ell_1 \bar \ell_2$ helicity amplitudes and angular
distribution. Can be used for $B\to V\ell^+\ell^-$, $B\to V\ell\nu$, and
lepton flavour violating decays."""


from flavio.physics.bdecays.common import lambda_K
from math import sqrt, pi
import cmath


def transversity_to_helicity(ta):
    H={}
    H['0' ,'V'] = -1j * (ta['0_R'] + ta['0_L'])
    H['0' ,'A'] = -1j * (ta['0_R'] - ta['0_L'])
    H['pl' ,'V'] = 1j * ((ta['para_R'] + ta['para_L']) + (ta['perp_R'] + ta['perp_L']))/sqrt(2)
    H['pl' ,'A'] = 1j * ((ta['para_R'] - ta['para_L']) + (ta['perp_R'] - ta['perp_L']))/sqrt(2)
    H['mi' ,'V'] = 1j * ((ta['para_R'] + ta['para_L']) - (ta['perp_R'] + ta['perp_L']))/sqrt(2)
    H['mi' ,'A'] = 1j * ((ta['para_R'] - ta['para_L']) - (ta['perp_R'] - ta['perp_L']))/sqrt(2)
    return H

def helicity_amps_v(q2, mB, mV, mqh, mql, ml1, ml2, ff, wc, prefactor):
    laB = lambda_K(mB**2, mV**2, q2)
    H = {}
    H['0','V'] = (4 * 1j * mB * mV)/(sqrt(q2) * (mB+mV)) * ((wc['v']-wc['vp']) * (mB+mV) * ff['A12']+mqh * (wc['7']-wc['7p']) * ff['T23'])
    H['0','A'] = 4 * 1j * mB * mV/sqrt(q2) * (wc['a']-wc['ap']) * ff['A12']
    H['pl','V'] = 1j/(2 * (mB+mV)) * (+(wc['v']+wc['vp']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['v']-wc['vp']) * ff['A1'])+1j * mqh/q2 * (+(wc['7']+wc['7p']) * sqrt(laB) * ff['T1']-(wc['7']-wc['7p']) * (mB**2-mV**2) * ff['T2'])
    H['mi','V'] = 1j/(2 * (mB+mV)) * (-(wc['v']+wc['vp']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['v']-wc['vp']) * ff['A1'])+1j * mqh/q2 * (-(wc['7']+wc['7p']) * sqrt(laB) * ff['T1']-(wc['7']-wc['7p']) * (mB**2-mV**2) * ff['T2'])
    H['pl','A'] = 1j/(2 * (mB+mV)) * (+(wc['a']+wc['ap']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['a']-wc['ap']) * ff['A1'])
    H['mi','A'] = 1j/(2 * (mB+mV)) * (-(wc['a']+wc['ap']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['a']-wc['ap']) * ff['A1'])
    H['P'] = 1j * sqrt(laB)/2 * ((wc['p']-wc['pp'])/(mqh+mql)+(ml1+ml2)/q2 * (wc['a']-wc['ap'])) * ff['A0']
    H['S'] = 1j * sqrt(laB)/2 * ((wc['s']-wc['sp'])/(mqh+mql)+(ml1-ml2)/q2 * (wc['v']-wc['vp'])) * ff['A0']
    H['0','T'] = 2 * sqrt(2) * mB * mV/(mB+mV) * (wc['t']+wc['tp']) * ff['T23']
    H['0','Tt'] = 2 * mB * mV/(mB+mV) * (wc['t']-wc['tp']) * ff['T23']
    H['pl','T'] = 1/(sqrt(2) * sqrt(q2)) * (+(wc['t']-wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']+wc['tp']) * (mB**2-mV**2) * ff['T2'])
    H['mi','T'] = 1/(sqrt(2) * sqrt(q2)) * (-(wc['t']-wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']+wc['tp']) * (mB**2-mV**2) * ff['T2'])
    H['pl','Tt'] = 1/(2 * sqrt(q2)) * (+(wc['t']+wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']-wc['tp']) * (mB**2-mV**2) * ff['T2'])
    H['mi','Tt'] = 1/(2 * sqrt(q2)) * (-(wc['t']+wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']-wc['tp']) * (mB**2-mV**2) * ff['T2'])
    return {k: prefactor*v for k, v in H.items()}

def _Re(z):
    return z.real
def _Im(z):
    return z.imag
def _Co(z):
    return z.conjugate()

def angularcoeffs_general_Gbasis_v(H, q2, mB, mV, mqh, mql, ml1, ml2):
    laB = lambda_K(mB**2, mV**2, q2)
    laGa = lambda_K(q2, ml1**2, ml2**2)
    E1 = sqrt(ml1**2+laGa/(4 * q2))
    E2 = sqrt(ml2**2+laGa/(4 * q2))
    G = {}
    G[0,0,0] = (
         4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (abs(H['pl','V'])**2+abs(H['mi','V'])**2+abs(H['0','V'])**2+abs(H['pl','A'])**2+abs(H['mi','A'])**2+abs(H['0','A'])**2)
         +4 * ml1 * ml2/3 * (abs(H['pl','V'])**2+abs(H['mi','V'])**2+abs(H['0','V'])**2-abs(H['pl','A'])**2-abs(H['mi','A'])**2-abs(H['0','A'])**2)
         +4/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * abs(H['S'])**2+4/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * abs(H['P'])**2
         +16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','Tt'])**2+abs(H['mi','Tt'])**2+abs(H['0','Tt'])**2)
         +8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','T'])**2+abs(H['mi','T'])**2+abs(H['0','T'])**2)
         +16/3 * (ml1 * E2+ml2 * E1) * _Im(H['pl','V'] * _Co(H['pl','Tt'])+H['mi','V'] * _Co(H['mi','Tt'])+H['0','V'] * _Co(H['0','Tt']))
         +8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im(H['pl','A'] * _Co(H['pl','T'])+H['mi','A'] * _Co(H['mi','T'])+H['0','A'] * _Co(H['0','T'])))
    G[0,1,0] = (4 * sqrt(laGa)/3 * (
        _Re(H['pl','V'] * _Co(H['pl','A'])-H['mi','V'] * _Co(H['mi','A']))
        +2 * sqrt(2)/q2 * (ml1**2-ml2**2) * _Re(H['pl','T'] * _Co(H['pl','Tt'])-H['mi','T'] * _Co(H['mi','Tt']))
        +2 * (ml1+ml2)/sqrt(q2) * _Im(H['pl','A'] * _Co(H['pl','Tt'])-H['mi','A'] * _Co(H['mi','Tt']))
        +sqrt(2)*(ml1-ml2)/sqrt(q2) * _Im(H['pl','V'] * _Co(H['pl','T'])-H['mi','V'] * _Co(H['mi','T']))
        -(ml1-ml2)/sqrt(q2) * _Re(H['0','A'] * _Co(H['P']))-(ml1+ml2)/sqrt(q2) * _Re(H['0','V'] * _Co(H['S']))
        +_Im(sqrt(2) * H['0','T'] * _Co(H['P'])+2 * H['0','Tt'] * _Co(H['S']))
        ))
    G[0,2,0] = -2/9 * laGa/q2 * (
    -abs(H['pl','V'])**2-abs(H['mi','V'])**2+2 * abs(H['0','V'])**2-abs(H['pl','A'])**2-abs(H['mi','A'])**2+2 * abs(H['0','A'])**2
    -2 * (-abs(H['pl','T'])**2-abs(H['mi','T'])**2+2 * abs(H['0','T'])**2)-4 * (-abs(H['pl','Tt'])**2-abs(H['mi','Tt'])**2+2 * abs(H['0','Tt'])**2))
    G[2,0,0] = (-4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (abs(H['pl','V'])**2+abs(H['mi','V'])**2-2 * abs(H['0','V'])**2+abs(H['pl','A'])**2+abs(H['mi','A'])**2
    -2 * abs(H['0','A'])**2)-4 * ml1 * ml2/3 * (abs(H['pl','V'])**2+abs(H['mi','V'])**2-2 * abs(H['0','V'])**2-abs(H['pl','A'])**2
    -abs(H['mi','A'])**2+2 * abs(H['0','A'])**2)+8/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * abs(H['S'])**2
    +8/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * abs(H['P'])**2
    -16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','Tt'])**2+abs(H['mi','Tt'])**2-2 * abs(H['0','Tt'])**2)
    -8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','T'])**2+abs(H['mi','T'])**2-2 * abs(H['0','T'])**2)
    -16/3 * (ml1 * E2+ml2 * E1) * _Im(H['pl','V'] * _Co(H['pl','Tt'])+H['mi','V'] * _Co(H['mi','Tt'])-2 * H['0','V'] * _Co(H['0','Tt']))
    -8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im(H['pl','A'] * _Co(H['pl','T'])+H['mi','A'] * _Co(H['mi','T'])-2 * H['0','A'] * _Co(H['0','T'])))
    G[2,1,0] = (-4 * sqrt(laGa)/3 * (_Re(H['pl','V'] * _Co(H['pl','A'])-H['mi','V'] * _Co(H['mi','A']))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * _Re(H['pl','T'] * _Co(H['pl','Tt'])-H['mi','T'] * _Co(H['mi','Tt']))
    +2 * (ml1+ml2)/sqrt(q2) * _Im(H['pl','A'] * _Co(H['pl','Tt'])-H['mi','A'] * _Co(H['mi','Tt']))
    +sqrt(2) * (ml1-ml2)/sqrt(q2) * _Im(H['pl','V'] * _Co(H['pl','T'])-H['mi','V'] * _Co(H['mi','T']))
    +2 * (ml1-ml2)/sqrt(q2) * _Re(H['0','A'] * _Co(H['P']))+2 * (ml1+ml2)/sqrt(q2) * _Re(H['0','V'] * _Co(H['S']))
    -2 * _Im(sqrt(2) * H['0','T'] * _Co(H['P'])+2 * H['0','Tt'] * _Co(H['S']))))
    G[2,2,0] = (-2/9 * laGa/q2 * (abs(H['pl','V'])**2+abs(H['mi','V'])**2+4 * abs(H['0','V'])**2+abs(H['pl','A'])**2+abs(H['mi','A'])**2
    +4 * abs(H['0','A'])**2-2 * (abs(H['pl','T'])**2+abs(H['mi','T'])**2+4 * abs(H['0','T'])**2)-4 * (abs(H['pl','Tt'])**2+abs(H['mi','Tt'])**2+4 * abs(H['0','Tt'])**2)))
    G[2,1,1] = (4/sqrt(3) * sqrt(laGa) * (H['pl','V'] * _Co(H['0','A'])+H['pl','A'] * _Co(H['0','V'])-H['0','V'] * _Co(H['mi','A'])-H['0','A'] * _Co(H['mi','V'])
    +(ml1+ml2)/sqrt(q2) * (H['pl','V'] * _Co(H['S'])+H['S'] * _Co(H['mi','V']))-sqrt(2) * 1j * (H['P'] * _Co(H['mi','T'])-H['pl','T'] * _Co(H['P'])
    +sqrt(2)*(H['S'] * _Co(H['mi','Tt'])-H['pl','Tt'] * _Co(H['S'])))
    +(ml1-ml2)/sqrt(q2) * (H['pl','A'] * _Co(H['P'])+H['P'] * _Co(H['mi','A']))
    -2 * 1j * (ml1+ml2)/sqrt(q2) * (H['pl','A'] * _Co(H['0','Tt'])+H['0','Tt'] * _Co(H['mi','A'])-H['pl','Tt'] * _Co(H['0','A'])-H['0','A'] * _Co(H['mi','Tt']))
    -sqrt(2) * 1j * (ml1-ml2)/sqrt(q2) * (H['pl','V'] * _Co(H['0','T'])+H['0','T'] * _Co(H['mi','V'])-H['pl','T'] * _Co(H['0','V'])-H['0','V'] * _Co(H['mi','T']))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * (H['pl','T'] * _Co(H['0','Tt'])+H['pl','Tt'] * _Co(H['0','T'])-H['0','T'] * _Co(H['mi','Tt'])-H['0','Tt'] * _Co(H['mi','T']))))
    G[2,2,1] = (4/3 * laGa/q2 * (H['pl','V'] * _Co(H['0','V'])+H['0','V'] * _Co(H['mi','V'])+H['pl','A'] * _Co(H['0','A'])+H['0','A'] * _Co(H['mi','A'])
    -2 * (H['pl','T'] * _Co(H['0','T'])+H['0','T'] * _Co(H['mi','T'])+2 * (H['pl','Tt'] * _Co(H['0','Tt'])+H['0','Tt'] * _Co(H['mi','Tt'])))))
    G[2,2,2] = -8/3 * laGa/q2 * (H['pl','V'] * _Co(H['mi','V'])+H['pl','A'] * _Co(H['mi','A'])-2 * (H['pl','T'] * _Co(H['mi','T'])+2 * H['pl','Tt'] * _Co(H['mi','Tt'])))
    prefactor = sqrt(laB)*sqrt(laGa)/(2**9 * pi**3 * mB**3 * q2)
    return {k: prefactor*v for k, v in G.items()}

def angularcoeffs_h_Gbasis_v(phi, H, Htilde, q2, mB, mV, mqh, mql, ml1, ml2):
    qp = -cmath.exp(1j * phi) # here it is assumed that q/p is a pure phase, as appropriate for B and Bs mixing
    laB = lambda_K(mB**2, mV**2, q2)
    laGa = lambda_K(q2, ml1**2, ml2**2)
    E1 = sqrt(ml1**2+laGa/(4 * q2))
    E2 = sqrt(ml2**2+laGa/(4 * q2))
    G = {}
    G[0,0,0] = (
         4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','V'] * _Co(H['pl','V']))+2 * _Re(-qp * Htilde['mi','V'] * _Co(H['mi','V']))+2 * _Re(-qp * Htilde['0','V'] * _Co(H['0','V']))+2 * _Re(-qp * Htilde['pl','A'] * _Co(H['pl','A']))+2 * _Re(-qp * Htilde['mi','A'] * _Co(H['mi','A']))+2 * _Re(-qp * Htilde['0','A'] * _Co(H['0','A'])))
         +4 * ml1 * ml2/3 * (2 * _Re(-qp * Htilde['pl','V'] * _Co(H['pl','V']))+2 * _Re(-qp * Htilde['mi','V'] * _Co(H['mi','V']))+2 * _Re(-qp * Htilde['0','V'] * _Co(H['0','V']))-2 * _Re(-qp * Htilde['pl','A'] * _Co(H['pl','A']))-2 * _Re(-qp * Htilde['mi','A'] * _Co(H['mi','A']))-2 * _Re(-qp * Htilde['0','A'] * _Co(H['0','A'])))
         +4/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['S'] * _Co(H['S']))+4/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['P'] * _Co(H['P']))
         +16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','Tt'] * _Co(H['pl','Tt']))+2 * _Re(-qp * Htilde['mi','Tt'] * _Co(H['mi','Tt']))+2 * _Re(-qp * Htilde['0','Tt'] * _Co(H['0','Tt'])))
         +8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','T'] * _Co(H['pl','T']))+2 * _Re(-qp * Htilde['mi','T'] * _Co(H['mi','T']))+2 * _Re(-qp * Htilde['0','T'] * _Co(H['0','T'])))
         +16/3 * (ml1 * E2+ml2 * E1) * _Im((-qp * Htilde['pl','V']  * _Co(H['pl','Tt']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['pl','Tt']))+(-qp * Htilde['mi','V']  * _Co(H['mi','Tt']) + _Co(-qp) * H['mi','V']  * _Co(Htilde['mi','Tt']))+(-qp * Htilde['0','V']  * _Co(H['0','Tt']) + _Co(-qp) * H['0','V']  * _Co(Htilde['0','Tt'])))
         +8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im((-qp * Htilde['pl','A']  * _Co(H['pl','T']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['pl','T']))+(-qp * Htilde['mi','A']  * _Co(H['mi','T']) + _Co(-qp) * H['mi','A']  * _Co(Htilde['mi','T']))+(-qp * Htilde['0','A']  * _Co(H['0','T']) + _Co(-qp) * H['0','A']  * _Co(Htilde['0','T']))))
    G[0,1,0] = (4 * sqrt(laGa)/3 * (
        _Re((-qp * Htilde['pl','V']  * _Co(H['pl','A']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['pl','A']))-(-qp * Htilde['mi','V']  * _Co(H['mi','A']) + _Co(-qp) * H['mi','V']  * _Co(Htilde['mi','A'])))
        +2 * sqrt(2)/q2 * (ml1**2-ml2**2) * _Re((-qp * Htilde['pl','T']  * _Co(H['pl','Tt']) + _Co(-qp) * H['pl','T']  * _Co(Htilde['pl','Tt']))-(-qp * Htilde['mi','T']  * _Co(H['mi','Tt']) + _Co(-qp) * H['mi','T']  * _Co(Htilde['mi','Tt'])))
        +2 * (ml1+ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','A']  * _Co(H['pl','Tt']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['pl','Tt']))-(-qp * Htilde['mi','A']  * _Co(H['mi','Tt']) + _Co(-qp) * H['mi','A']  * _Co(Htilde['mi','Tt'])))
        +sqrt(2)*(ml1-ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','V']  * _Co(H['pl','T']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['pl','T']))-(-qp * Htilde['mi','V']  * _Co(H['mi','T']) + _Co(-qp) * H['mi','V']  * _Co(Htilde['mi','T'])))
        -(ml1-ml2)/sqrt(q2) * _Re((-qp * Htilde['0','A']  * _Co(H['P']) + _Co(-qp) * H['0','A']  * _Co(Htilde['P'])))-(ml1+ml2)/sqrt(q2) * _Re((-qp * Htilde['0','V']  * _Co(H['S']) + _Co(-qp) * H['0','V']  * _Co(Htilde['S'])))
        +_Im(sqrt(2) * (-qp * Htilde['0','T']  * _Co(H['P']) + _Co(-qp) * H['0','T']  * _Co(Htilde['P']))+2 * (-qp * Htilde['0','Tt']  * _Co(H['S']) + _Co(-qp) * H['0','Tt']  * _Co(Htilde['S'])))
        ))
    G[0,2,0] = -2/9 * laGa/q2 * (
    -2 * _Re(-qp * Htilde['pl','V'] * _Co(H['pl','V']))-2 * _Re(-qp * Htilde['mi','V'] * _Co(H['mi','V']))+2 * 2 * _Re(-qp * Htilde['0','V'] * _Co(H['0','V']))-2 * _Re(-qp * Htilde['pl','A'] * _Co(H['pl','A']))-2 * _Re(-qp * Htilde['mi','A'] * _Co(H['mi','A']))+2 * 2 * _Re(-qp * Htilde['0','A'] * _Co(H['0','A']))
    -2 * (-2 * _Re(-qp * Htilde['pl','T'] * _Co(H['pl','T']))-2 * _Re(-qp * Htilde['mi','T'] * _Co(H['mi','T']))+2 * 2 * _Re(-qp * Htilde['0','T'] * _Co(H['0','T'])))-4 * (-2 * _Re(-qp * Htilde['pl','Tt'] * _Co(H['pl','Tt']))-2 * _Re(-qp * Htilde['mi','Tt'] * _Co(H['mi','Tt']))+2 * 2 * _Re(-qp * Htilde['0','Tt'] * _Co(H['0','Tt']))))
    G[2,0,0] = (-4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','V'] * _Co(H['pl','V']))+2 * _Re(-qp * Htilde['mi','V'] * _Co(H['mi','V']))-2 * 2 * _Re(-qp * Htilde['0','V'] * _Co(H['0','V']))+2 * _Re(-qp * Htilde['pl','A'] * _Co(H['pl','A']))+2 * _Re(-qp * Htilde['mi','A'] * _Co(H['mi','A']))
    -2 * 2 * _Re(-qp * Htilde['0','A'] * _Co(H['0','A'])))-4 * ml1 * ml2/3 * (2 * _Re(-qp * Htilde['pl','V'] * _Co(H['pl','V']))+2 * _Re(-qp * Htilde['mi','V'] * _Co(H['mi','V']))-2 * 2 * _Re(-qp * Htilde['0','V'] * _Co(H['0','V']))-2 * _Re(-qp * Htilde['pl','A'] * _Co(H['pl','A']))
    -2 * _Re(-qp * Htilde['mi','A'] * _Co(H['mi','A']))+2 * 2 * _Re(-qp * Htilde['0','A'] * _Co(H['0','A'])))+8/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['S'] * _Co(H['S']))
    +8/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['P'] * _Co(H['P']))
    -16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','Tt'] * _Co(H['pl','Tt']))+2 * _Re(-qp * Htilde['mi','Tt'] * _Co(H['mi','Tt']))-2 * 2 * _Re(-qp * Htilde['0','Tt'] * _Co(H['0','Tt'])))
    -8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','T'] * _Co(H['pl','T']))+2 * _Re(-qp * Htilde['mi','T'] * _Co(H['mi','T']))-2 * 2 * _Re(-qp * Htilde['0','T'] * _Co(H['0','T'])))
    -16/3 * (ml1 * E2+ml2 * E1) * _Im((-qp * Htilde['pl','V']  * _Co(H['pl','Tt']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['pl','Tt']))+(-qp * Htilde['mi','V']  * _Co(H['mi','Tt']) + _Co(-qp) * H['mi','V']  * _Co(Htilde['mi','Tt']))-2 * (-qp * Htilde['0','V']  * _Co(H['0','Tt']) + _Co(-qp) * H['0','V']  * _Co(Htilde['0','Tt'])))
    -8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im((-qp * Htilde['pl','A']  * _Co(H['pl','T']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['pl','T']))+(-qp * Htilde['mi','A']  * _Co(H['mi','T']) + _Co(-qp) * H['mi','A']  * _Co(Htilde['mi','T']))-2 * (-qp * Htilde['0','A']  * _Co(H['0','T']) + _Co(-qp) * H['0','A']  * _Co(Htilde['0','T']))))
    G[2,1,0] = (-4 * sqrt(laGa)/3 * (_Re((-qp * Htilde['pl','V']  * _Co(H['pl','A']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['pl','A']))-(-qp * Htilde['mi','V']  * _Co(H['mi','A']) + _Co(-qp) * H['mi','V']  * _Co(Htilde['mi','A'])))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * _Re((-qp * Htilde['pl','T']  * _Co(H['pl','Tt']) + _Co(-qp) * H['pl','T']  * _Co(Htilde['pl','Tt']))-(-qp * Htilde['mi','T']  * _Co(H['mi','Tt']) + _Co(-qp) * H['mi','T']  * _Co(Htilde['mi','Tt'])))
    +2 * (ml1+ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','A']  * _Co(H['pl','Tt']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['pl','Tt']))-(-qp * Htilde['mi','A']  * _Co(H['mi','Tt']) + _Co(-qp) * H['mi','A']  * _Co(Htilde['mi','Tt'])))
    +sqrt(2) * (ml1-ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','V']  * _Co(H['pl','T']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['pl','T']))-(-qp * Htilde['mi','V']  * _Co(H['mi','T']) + _Co(-qp) * H['mi','V']  * _Co(Htilde['mi','T'])))
    +2 * (ml1-ml2)/sqrt(q2) * _Re((-qp * Htilde['0','A']  * _Co(H['P']) + _Co(-qp) * H['0','A']  * _Co(Htilde['P'])))+2 * (ml1+ml2)/sqrt(q2) * _Re((-qp * Htilde['0','V']  * _Co(H['S']) + _Co(-qp) * H['0','V']  * _Co(Htilde['S'])))
    -2 * _Im(sqrt(2) * (-qp * Htilde['0','T']  * _Co(H['P']) + _Co(-qp) * H['0','T']  * _Co(Htilde['P']))+2 * (-qp * Htilde['0','Tt']  * _Co(H['S']) + _Co(-qp) * H['0','Tt']  * _Co(Htilde['S'])))))
    G[2,2,0] = (-2/9 * laGa/q2 * (2 * _Re(-qp * Htilde['pl','V'] * _Co(H['pl','V']))+2 * _Re(-qp * Htilde['mi','V'] * _Co(H['mi','V']))+4 * 2 * _Re(-qp * Htilde['0','V'] * _Co(H['0','V']))+2 * _Re(-qp * Htilde['pl','A'] * _Co(H['pl','A']))+2 * _Re(-qp * Htilde['mi','A'] * _Co(H['mi','A']))
    +4 * 2 * _Re(-qp * Htilde['0','A'] * _Co(H['0','A']))-2 * (2 * _Re(-qp * Htilde['pl','T'] * _Co(H['pl','T']))+2 * _Re(-qp * Htilde['mi','T'] * _Co(H['mi','T']))+4 * 2 * _Re(-qp * Htilde['0','T'] * _Co(H['0','T'])))-4 * (2 * _Re(-qp * Htilde['pl','Tt'] * _Co(H['pl','Tt']))+2 * _Re(-qp * Htilde['mi','Tt'] * _Co(H['mi','Tt']))+4 * 2 * _Re(-qp * Htilde['0','Tt'] * _Co(H['0','Tt'])))))
    G[2,1,1] = (4/sqrt(3) * sqrt(laGa) * ((-qp * Htilde['pl','V']  * _Co(H['0','A']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['0','A']))+(-qp * Htilde['pl','A']  * _Co(H['0','V']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['0','V']))-(-qp * Htilde['0','V']  * _Co(H['mi','A']) + _Co(-qp) * H['0','V']  * _Co(Htilde['mi','A']))-(-qp * Htilde['0','A']  * _Co(H['mi','V']) + _Co(-qp) * H['0','A']  * _Co(Htilde['mi','V']))
    +(ml1+ml2)/sqrt(q2) * ((-qp * Htilde['pl','V']  * _Co(H['S']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['S']))+(-qp * Htilde['S']  * _Co(H['mi','V']) + _Co(-qp) * H['S']  * _Co(Htilde['mi','V'])))-sqrt(2) * 1j * ((-qp * Htilde['P']  * _Co(H['mi','T']) + _Co(-qp) * H['P']  * _Co(Htilde['mi','T']))-(-qp * Htilde['pl','T']  * _Co(H['P']) + _Co(-qp) * H['pl','T']  * _Co(Htilde['P']))
    +sqrt(2)*((-qp * Htilde['S']  * _Co(H['mi','Tt']) + _Co(-qp) * H['S']  * _Co(Htilde['mi','Tt']))-(-qp * Htilde['pl','Tt']  * _Co(H['S']) + _Co(-qp) * H['pl','Tt']  * _Co(Htilde['S']))))
    +(ml1-ml2)/sqrt(q2) * ((-qp * Htilde['pl','A']  * _Co(H['P']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['P']))+(-qp * Htilde['P']  * _Co(H['mi','A']) + _Co(-qp) * H['P']  * _Co(Htilde['mi','A'])))
    -2 * 1j * (ml1+ml2)/sqrt(q2) * ((-qp * Htilde['pl','A']  * _Co(H['0','Tt']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['0','Tt']))+(-qp * Htilde['0','Tt']  * _Co(H['mi','A']) + _Co(-qp) * H['0','Tt']  * _Co(Htilde['mi','A']))-(-qp * Htilde['pl','Tt']  * _Co(H['0','A']) + _Co(-qp) * H['pl','Tt']  * _Co(Htilde['0','A']))-(-qp * Htilde['0','A']  * _Co(H['mi','Tt']) + _Co(-qp) * H['0','A']  * _Co(Htilde['mi','Tt'])))
    -sqrt(2) * 1j * (ml1-ml2)/sqrt(q2) * ((-qp * Htilde['pl','V']  * _Co(H['0','T']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['0','T']))+(-qp * Htilde['0','T']  * _Co(H['mi','V']) + _Co(-qp) * H['0','T']  * _Co(Htilde['mi','V']))-(-qp * Htilde['pl','T']  * _Co(H['0','V']) + _Co(-qp) * H['pl','T']  * _Co(Htilde['0','V']))-(-qp * Htilde['0','V']  * _Co(H['mi','T']) + _Co(-qp) * H['0','V']  * _Co(Htilde['mi','T'])))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * ((-qp * Htilde['pl','T']  * _Co(H['0','Tt']) + _Co(-qp) * H['pl','T']  * _Co(Htilde['0','Tt']))+(-qp * Htilde['pl','Tt']  * _Co(H['0','T']) + _Co(-qp) * H['pl','Tt']  * _Co(Htilde['0','T']))-(-qp * Htilde['0','T']  * _Co(H['mi','Tt']) + _Co(-qp) * H['0','T']  * _Co(Htilde['mi','Tt']))-(-qp * Htilde['0','Tt']  * _Co(H['mi','T']) + _Co(-qp) * H['0','Tt']  * _Co(Htilde['mi','T'])))))
    G[2,2,1] = (4/3 * laGa/q2 * ((-qp * Htilde['pl','V']  * _Co(H['0','V']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['0','V']))+(-qp * Htilde['0','V']  * _Co(H['mi','V']) + _Co(-qp) * H['0','V']  * _Co(Htilde['mi','V']))+(-qp * Htilde['pl','A']  * _Co(H['0','A']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['0','A']))+(-qp * Htilde['0','A']  * _Co(H['mi','A']) + _Co(-qp) * H['0','A']  * _Co(Htilde['mi','A']))
    -2 * ((-qp * Htilde['pl','T']  * _Co(H['0','T']) + _Co(-qp) * H['pl','T']  * _Co(Htilde['0','T']))+(-qp * Htilde['0','T']  * _Co(H['mi','T']) + _Co(-qp) * H['0','T']  * _Co(Htilde['mi','T']))+2 * ((-qp * Htilde['pl','Tt']  * _Co(H['0','Tt']) + _Co(-qp) * H['pl','Tt']  * _Co(Htilde['0','Tt']))+(-qp * Htilde['0','Tt']  * _Co(H['mi','Tt']) + _Co(-qp) * H['0','Tt']  * _Co(Htilde['mi','Tt']))))))
    G[2,2,2] = -8/3 * laGa/q2 * ((-qp * Htilde['pl','V']  * _Co(H['mi','V']) + _Co(-qp) * H['pl','V']  * _Co(Htilde['mi','V']))+(-qp * Htilde['pl','A']  * _Co(H['mi','A']) + _Co(-qp) * H['pl','A']  * _Co(Htilde['mi','A']))-2 * ((-qp * Htilde['pl','T']  * _Co(H['mi','T']) + _Co(-qp) * H['pl','T']  * _Co(Htilde['mi','T']))+2 * (-qp * Htilde['pl','Tt']  * _Co(H['mi','Tt']) + _Co(-qp) * H['pl','Tt']  * _Co(Htilde['mi','Tt']))))
    prefactor = sqrt(laB)*sqrt(laGa)/(2**9 * pi**3 * mB**3 * q2)
    return {k: prefactor*v for k, v in G.items()}

def G_to_g(G):
    g = {}
    g['1s'] = 1/32 * (8 * G[0,0,0] + 2 * G[0,2,0] - 4 * G[2,0,0] - G[2,2,0] )
    g['1c'] = 1/16 * (4 * G[0,0,0] +  G[0,2,0] + 4 * G[2,0,0] + G[2,2,0] )
    g['2s'] = 3/32 * ( 2 * G[0,2,0] - G[2,2,0] )
    g['2c'] = 3/16 * (G[0,2,0] + G[2,2,0] )
    g['6s'] = 1/8 * ( 2 * G[0,1,0] - G[2,1,0] )
    g['6c'] = 1/4 * ( G[0,1,0] + G[2,1,0] )
    g[3] = 3/32 * _Re(G[2,2,2])
    g[4] = 3/32 * _Re(G[2,2,1])
    g[5] = sqrt(3)/16 * _Re(G[2,1,1])
    g[7] = sqrt(3)/16 * _Im(G[2,1,1])
    g[8] = 3/32 * _Im(G[2,2,1])
    g[9] = 3/32 * _Im(G[2,2,2])
    return g

def angularcoeffs_general_v(*args, **kwargs):
    G = angularcoeffs_general_Gbasis_v(*args, **kwargs)
    g = G_to_g(G)
    signflip = [4, '6s', '6c', 7, 9]
    J = {k: -8*4/3.*g[k] if k in signflip else 8*4/3.*g[k] for k in g}
    return J

def angularcoeffs_h_v(*args, **kwargs):
    h = angularcoeffs_h_Gbasis_v(*args, **kwargs)
    g_h = G_to_g(h)
    signflip = [4, '6s', '6c', 7, 9]
    J_h = {k: -8*4/3.*g_h[k] if k in signflip else 8*4/3.*g_h[k] for k in g_h}
    return J_h

def helicity_amps_p(q2, mB, mP, mqh, mql, ml1, ml2, ff, wc, prefactor):
    laB = lambda_K(mB**2, mP**2, q2)
    h = {}
    h['V'] = sqrt(laB)/(2*sqrt(q2)) * (
        2*mqh/(mB+mP)*(wc['7']+wc['7p'])*ff['fT']+(wc['v']+wc['vp'])*ff['f+'] )
    h['A'] = sqrt(laB)/(2*sqrt(q2)) * (wc['a']+wc['ap'])*ff['f+']
    h['S'] = (mB**2-mP**2)/2. * ff['f0'] * (
            (wc['s']+wc['sp'])/(mqh-mql) + (ml1-ml2)/q2*(wc['v']+wc['vp']) )
    h['P'] = (mB**2-mP**2)/2. * ff['f0'] * (
            (wc['p']+wc['pp'])/(mqh-mql) + (ml1+ml2)/q2*(wc['a']+wc['ap']) )
    h['T']  = -1j*sqrt(laB)/(2*(mB+mP)) * (wc['t']-wc['tp']) * ff['fT']
    h['Tt'] = -1j*sqrt(laB)/(2*(mB+mP)) * (wc['t']+wc['tp']) * ff['fT']
    return {k: prefactor*v for k, v in h.items()}

def angularcoeffs_general_Gbasis_p(h, q2, mB, mP, mqh, mql, ml1, ml2):
        laB = lambda_K(mB**2, mP**2, q2)
        laGa = lambda_K(q2, ml1**2, ml2**2)
        E1 = sqrt(ml1**2+laGa/(4 * q2))
        E2 = sqrt(ml2**2+laGa/(4 * q2))
        G = {}
        G[0] = (
              ( 4*(E1*E2 + ml1*ml2) + laGa/(3*q2) ) * abs(h['V'])**2
            + ( 4*(E1*E2 - ml1*ml2) + laGa/(3*q2) ) * abs(h['A'])**2
            + ( 4*(E1*E2 - ml1*ml2) + laGa/(  q2) ) * abs(h['S'])**2
            + ( 4*(E1*E2 + ml1*ml2) + laGa/(  q2) ) * abs(h['P'])**2
            +  16*(E1*E2 + ml1*ml2  - laGa/(12*q2)) * abs(h['Tt'])**2
            +   8*(E1*E2 - ml1*ml2  - laGa/(12*q2)) * abs(h['T'])**2
            +      16 * (ml1*E2 + ml2*E1) * _Im( h['V'] * _Co(h['Tt']) )
            + 8*sqrt(2)*(ml1*E2 - ml2*E1) * _Im( h['A'] * _Co(h['T']) ) )
        G[1] = -4*sqrt(laGa) * (
              _Re(   (ml1+ml2)/sqrt(q2) * h['V'] * _Co(h['S'])
                   + (ml1-ml2)/sqrt(q2) * h['A'] * _Co(h['P']) )
            - _Im( 2 * h['Tt'] * _Co(h['S']) + sqrt(2) * h['T'] * _Co(h['P'])) )
        G[2] = -4*laGa/(3*q2) * (
            abs(h['V'])**2 + abs(h['A'])**2 - 2*abs(h['T'])**2 - 4*abs(h['Tt'])**2 )
        prefactor = sqrt(laB)*sqrt(laGa)/(2**9 * pi**3 * mB**3 * q2)
        return {k: prefactor*v for k, v in G.items()}

def angularcoeffs_general_p(*args, **kwargs):
    G = angularcoeffs_general_Gbasis_p(*args, **kwargs)
    J = {}
    J['a'] = G[0] - G[2]/2.
    J['b'] = G[1]
    J['c'] = 3*G[2]/2.
    return J
