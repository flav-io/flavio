from math import sqrt, exp
import warnings
import flavio

def omega_fct(q2, mLb, mLst):
    # eq. (73) in arXiv:2009.09313v1
    return (mLb*mLb + mLst*mLst - q2)/(2*mLb*mLst)

def ff_formula(F, A, omega):
    # eq. (75)
    return F + A*(omega - 1)

_process_dict = {}
_process_dict['Lambdab->Lambda(1520)'] = {'X': 'Lambda(1520)'}

def formfactors(process, par, q2):
    r"Formfactors for $\Lambda_b\to L(1520)$ from Lattice QCD as in arXiv:2009.09313v1"
    flavio.citations.register("Meinel:2020owd")

    pd = _process_dict[process]
    mL = par['m_'+pd['X']]
    mLb = par['m_Lambdab']

    omega = omega_fct(q2, mLb, mL)
    ff_list = ['f0', 'fplus', 'fperp', 'fperpPrim',
               'g0', 'gplus', 'gperp', 'gperpPrim',
               'hplus', 'hperp', 'hperpPrim', 'hTplus', 'hTperp', 'hTperpPrim']

    ff_dict = {}
    for e in ff_list:
        F = par[process+' '+e+' F']
        A = par[process+' '+e+' A']

        ff_dict[e] = ff_formula(F, A, omega)

    return ff_dict, mL, mLb


def ff_equiv(process, q2, par):
    # eq. (A21) - (A34) and (6)
    ff_dict, mLst, mLb = formfactors(process, par, q2)
    splus  = (mLb + mLst)**2 - q2
    sminus = (mLb - mLst)**2 - q2

    if q2 < 16.0 or q2 > (mLb - mLst)**2:
        warnings.warn(f'Lattice QCD form factors are used out of the allowed q2 region [16.0; {(mLb - mLst)**2:.3}].')

    ff = {}
    ff['fVt']    = ( mLst/splus )*ff_dict['f0']
    ff['fV0']    = ( mLst/sminus )*ff_dict['fplus']
    ff['fVperp'] = ( mLst/sminus )*ff_dict['fperp']
    ff['fVg']    = ff_dict['fperpPrim']

    ff['fAt']    = ( mLst/sminus )*ff_dict['g0']
    ff['fA0']    = ( mLst/splus )*ff_dict['gplus']
    ff['fAperp'] = ( mLst/splus )*ff_dict['gperp']
    ff['fAg']    = -ff_dict['gperpPrim']

    ff['fTt']    = 0
    ff['fT0']    = ( mLst/sminus )*ff_dict['hplus']
    ff['fTperp'] = ( mLst/sminus )*ff_dict['hperp']
    ff['fTg']    = ( mLb + mLst )*ff_dict['hperpPrim']

    ff['fT5t']   = 0
    ff['fT50']   = ( mLst/splus )*ff_dict['hTplus']
    ff['fT5perp']= ( mLst/splus )*ff_dict['hTperp']
    ff['fT5g']   = -( mLb - mLst )*ff_dict['hTperpPrim']

    return ff
