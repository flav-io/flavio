from math import sqrt, exp

def omega(q2, mLb, mLst):
    # eq. (73) in arXiv:2009.09313v1
    return (mLb*mLb + m_Lst*m_Lst - q2)/(2*mLb*m_Lst)

def F(q2, Ff, Af, Omega):
    # eq. (75) 
    return Ff + Af*(Omega - 1)

_process_dict = {}
_process_dict['Lambdab->Lambda(1520)'] = {'X': 'Lambda(1520)'}

def formfactors(process, par, q2):
    r"Formfactors for $\Lambda_b\to L(1520)$ from Lattice QCD as in arXiv:2009.09313v1"
    pd = _process_dict[process]
    mL = par['m_'+pd['X']]
    mLb = par['m_Lambdab']

    Omega = omega(q2, mLb, mL)
    FList = ['f0', 'fplus', 'fperp', 'fperpPrim',
             'g0', 'gplus', 'gperp', 'gperpPrim',
             'hplus', 'hperp', 'hperpPrim', 'hTplus', 'hTperp', 'hTperpPrim']

    FormDict = {}
    for e in FList:
        Ff = par[process+' '+e+' F']
        Af = par[process+' '+e+' A']

        FormDict[e] = F(q2, Ff, Af, Omega)

    return FormDict, mL, mLb


def ff_equiv(process, q2, par):
    # eq. (A21) - (A34) and (6)
    FD, mLst, mLb = formfactors(process, par, q2)

    splus  = (mLb + mLst)**2 - q2
    sminus = (mLb - mLst)**2 - q2
    
    ff = {}
    ff['fVt']    = ( mLst/splus )*FD['f0']
    ff['fV0']    = ( mLst/sminus )*FD['fplus']
    ff['fVperp'] = ( mLst/sminus )*FD['fperp']
    ff['fVg']    = FD['fperpPrim']

    ff['fAt']    = ( mLst/sminus )*FD['g0']
    ff['fA0']    = ( mLst/splus )*FD['gplus']
    ff['fAperp'] = ( mLst/splus )*FD['gperp']
    ff['fAg']    = -FD['gperpPrim']

    ff['fTt']    = 0
    ff['fT0']    = ( mLst/sminus )*FD['hplus']
    ff['fTperp'] = ( mLst/sminus )*FD['hperp']
    ff['fTg']    = ( mLb + mLst )*FD['hperpPrim']

    ff['fT5t']   = 0
    ff['fT50']   = ( mLst/splus )*FD['hTplus']
    ff['fT5perp']= ( mLst/splus )*FD['hTperp']
    ff['fT5g']   = -( mLb - mLst )*FD['hTperpPrim']
    
    return ff
