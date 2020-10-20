from math import sqrt, exp

def p_Lambda(q2, mL, mLb):
    # daughter baryon momentum in the Lb rest frame
    s = q2/mLb**2
    r = (mL/mLb)**2
    return mLb/2*sqrt((1-r)**2 - 2*(1+r)*s + s**2)
    
def m_Lambda(m_q, m_s):
    return 2*m_q + m_s

    
def alpha_ll(alpha_l1, alpha_l2):
    return sqrt((alpha_l1**2 + alpha_l2**2)/2)


def F(a0, a2, a4, p_L, m_q, m_L, alpha):
    return (a0 + a2*p_L**2 + a4*p_L**4)*exp(-3*(m_q*p_L)**2/(2*(m_L*alpha)**2))


_process_dict = {}
_process_dict['Lambdab->Lambda(1520)'] = {'X': 'Lambda(1520)', 'P': 'K-', 'q': 'b->s'}


def formfactors(process, par, q2):
    r"Functions for $\Lambda_b\to X_{3/2}$ form factors where $X_{3/2} is a spin-3/2 baryon$ using the Quark Model and the MCN approach treated in arXiv:1108.6129 [nucl-th]"

    # experimental masses because of small difference to model masses, differences will be covered by uncertainties
    pd = _process_dict[process]
    mL = par['m_'+pd['X']]
    mLb = par['m_Lambdab']
    m_q = par[process + ' m_q']
    m_s = par[process + ' m_s']
    alpha_l1 = par[process +' alpha_Lambdab']
    alpha_l2 = par[process +' alpha_'+pd['X']]

    FList = ['F1', 'F2', 'F3', 'F4', 'G1', 'G2', 'G3', 'G4', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']

    p_L = p_Lambda(q2, mL, mLb)
    mL_tilde = m_Lambda(m_q, m_s)
    alpha_LL = alpha_ll(alpha_l1, alpha_l2)

    FormDict = {}
    for e in FList:
        a0 = par[process+' '+e+' a0']
        a2 = par[process+' '+e+' a2']
        a4 = par[process+' '+e+' a4']

        FormDict[e] = F(a0, a2, a4, p_L, m_q, mL_tilde, alpha_LL)

    return FormDict, mL, mLb


def ff_equiv(process, q2, par):
    # transform FormDict in form factors used in arXiv:1903.00448
    FD, mL, mLb = formfactors(process, par, q2)

    e10 = par[process+' err_10percent']
    e30 = par[process+' err_30percent']

    ff = {}
    ff['fVt'] = ( FD['F2']*mL*(mL**2 - mLb**2 - q2) + mLb*(2*FD['F1']*mL*(mL - mLb) - 2*FD['F4']*mL*mLb + FD['F3']*(mL**2 - mLb**2 + q2)) )/( 2*mL*(mL-mLb)*mLb**2 ) * e10
    ff['fVperp'] = ( FD["F1"]/mLb - FD["F4"]*mL/(mL**2 - 2*mL*mLb + mLb**2 - q2) )*e10
    ff['fV0'] = ( FD["F2"]*mL*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)) + mLb*(2*FD["F1"]*mL*(mL + mLb)*(mL**2 - 2*mL*mLb + mLb**2 - q2) - 2*FD["F4"]*mL*mLb*(mL**2 - mLb**2 + q2) + FD["F3"]*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2))) )/( 2*mL*mLb**2*(mL+mLb)*(mL**2 - 2*mL*mLb + mLb**2 - q2) )*e10 
    ff['fVg'] = FD["F4"]*e30

    ff['fAt'] = ( FD["G2"]*mL*(mL**2 - mLb**2 - q2) + mLb*(-2*FD["G4"]*mL*mLb - 2*FD["G1"]*mL*(mL + mLb) + FD["G3"]*(mL**2 - mLb**2 + q2)) )/( 2*mL*mLb**2*(mL + mLb) )*e10
    ff['fAperp'] = ( FD["G1"]/mLb - (FD["G4"]*mL)/(mL**2 + 2*mL*mLb + mLb**2 - q2) )*e10
    ff['fA0'] = ( FD["G2"]*mL*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)) + mLb*(2*FD["G1"]*mL*(mL - mLb)*(mL**2 + 2*mL*mLb + mLb**2 - q2) - 2*FD["G4"]*mL*mLb*(mL**2 - mLb**2 + q2) + FD["G3"]*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2))) )/( 2*mL*(mL - mLb)*mLb**2*(mL**2 + 2*mL*mLb + mLb**2 - q2) )*e10 
    ff['fAg'] = -FD["G4"]*e30

    ff['fTt'] = 0*e10
    ff['fTperp'] = ( 2*FD["H5"]*mL - ((FD["H3"]+FD["H6"])*mL**2)/mLb + FD["H3"]*mLb + 2*FD["H1"]*mL*(mL + mLb)/mLb - 2*(FD["H5"] + FD["H6"])*mL**2*(mL - mLb)/((mL - mLb)**2 - q2) - FD["H3"]*q2/mLb + FD["H2"]*mL*(-mL**2 + mLb**2 + q2)/mLb**2 )/( 2*mL*(mL + mLb) )*e10
    ff['fT0'] = ( (FD["H1"] + FD["H2"] - FD["H3"] - FD["H6"])/mLb - 2*(FD["H5"] + FD["H6"]*mL)/((mL - mLb)**2 - q2) + FD["H4"]*((mL + mLb)**2 - q2)/(2*mL*mLb**2) )*e10
    ff['fTg'] = ( FD["H5"]*(mL- mLb) - FD["H6"]*(-mL**2 + mLb**2 + q2)/(2*mLb) )*e30

    ff['fT5t'] = 0*e10
    ff['fT5perp'] = ( -1/(2*mL*(mL-mLb)*mLb**2*(mL**2 + 2*mL*mLb + mLb**2 - q2)) * (FD["H2"]*mL*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)) + mLb*(mL*(2*FD["H5"]*mLb*(mL*mLb + mLb**2 - q2) + FD["H6"]*mL*(mL**2 + 2*mL*mLb + mLb**2 -q2)) - 2*FD["H1"]*mL*(mL - mLb)*(mL**2 + 2*mL*mLb + mLb**2 -q2) + FD["H3"]*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)))) )*e10
    ff['fT50'] = ( FD["H1"]/mLb + 2*FD["H5"]*mL/(mL**2 + 2*mL*mLb + mLb**2 - q2) )*e10
    ff['fT5g'] = ( -FD["H5"]*(mL + mLb) - FD["H6"]*(mL**2 + 2*mL*mLb + mLb**2 - q2)/(2*mLb) )*e30

    return ff


    
