from math import sqrt, exp
import flavio

def lambda_momentum(q2, mL, mLb):
    # daughter baryon momentum in the Lb rest frame
    s = q2/(mLb**2)
    r = (mL/mLb)**2
    phi = (1-r)**2 - 2*(1+r)*s + s**2
    if phi > 0 :
        return mLb/2*sqrt(phi)
    else :
        return mLb/2*sqrt(-phi)

def lambda_mass(m_q, m_s):
    return 2*m_q + m_s


def alpha_lambda_lambdaprime(alpha_l1, alpha_l2):
    return sqrt((alpha_l1**2 + alpha_l2**2)/2)


def F(a0, a2, a4, p_L, m_q, m_L, alpha):
    return (a0 + a2*p_L**2 + a4*p_L**4)*exp(-3*(m_q*p_L)**2/(2*(m_L*alpha)**2))


_process_dict = {}
_process_dict['Lambdab->Lambda(1520)'] = {'X': 'Lambda(1520)'}


def formfactors(process, par, q2):
    r"Functions for $\Lambda_b\to X_{3/2}$ form factors where $X_{3/2} is a spin-3/2 baryon$ using the Quark Model and the MCN approach treated in arXiv:1108.6129 [nucl-th]"
    flavio.citations.register("Mott:2011cx")

    # Using the PDG mass values instead the model ones, will be covered by the uncertainties attached to the form factors.
    pd = _process_dict[process]
    mL = par['m_Lambda(1520)']
    mLb = par['m_Lambdab']
    m_q = par[process + ' m_q']
    m_s = par[process + ' m_s']
    alpha_l1 = par[process +' alpha_Lambdab']
    alpha_l2 = par[process +' alpha_'+pd['X']]

    ff_list = ['F1', 'F2', 'F3', 'F4',
               'G1', 'G2', 'G3', 'G4',
               'H1', 'H2', 'H3', 'H4', 'H5', 'H6']

    p_lambda = lambda_momentum(q2, mL, mLb)
    m_lambda_tilde = lambda_mass(m_q, m_s)
    alpha_ll = alpha_lambda_lambdaprime(alpha_l1, alpha_l2)

    ff_dict = {}
    for e in ff_list:
        a0 = par[process+' '+e+' a0']
        a2 = par[process+' '+e+' a2']
        a4 = par[process+' '+e+' a4']

        ff_dict[e] = F(a0, a2, a4, p_lambda, m_q, m_lambda_tilde, alpha_ll)

    return ff_dict, mL, mLb


def ff_equiv(process, q2, par):
    # transform FormDict in form factors used in arXiv:1903.00448
    ff_dict, mL, mLb = formfactors(process, par, q2)

    e_fVt = par[process+' fVt uncertainty']
    e_fVperp = par[process+' fVperp uncertainty']
    e_fV0 = par[process+' fV0 uncertainty']
    e_fVg = par[process+' fVg uncertainty']
    e_fAt = par[process+' fAt uncertainty']
    e_fAperp = par[process+' fAperp uncertainty']
    e_fA0 = par[process+' fA0 uncertainty']
    e_fAg = par[process+' fAg uncertainty']
    e_fTt = par[process+' fTt uncertainty']
    e_fTperp = par[process+' fTperp uncertainty']
    e_fT0 = par[process+' fT0 uncertainty']
    e_fTg = par[process+' fTg uncertainty']
    e_fT5t = par[process+' fT5t uncertainty']
    e_fT5perp = par[process+' fT5perp uncertainty']
    e_fT50 = par[process+' fT50 uncertainty']
    e_fT5g = par[process+' fT5g uncertainty']

    ff = {}
    ff['fVt'] = ( ff_dict['F2']*mL*(mL**2 - mLb**2 - q2) + mLb*(2*ff_dict['F1']*mL*(mL - mLb) - 2*ff_dict['F4']*mL*mLb + ff_dict['F3']*(mL**2 - mLb**2 + q2)) )/( 2*mL*(mL-mLb)*mLb**2 ) * e_fVt
    ff['fVperp'] = ( ff_dict["F1"]/mLb - ff_dict["F4"]*mL/(mL**2 - 2*mL*mLb + mLb**2 - q2) )*e_fVperp
    ff['fV0'] = ( ff_dict["F2"]*mL*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)) + mLb*(2*ff_dict["F1"]*mL*(mL + mLb)*(mL**2 - 2*mL*mLb + mLb**2 - q2) - 2*ff_dict["F4"]*mL*mLb*(mL**2 - mLb**2 + q2) + ff_dict["F3"]*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2))) )/( 2*mL*mLb**2*(mL+mLb)*(mL**2 - 2*mL*mLb + mLb**2 - q2) )*e_fV0
    ff['fVg'] = ff_dict["F4"]*e_fVg

    ff['fAt'] = ( ff_dict["G2"]*mL*(mL**2 - mLb**2 - q2) + mLb*(-2*ff_dict["G4"]*mL*mLb + 2*ff_dict["G1"]*mL*(mL + mLb) + ff_dict["G3"]*(mL**2 - mLb**2 + q2)) )/( 2*mL*mLb**2*(mL + mLb) )*e_fAt
    ff['fAperp'] = ( ff_dict["G1"]/mLb - (ff_dict["G4"]*mL)/(mL**2 + 2*mL*mLb + mLb**2 - q2) )*e_fAperp
    ff['fA0'] = ( ff_dict["G2"]*mL*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)) + mLb*(2*ff_dict["G1"]*mL*(mL - mLb)*(mL**2 + 2*mL*mLb + mLb**2 - q2) - 2*ff_dict["G4"]*mL*mLb*(mL**2 - mLb**2 + q2) + ff_dict["G3"]*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2))) )/( 2*mL*(mL - mLb)*mLb**2*(mL**2 + 2*mL*mLb + mLb**2 - q2) )*e_fA0
    ff['fAg'] = -ff_dict["G4"]*e_fAg

    ff['fTt'] = 0*e_fTt
    ff['fTperp'] = ( 2*ff_dict["H5"]*mL - ((ff_dict["H3"]+ff_dict["H6"])*mL**2)/mLb + ff_dict["H3"]*mLb + 2*ff_dict["H1"]*mL*(mL + mLb)/mLb - 2*(ff_dict["H5"] + ff_dict["H6"])*mL**2*(mL - mLb)/((mL - mLb)**2 - q2) - ff_dict["H3"]*q2/mLb + ff_dict["H2"]*mL*(-mL**2 + mLb**2 + q2)/mLb**2 )/( 2*mL*(mL + mLb) )*e_fTperp
    ff['fT0'] = ( (ff_dict["H1"] + ff_dict["H2"] - ff_dict["H3"] - ff_dict["H6"])/mLb - 2*((ff_dict["H5"] + ff_dict["H6"])*mL)/((mL - mLb)**2 - q2) + ff_dict["H4"]*((mL + mLb)**2 - q2)/(2*mL*mLb**2) )*e_fT0
    ff['fTg'] = ( ff_dict["H5"]*(mL- mLb) - ff_dict["H6"]*(-mL**2 + mLb**2 + q2)/(2*mLb) )*e_fTg

    ff['fT5t'] = 0*e_fT5t
    ff['fT5perp'] = ( -1/(2*mL*(mL-mLb)*mLb**2*(mL**2 + 2*mL*mLb + mLb**2 - q2)) * (ff_dict["H2"]*mL*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)) + mLb*(mL*(2*ff_dict["H5"]*mLb*(mL*mLb + mLb**2 - q2) + ff_dict["H6"]*mL*(mL**2 + 2*mL*mLb + mLb**2 -q2)) - 2*ff_dict["H1"]*mL*(mL - mLb)*(mL**2 + 2*mL*mLb + mLb**2 -q2) + ff_dict["H3"]*(mL**4 + (mLb**2 - q2)**2 - 2*mL**2*(mLb**2 + q2)))) )*e_fT5perp
    ff['fT50'] = ( ff_dict["H1"]/mLb + 2*ff_dict["H5"]*mL/(mL**2 + 2*mL*mLb + mLb**2 - q2) )*e_fT50
    ff['fT5g'] = ( -ff_dict["H5"]*(mL + mLb) - ff_dict["H6"]*(mL**2 + 2*mL*mLb + mLb**2 - q2)/(2*mLb) )*e_fT5g

    return ff
