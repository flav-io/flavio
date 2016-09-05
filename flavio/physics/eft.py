"""Classes for effective field theory (EFT) Wilson coefficients"""

import numpy as np
from flavio.physics.running import running
from flavio.physics.bdecays import rge as rge_db1
from flavio.physics.mesonmixing import rge as rge_df2

# Anomalous dimensions for DeltaF=2
def adm_df2(nf, alpha_s, alpha_e):
    return rge_df2.gamma_df2_array(nf, alpha_s)

# Anomalous dimensions for DeltaB=1
def adm_db1(nf, alpha_s, alpha_e):
    # this is the ADM for the SM basis
    A_L = rge_db1.gamma_all(nf, alpha_s, alpha_e)
    # initialize with zeros
    A = np.zeros((34,34))
    # fill in the SM ADM
    A[:15,:15] = A_L
    # the ADM for the primed SM basis is the same as for the SM
    A[15:30,15:30] = A_L
    # note that the enties 30-33 remain zero: these are the scalar and
    # pseudoscalar operators
    return A

# Anomalous dimensions for d_i->d_jlnu processes
def adm_ddlnu(nf, alpha_s, alpha_e):
    return rge_db1.gamma_fccc(alpha_s, alpha_e)

# List of all Wilson coefficients in the Standard basis

# A "sector" is a set of WCs relevant for a particular class of processes
# (e.g. b->sll) that closes under renormalization.
# Individual WCs can appear in more than one set.

coefficients = {}
adm = {}
rge_derivative = {}

_fcnc = ['bs', 'bd', 'sd', ]
_ll = ['ee', 'mumu', 'tautau']
_lilj = ['emu', 'mue', 'etau', 'taue', 'mutau', 'taumu']
_lnu = ['enu', 'munu', 'taunu']
_fccc = ['bc', 'bu', 'su', 'du', 'dc', 'sc', ]
_nunu = ['nuenue', 'numunumu', 'nutaunutau']
_nuinuj = ['nuenumu', 'numunue', 'nuenutau', 'nutaunue', 'numunutau', 'nutaunumu']

# DeltaF=2 operators
for qq in _fcnc:
    # DeltaF=2 operators
    coefficients[qq + qq] = [ 'CVLL_'+qq+qq, 'CSLL_'+qq+qq, 'CTLL_'+qq+qq,
                        'CVRR_'+qq+qq, 'CSRR_'+qq+qq, 'CTRR_'+qq+qq,
                        'CVLR_'+qq+qq, 'CSLR_'+qq+qq, ]
    adm[qq + qq] = adm_df2

    # DeltaF=1 operators
    for ll in _ll:
        coefficients[qq + ll] = [ 'C1_'+qq, 'C2_'+qq, # current-current
                            'C3_'+qq, 'C4_'+qq, 'C5_'+qq, 'C6_'+qq, # QCD penguins
                            'C7eff_'+qq, 'C8eff_'+qq, # dipoles
                            'C9_'+qq+ll, 'C10_'+qq+ll, # semi-leptonic
                            'C3Q_'+qq, 'C4Q_'+qq, 'C5Q_'+qq, 'C6Q_'+qq, 'Cb_'+qq, # EW penguins
                            # and everything with flipped chirality ...
                            'C1p_'+qq, 'C2p_'+qq,
                            'C3p_'+qq, 'C4p_'+qq, 'C5p_'+qq, 'C6p_'+qq,
                            'C7effp_'+qq, 'C8effp_'+qq,
                            'C9p_'+qq+ll, 'C10p_'+qq+ll,
                            'C3Qp_'+qq, 'C4Qp_'+qq, 'C5Qp_'+qq, 'C6Qp_'+qq, 'Cbp_'+qq,
                            # scalar and pseudoscalar
                            'CS_'+qq+ll, 'CP_'+qq+ll,
                            'CSp_'+qq+ll, 'CPp_'+qq+ll, ]
        adm[qq + ll] = adm_db1 # FIXME this is not correct yetfor s->dll

    # DeltaF=1 decays with same-flavour neutrinos in the final state
    for ll in _nunu:
        coefficients[qq + ll] = [ 'CL_'+qq+ll, 'CR_'+qq+ll, ]
        adm[qq + ll] = None # they don't run

    # DeltaF=1 decays with differently flavoured neutrinos in the final state
    for ll in _nuinuj:
        coefficients[qq + ll] = [ 'CL_'+qq+ll, 'CR_'+qq+ll, ]
        adm[qq + ll] = None # they don't run

    # DeltaF=1 LFV decays
    for ll in _lilj:
        coefficients[qq + ll] = [ 'C9_'+qq+ll, 'C10_'+qq+ll, # semi-leptonic
                            'C9p_'+qq+ll, 'C10p_'+qq+ll,
                            'CS_'+qq+ll, 'CP_'+qq+ll, # scalar and pseudoscalar
                            'CSp_'+qq+ll, 'CPp_'+qq+ll, ]
        adm[qq + ll] = None

    # tree-level weak decays
    for qq in _fccc:
        for ll in _lnu:
            coefficients[qq + ll] = [ 'CV_'+qq+ll, 'CS_'+qq+ll, 'CT_'+qq+ll,
                                'CVp_'+qq+ll, 'CSp_'+qq+ll, ]
            adm[qq + ll] = adm_ddlnu


class WilsonCoefficients(object):
    """
    """
    def __init__(self):
        self.initial = {}
        self.coefficients = coefficients
        self.all_wc = [c for v in self.coefficients.values() for c in v] # list of all coeffs

    rge_derivative = {}
    for sector in coefficients.keys():
        rge_derivative[sector] = running.make_wilson_rge_derivative(adm[sector])

    def set_initial(self, dict, scale):
        for name, value in dict.items():
            if name not in self.all_wc:
                raise KeyError("Wilson coefficient " + name + " not known")
            self.initial[name] = (scale, value)

    def get_wc(self, sector, scale, par, nf_out=None):
        # intialize with complex zeros
        values_in = np.zeros(len(self.coefficients[sector]), dtype=complex)
        # see if an initial value exists
        scale_in = None
        for idx, name in enumerate(self.coefficients[sector]):
            if name in self.initial.keys():
                scale_in_new = self.initial[name][0]
                # make sure that if there are several initial values, they are at the same scale
                if scale_in is not None and scale_in_new != scale_in:
                    raise ValueError("You cannot define initial values at different scales for Wilson coefficients that mix under renormalization")
                else:
                    scale_in = scale_in_new
                values_in[idx] = self.initial[name][1]
        if scale_in is None:
            # if no initial values have been given, no need to run anyway!
            return dict(zip(self.coefficients[sector],values_in)) # these are all zero
        if self.rge_derivative[sector] is None:
            # if the sector has vanishing anomalous dimensions, nno need to run!
            return dict(zip(self.coefficients[sector],values_in)) # these are just the initial values
        # otherwise, run
        values_out = running.get_wilson(par, values_in, self.rge_derivative[sector], scale_in, scale, nf_out=nf_out)
        return dict(zip(self.coefficients[sector],values_out))
