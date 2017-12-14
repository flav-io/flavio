"""Classes for effective field theory (EFT) Wilson coefficients"""

import numpy as np
from flavio.physics.running import running
from flavio.physics.bdecays import rge as rge_db1
from flavio.physics.mesonmixing import rge as rge_df2
import warnings


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
_lnu = ['enue', 'munumu', 'taunutau',
        'enumu', 'enutau', 'munue', 'munutau', 'taunue', 'taunumu']
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
                            'C7_'+qq, 'C8_'+qq, # dipoles
                            'C9_'+qq+ll, 'C10_'+qq+ll, # semi-leptonic
                            'C3Q_'+qq, 'C4Q_'+qq, 'C5Q_'+qq, 'C6Q_'+qq, 'Cb_'+qq, # EW penguins
                            # and everything with flipped chirality ...
                            'C1p_'+qq, 'C2p_'+qq,
                            'C3p_'+qq, 'C4p_'+qq, 'C5p_'+qq, 'C6p_'+qq,
                            'C7p_'+qq, 'C8p_'+qq,
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
            coefficients[qq + ll] = [ 'CVL_'+qq+ll, 'CSR_'+qq+ll, 'CT_'+qq+ll,
                                'CVR_'+qq+ll, 'CSL_'+qq+ll, ]
            adm[qq + ll] = adm_ddlnu


class WilsonCoefficients(object):
    """Class representing a point in the EFT parameter space and giving
    access to RG evolution.

    Note that all Wilson coefficient values refer to new physics contributions
    only, i.e. they vanish in the SM.

    Methods:

    - set_initial: set the initial values of Wilson coefficients at some scale
    - set_initial_wcxf: set the initial values from a wcxf.WC instance
    - get_wc: get the values of the Wilson coefficients at some scale
    """
    def __init__(self):
        self.initial = {}
        self.coefficients = coefficients
        self.all_wc = [c for v in self.coefficients.values() for c in v] # list of all coeffs

    rge_derivative = {}
    for sector in coefficients.keys():
        rge_derivative[sector] = running.make_wilson_rge_derivative(adm[sector])

    def set_initial(self, dict, scale):
        """Set initial values of Wilson coefficients.

        Parameters:

        - dict: dictionary where keys are Wilson coefficient name strings and
          values are Wilson coefficient NP contribution values
        - scale: $\overline{\text{MS}}$ renormalization scale
        """
        for name, value in dict.items():
            if name.split('_')[0] in ['C7eff', 'C7effp', 'C8eff', 'C8effp']:
                raise KeyError("The dipole Wilson coefficients like " + name
                               + "have been renamed in v0.25.")
            elif (name.split('_')[0] in ['CV', 'CVp', 'CS', 'CSp', 'CT']
                  and name[-2:] == 'nu'):
                raise KeyError("The semileptonic Wilson coefficients " + name
                               + " have been redefined in v0.25.")
            elif name not in self.all_wc:
                raise KeyError("Wilson coefficient " + name + " not known")
            # warn for four-quark operators that will be changed soon
            elif name.split('_')[0] in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                                        'C1p', 'C2p', 'C3p', 'C4p', 'C5p', 'C6p',
                                        'C3Q', 'C4Q', 'C5Q', 'C6Q',
                                        'C3Qp', 'C4Qp', 'C5Qp', 'C6Qp',
                                        ]:
                warnings.warn("New physics in four-quark operators "
                              "has been deprecated in v0.25. It will be "
                              "reimplemented at a later stage")
            self.initial[name] = (scale, value)

    def set_initial_wcxf(self, wc):
        """Set initial values of Wilson coefficients from a WCxf WC instance.

        If the instance is given in a basis other than the flavio basis,
        the translation is performed automatically, if implemented in the
        `wcxf` package."""
        import wcxf
        if not isinstance(wc, wcxf.WC):
            raise ValueError("`wc` should be an instance of `wcxf.WC`")
        if wc.eft != 'WET':
            raise NotImplementedError("Matching from a different EFT is currently not implemented.")
        if wc.basis == 'flavio':
            wc_dict = wc.dict
        else:
            wc_trans = wc.translate('flavio')
            wc_dict = wc_trans.dict
        self.set_initial(wc_dict, wc.scale)

    def get_wc(self, sector, scale, par, nf_out=None):
        """Get the values of the Wilson coefficients belonging to a specific
        sector (e.g. `bsmumu`) at a given scale.

        Returns a dictionary of WC values.

        Parameters:

        - sector: string name of the sector
        - scale: $\overline{\text{MS}}$ renormalization scale
        - par: dictionary of parameters
        - nf_out: number of quark flavours at the output scale
        """
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

# this global variable is simply an instance that is not meant to be modifed -
# i.e., a Standard Model Wilson coefficient instance. This is useful since it
# allows caching SM intermediate results where needed.
_wc_sm = WilsonCoefficients()
