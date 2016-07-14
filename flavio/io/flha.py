"""Functions for reading a set of Wilson coefficients from an FLHA file.
This assumes the normalization convention used by SPheno modules generated
by FlavorKit with SARAH 4.8.1+ and should not be used with other FLHA-compatible
files, as the normalization is not fixed by the accord."""


import flavio.io.slha
import flavio
from math import pi, sqrt
import os
import logging

log = logging.getLogger('SLHA')
log.setLevel('WARNING')

def _prefactors_bsll(par, scale):
    GF = par['GF']
    alpha_e = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    xi_t = flavio.physics.ckm.xi('t', 'bs')(par)
    pre_all = -4*GF/sqrt(2)*xi_t
    pre_910 = pre_all * alpha_e/(4*pi)
    pre_7 =   pre_all /(16*pi**2)
    pre_8 =   pre_all /(16*pi**2)
    pre_nu =  pre_all * alpha_e/(4*pi)
    return {
            'C7eff_bs': pre_7,
            'C7effp_bs': pre_7,
            'C8eff_bs': pre_8,
            'C8effp_bs': pre_8,
            'C9_bsee': pre_910,
            'C9p_bsee': pre_910,
            'C10_bsee': pre_910,
            'C10p_bsee': pre_910,
            'C9_bsmumu': pre_910,
            'C9p_bsmumu': pre_910,
            'C10_bsmumu': pre_910,
            'C10p_bsmumu': pre_910,
            'CL_bsnuenue': pre_nu,
            'CR_bsnuenue': pre_nu,
            'CL_bsnumunumu': pre_nu,
            'CR_bsnumunumu': pre_nu,
            'CL_bsnutaunutau': pre_nu,
            'CR_bsnutaunutau': pre_nu,
           }


_flha_dict ={
(305, 4422): 'C7eff_bs',
(305, 4322): 'C7effp_bs',
(305, 6421): 'C8eff_bs',
(305, 6321): 'C8effp_bs',
(3051111, 4133): 'C9_bsee',
(3051111, 4233): 'C9p_bsee',
(3051111, 4137): 'C10_bsee',
(3051111, 4237): 'C10p_bsee',
(3051313, 4133): 'C9_bsmumu',
(3051313, 4233): 'C9p_bsmumu',
(3051313, 4137): 'C10_bsmumu',
(3051313, 4237): 'C10p_bsmumu',
(3051212, 4141): 'CL_bsnuenue',
(3051212, 4241): 'CR_bsnuenue',
(3051414, 4141): 'CL_bsnumunumu',
(3051414, 4241): 'CR_bsnumunumu',
(3051616, 4141): 'CL_bsnutaunutau',
(3051616, 4241): 'CR_bsnutaunutau',
}

def read_wilson(filename):
    r"""Read new physics contributions to Wilson coefficients from an output file
    in FLHA format produced by a SPheno module generated with SARAH 4.9.0+
    with FlavorKit.

    *Caution*: this function should not be used with FLHA files produced by any
    other code, in particular not the default version of SPheno or older
    SARAH/FlavorKit versions, as they use different normalization conventions
    for the Wilson coefficients.

    Input
    -----

    - `filename`: the path to the file as a string

    Returns an instance of `flavio.WilsonCoefficients` that can be used for
    `flavio.np_prediction`, for instance.
    """
    if not os.path.exists(filename):
        raise ValueError("File " + filename + " not found.")
    card = flavio.io.slha.read(filename)
    wc_flha = card.matrices['fwcoef']
    scale = wc_flha.scale
    par_dict = flavio.default_parameters.get_central_all()
    prefac = _prefactors_bsll(par_dict, scale)
    wc_dict = {}
    for k, v in wc_flha.dict().items():
        if k[-1] != 1: # only look at NP-only Wilson coefficients
            continue
        if k[:2] not in _flha_dict:
            log.warning('Wilson coefficient ' + str(k[:2]) + ' unknown to flavio; ignored.')
            continue
        wc_name = _flha_dict[k[:2]]
        if wc_name not in prefac:
            wc_dict[wc_name] = v
        else:
            wc_dict[wc_name] = v / prefac[wc_name]
    wc_obj =  flavio.WilsonCoefficients()
    wc_obj.set_initial(wc_dict, scale=scale)
    return wc_obj

def read_ckm(filename, par_constraints):
    r"""Read CKM Wolfenstein parameters from the `VCKMIN` block of an FLHA
    file.

    Input
    -----

    - `filename`: the path to the file as a string
    - `par_constraints`: an instance of `flavio.ParameterConstraints`, e.g.
    `flavio.default_parameters`

    Note that you have to set the config option
    `config['implementation']['CKM matrix']`
    to `Wolfenstein` first; otherwise a warning is issued.

    *Caution*: since the FLHA/SLHA format does not specify uncertainties,
    the parameters are assumed to be known exactly.
    """
    if not os.path.exists(filename):
        raise ValueError("File " + filename + " not found.")
    if flavio.config['implementation']['CKM matrix'] != 'Wolfenstein':
        log.warning('CKM matrix parametrization is not set to "Wolfenstein". read_ckm will have no effect!')
    card = flavio.io.slha.read(filename)
    try:
        wc_flha = card.blocks['vckmin']
        try:
            par_constraints.set_constraint('laC', wc_flha[1])
            par_constraints.set_constraint('A', wc_flha[2])
            par_constraints.set_constraint('rhobar', wc_flha[3])
            par_constraints.set_constraint('etabar', wc_flha[4])
        except KeyError:
            raise KeyError("One of the Wolfenstein parameters seems to be missing from the VCKMIN block")
    except KeyError:
        raise ValueError("This file does not contain a VCKMIN block")
