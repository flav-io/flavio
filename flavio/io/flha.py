"""Functions for reading a set of Wilson coefficients from an FLHA file.
This assumes the normalization convention used by SPheno modulues generated
by FlavorKit with SARAH 4.8.0 and should not be used with other FLHA-compatible
files, as the normalization is not fixed by the accord."""


import flavio.io.slha
import flavio
from math import pi, sqrt
import os
import logging

log = logging.getLogger('SLHA')

def prefactors_bsll(par, scale):
    GF = par['GF']
    alpha_e = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    xi_t = flavio.physics.ckm.xi('t', 'bs')(par)
    pre_all = -4*GF/sqrt(2)*xi_t
    pre_910 = pre_all * alpha_e/(4*pi)*2
    pre_7 =   pre_all /(16*pi**2)
    pre_8 =   pre_all /(16*pi**2)
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
           }


flha_dict ={
(30, 4422): 'C7eff_bs',
(30, 4322): 'C7effp_bs',
(30, 6421): 'C8eff_bs',
(30, 6321): 'C8effp_bs',
(305111, 4133): 'C9_bsee',
(305111, 4233): 'C9p_bsee',
(305111, 4137): 'C10_bsee',
(305111, 4237): 'C10p_bsee',
(305131, 4133): 'C9_bsmumu',
(305131, 4233): 'C9p_bsmumu',
(305131, 4137): 'C10_bsmumu',
(305131, 4237): 'C10p_bsmumu',
 }

def get_wc_from_file(filename):
    if not os.path.exists(filename):
        log.error("File " + filename + " not found.")
        return keys
    card = flavio.io.slha.read(filename)
    wc_flha = card.matrices['fwcoef']
    scale = wc_flha.scale
    par_dict = flavio.default_parameters.get_central_all()
    prefac = prefactors_bsll(par_dict, scale)
    wc_dict = {}
    for k, v in wc_flha.dict().items():
        if k[-1] != 1:
            continue
        if k[:2] not in flha_dict:
            log.warning('Wilson coefficient ' + str(k[:2]) + ' unknown to flavio; ignored.')
            continue
        wc_name = flha_dict[k[:2]]
        if wc_name not in prefac:
            wc_dict[wc_name] = v
        else:
            print(wc_name, v)
            wc_dict[wc_name] = v / prefac[wc_name]
    wc_obj =  flavio.WilsonCoefficients()
    wc_obj.set_initial(wc_dict, scale=scale)
    return wc_obj
