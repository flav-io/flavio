r"""Functions for $e^+ e^-\to W^+ W^- scattering"""

import flavio


def ee_WW(C, E):
    r"""$e^+e^-\to W^+W^-$ cross section normalized to the SM"""
    if E == 161.3:
        np = (-0.196 * C['phiWB']
              -0.088 * C['phiD']
              +0.069 * C['phil3_11']
              -0.175 * C['phil3_22']
              +0.087 * C['ll_1221']
              -0.001 * C['phil1_11']
              -0.001 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 172.1:
        np = (-0.001 * C['W']
              -0.186 * C['phiWB']
              -0.086 * C['phiD']
              +0.072 * C['phil3_11']
              -0.172 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.005 * C['phil1_11']
              -0.006 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 182.7:
        np = (-0.002 * C['W']
              -0.18 * C['phiWB']
              -0.085 * C['phiD']
              +0.076 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.009 * C['phil1_11']
              -0.009 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 188.6:
        np = (-0.002 * C['W']
              -0.178 * C['phiWB']
              -0.085 * C['phiD']
              +0.078 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.012 * C['phil1_11']
              -0.011 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 191.6:
        np = (-0.003 * C['W']
              -0.178 * C['phiWB']
              -0.086 * C['phiD']
              +0.079 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.013 * C['phil1_11']
              -0.012 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 195.5:
        np = (-0.003 * C['W']
              -0.177 * C['phiWB']
              -0.085 * C['phiD']
              +0.081 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.014 * C['phil1_11']
              -0.013 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 199.5:
        np = (-0.003 * C['W']
              -0.176 * C['phiWB']
              -0.085 * C['phiD']
              +0.082 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.085 * C['ll_1221']
              -0.016 * C['phil1_11']
              -0.013 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 201.6:
        np = (-0.004 * C['W']
              -0.176 * C['phiWB']
              -0.086 * C['phiD']
              +0.083 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.016 * C['phil1_11']
              -0.014 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 204.9:
        np = (-0.004 * C['W']
              -0.175 * C['phiWB']
              -0.086 * C['phiD']
              +0.084 * C['phil3_11']
              -0.17 * C['phil3_22']
              +0.085 * C['ll_1221']
              -0.018 * C['phil1_11']
              -0.014 * C['phie_11'])
        return 1 + 1e6 * np
    if E == 206.6:
        np = (-0.004 * C['W']
              -0.175 * C['phiWB']
              -0.086 * C['phiD']
              +0.085 * C['phil3_11']
              -0.171 * C['phil3_22']
              +0.086 * C['ll_1221']
              -0.018 * C['phil1_11']
              -0.015 * C['phie_11'])
        return 1 + 1e6 * np
    raise ValueError("The ee->WW cross section is not defined for {} GeV.".format(E))


def ee_WW_obs(wc_obj, par, E):
    scale = flavio.config['renormalization scale']['ee_ww']
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_WW(C, E)


_process_tex = r"e^+e^- \to W^+W^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to VV$ :: $' + _process_tex + r"$"
_obs_name = "R(ee->WW)"
_obs = flavio.classes.Observable(_obs_name)
_obs.arguments = ['E']
flavio.classes.Prediction(_obs_name, ee_WW_obs)
_obs.set_description(r"Cross section of $" + _process_tex + r"$ at energy $E$ normalized to the SM")
_obs.tex = r"R(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
