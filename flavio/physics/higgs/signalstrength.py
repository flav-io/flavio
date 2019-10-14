r"""Functions for Higgs signal strengths."""

import flavio
from . import production
from . import decay
from . import width


prod_modes = {
    'ggF': {
        'desc': 'gluon fusion production',
        'tex': 'gg',
        'str': 'gg',
    },
    'hw': {
        'desc': '$W$ boson associated production',
        'tex': 'Wh',
        'str': 'Wh',
    },
    'hz': {
        'desc': '$Z$ boson associated production',
        'tex': 'Zh',
        'str': 'Zh',
    },
    'hv': {
        'desc': '$Z$ or $W$ boson associated production',
        'tex': 'Vh',
        'str': 'Vh',
    },
    'tth': {
        'desc': 'top pair associated production',
        'tex': r't\bar t h',
        'str': 'tth',
    },
    'vv_h': {
        'desc': 'weak boson fusion',
        'tex': r'\text{VBF}',
        'str': 'VBF',
    },
}

decay_modes = {
    'h_bb': {
        'tex': r'b\bar b',
        'str': 'bb',
        'tex_class': r'h\to ff',
    },
    'h_cc': {
        'tex': r'c\bar c',
        'str': 'cc',
        'tex_class': r'h\to ff',
    },
    'h_tautau': {
        'tex': r'\tau^+\tau^-',
        'str': 'tautau',
        'tex_class': r'h\to ff',
    },
    'h_mumu': {
        'tex': r'\mu^+\mu^-',
        'str': 'mumu',
        'tex_class': r'h\to ff',
    },
    'h_ww': {
        'tex': r'W^+W^-',
        'str': 'WW',
        'tex_class': r'h\to VV',
    },
    'h_zz': {
        'tex': r'ZZ',
        'str': 'ZZ',
        'tex_class': r'h\to VV',
    },
    'h_vv': {
        'tex': r'VV',
        'str': 'VV',
        'tex_class': r'h\to VV',
    },
    'h_zga': {
        'tex': r'Z\gamma',
        'str': 'Zgamma',
        'tex_class': r'h\to VV',
    },
    'h_gaga': {
        'tex': r'\gamma\gamma',
        'str': 'gammagamma',
        'tex_class': r'h\to VV',
    },
}


def higgs_signalstrength(wc_obj, par, name_prod, name_dec):
    scale = flavio.config['renormalization scale']['hdecays']
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    f_prod = getattr(production, name_prod)
    f_dec = getattr(decay, name_dec)
    return f_prod(C) * f_dec(C) / width.Gamma_h(par,  C)


def make_obs_higgs(name_prod, name_dec):
    d_dec = decay_modes[name_dec]
    d_prod = prod_modes[name_prod]
    process_tex = r"h \to {}".format(d_dec['tex'])
    process_taxonomy = r'Process :: Higgs production and decay :: $' + d_dec['tex_class'] + r'$ :: $' + process_tex + r"$"
    obs_name = "mu_{}(h->{})".format(d_prod['str'], d_dec['str'])
    obs = flavio.classes.Observable(obs_name)
    obs.set_description(r"Signal strength of $" + process_tex + r"$ from " + d_prod['desc'])
    obs.tex = r"$\mu_{" + d_prod['tex'] + r"}(" + process_tex + r")$"
    obs.add_taxonomy(process_taxonomy)

    def obs_fct(wc_obj, par):
        return higgs_signalstrength(wc_obj, par, name_prod, name_dec)

    flavio.classes.Prediction(obs_name, obs_fct)


for prod in prod_modes:
    for dec in decay_modes:
        make_obs_higgs(prod, dec)
