import numpy as np
from flavio.physics import running, bdecays, mesonmixing

class WilsonCoefficients(object):
    """
    """
    def __init__(self):
        self.initial = {}

    sectors = [ 'df2_sd', 'df2_cu', 'df2_bs', 'df2_bd',
                'df1_sd', 'df1_cu', 'df1_bs', 'df1_bd', ]

    coefficients_df1 = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7eff', 'C8eff', 'C9', 'C10', 'C3Q', 'C4Q', 'C5Q', 'C6Q', 'Cb' ]
    coefficients_df2 = ['CVLL', 'CSLL', 'CTLL','CVRR','CSRR','CTRR','CVLR','CSLR']

    def adm_df2(nf, alpha_s, alpha_e): return mesonmixing.rge.gamma_df2_array(nf, alpha_s)
    adm_db1 = bdecays.rge.gamma_all

    coefficients = {}
    adm = {}
    for s in [ 'df2_sd', 'df2_cu', 'df2_bs', 'df2_bd', ]:
        coefficients[s] = coefficients_df2
        adm[s] = adm_df2

    for s in [ 'df1_sd', 'df1_cu', 'df1_bs', 'df1_bd', ]:
        coefficients[s] = coefficients_df1

    for s in [ 'df1_bs', 'df1_bd', ]:
        adm[s] = adm_db1

    def set_initial(self, sector, scale, values):
        if sector not in self.sectors:
            raise KeyError("Sector " + sector + " not defined in this basis")
        self.initial[sector] = (scale, values)

    def get_wc(self, sector, scale, par):
        if sector not in self.sectors:
            raise KeyError("Sector " + sector + " not defined in this basis")
        if sector not in self.initial.keys():
            raise KeyError("No initial condition defined for " + sector)
        scale_in, values_in = self.initial[sector]
        values_out = running.running.get_wilson(par, values_in, self.adm[sector], scale_in, scale)
        return values_out
