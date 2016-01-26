import numpy as np
from flavio.physics.running import running
from flavio.physics.bdecays import rge as rge_db1
from flavio.physics.mesonmixing import rge as rge_df2


class WilsonCoefficients(object):
    """
    """
    def __init__(self):
        self.initial = {}

    sectors = [ 'df2_sd', 'df2_cu', 'df2_bs', 'df2_bd',
                'df1_sd', 'df1_cu', 'df1_bs', 'df1_bd', ]

    coefficients_df1 = [ 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7eff', 'C8eff', 'C9', 'C10', 'C3Q', 'C4Q', 'C5Q', 'C6Q', 'Cb',
                         'C1p', 'C2p', 'C3p', 'C4p', 'C5p', 'C6p', 'C7effp', 'C8effp', 'C9p', 'C10p', 'C3Qp', 'C4Qp', 'C5Qp', 'C6Qp', 'Cbp',
                         'CS', 'CP', 'CSp', 'CPp', ]
    coefficients_df2 = [ 'CVLL', 'CSLL', 'CTLL','CVRR','CSRR','CTRR','CVLR','CSLR', ]

    def adm_df2(nf, alpha_s, alpha_e):
        return rge_df2.gamma_df2_array(nf, alpha_s)
    def adm_db1(nf, alpha_s, alpha_e):
        A_L = rge_db1.gamma_all(nf, alpha_s, alpha_e)
        A = np.zeros((34,34))
        A[:15,:15] = A_L
        A[15:30,15:30] = A_L
        return A

    coefficients = {}
    adm = {}
    rge_derivative = {}
    for s in [ 'df2_sd', 'df2_cu', 'df2_bs', 'df2_bd', ]:
        coefficients[s] = coefficients_df2
        adm[s] = adm_df2
        rge_derivative[s] = running.make_wilson_rge_derivative(adm[s])

    for s in [ 'df1_sd', 'df1_cu', 'df1_bs', 'df1_bd', ]:
        coefficients[s] = coefficients_df1

    for s in [ 'df1_bs', 'df1_bd', ]:
        adm[s] = adm_db1
        rge_derivative[s] = running.make_wilson_rge_derivative(adm[s])

    def set_initial(self, sector, scale, values):
        if sector not in self.sectors:
            raise KeyError("Sector " + sector + " not defined in this basis")
        self.initial[sector] = (scale, values)

    def get_wc(self, sector, scale, par):
        if sector not in self.sectors:
            raise KeyError("Sector " + sector + " not defined in this basis")
        if sector not in self.initial.keys():
            return np.zeros(len(self.coefficients[sector]))
        scale_in, values_in = self.initial[sector]
        values_out = running.get_wilson(par, values_in, self.rge_derivative[sector], scale_in, scale)
        return values_out
