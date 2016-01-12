from flavio.physics.bdecays.formfactors.b_v import bsz

# ff = {}
#
# ff['bsz2'] = lambda process, q2, par: formfactors_btov_bsz.ff(process, q2, par, n=1)
# ff['bsz3'] = lambda process, q2, par: formfactors_btov_bsz.ff(process, q2, par, n=2)


class FormFactorParametrization(object):

    parametrizations = {}

    """docstring for """
    def __init__(self, name, transition, processes, parameters, function):
        self.name = name
        self.transition = transition
        self.processes = processes
        self.parameters = parameters
        self.function = function
        FormFactorParametrization.parametrizations[name] = self

    def get_ff(self, process, q2, par):
        return self.function(process, q2, par)

FFs = ["A0","A1","A12","V","T1","T2","T23"]

bsz_parnames = [ ff + '_' + a for ff in FFs for a in ["a0","a1","a2"]]
bsz_processes = ['B->K*','B->rho','B->omega','Bs->phi','Bs->K*']

bsz2 = FormFactorParametrization(name='bsz2',
                                 transition='b_v',
                                 processes=bsz_processes,
                                 parameters=bsz_parnames,
                                 function=lambda process, q2, par: bsz.ff(process, q2, par, n=1))

bsz3 = FormFactorParametrization(name='bsz3',
                                 transition='b_v',
                                 processes=bsz_processes,
                                 parameters=bsz_parnames,
                                 function=lambda process, q2, par: bsz.ff(process, q2, par, n=2))
