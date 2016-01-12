from flavio.physics.bdecays.formfactors.b_p import lattice
from flavio.physics.bdecays.formfactors.common import FormFactorParametrization

FFs = ["f+","fT","f0"]

lattice_parnames = [ ff + '_' + a for ff in FFs for a in ["a0","a1","a2"]]
lattice_processes = ['B->K']

lattice = FormFactorParametrization(name='btop_lattice',
                                 transition='b_p',
                                 processes=lattice_processes,
                                 parameters=lattice_parnames,
                                 function=lattice.ff)
