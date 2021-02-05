from flavio.physics.bdecays.formfactors.lambdab_32 import QuarkModel_MCN, LatticeQCD
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config


processes = ['Lambdab->Lambda(1520)',]


def ff_function(function, process):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict)


for p in processes:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')

    # MCN quark model approach as in arXiv:1108.6129
    iname = p + ' MCN'
    i = Implementation(name=iname, quantity=quantity,
                       function=ff_function(QuarkModel_MCN.ff_equiv, p))
    i.set_description("Form factors calculated using the full quark model wave function with the full relativistic form of the quark current")

    # Lattice QCD prediction arXiv:2009.09313
    iname = p + ' LatticeQCD'
    i = Implementation(name=iname, quantity=quantity,
                       function=ff_function(LatticeQCD.ff_equiv, p))
    i.set_description("Form factors from lattice QCD valid in $q^2 \in [16; 16.8] GeV^2$")
