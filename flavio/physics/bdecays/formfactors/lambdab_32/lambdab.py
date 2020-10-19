from flavio.physics.bdecays.formfactors.lambdab_32 import QuarkModel_MCN
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config


def ff_function(function, process):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict)


_process_dict = {}
_process_dict['Lambdab->Lambda(1520)'] = {'X': 'Lambda(1520)', 'P': 'K-', 'q': 'b->s'}

processes = ['Lambdab->Lambda(1520)',]


for p in processes:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')
    # MCN approach as in arXiv:1108.6129
    iname = p + ' MCN'
    i = Implementation(name=iname, quantity=quantity,
                       function=ff_function(QuarkModel_MCN.ff_equiv, p))
    i.set_description("Form factors calculated using the full quark model wave function with the full relativistic form of the quark current")

