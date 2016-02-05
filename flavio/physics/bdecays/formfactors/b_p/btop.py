from flavio.physics.bdecays.formfactors.b_p import bcl
from flavio.classes import AuxiliaryQuantity, Implementation

processes = ['B->K', 'B->D']

def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)

for p in processes:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')

    iname = p + ' BCL'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff, p, implementation=iname))
    i.set_description("BCL parametrization (see arXiv:0807.2722).")
