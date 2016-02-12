from flavio.physics.bdecays.formfactors.b_p import bcl
from flavio.classes import AuxiliaryQuantity, Implementation

processes = ['B->K', 'B->D', 'B->pi']

def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)

for p in processes:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')

    iname = p + ' BCL3'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff, p, implementation=iname, n=3))
    i.set_description("3-parameter BCL parametrization (see arXiv:0807.2722).")

    iname = p + ' BCL4'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff, p, implementation=iname, n=4))
    i.set_description("4-parameter BCL parametrization (see arXiv:0807.2722).")
