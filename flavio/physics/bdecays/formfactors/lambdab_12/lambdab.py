from flavio.physics.bdecays.formfactors.lambdab_12 import sse
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config

processes = ['Lambdab->Lambda',]

def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)

for p in processes:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')

    iname = p + ' SSE2'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(sse.ff, p, n=2))
    i.set_description("2-parameter simplified series expansion")

    iname = p + ' SSE3'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(sse.ff, p, n=2))
    i.set_description("3-parameter simplified series expansion")
