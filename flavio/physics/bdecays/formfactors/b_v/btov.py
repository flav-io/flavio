from flavio.physics.bdecays.formfactors.b_v import bsz, sse, cln
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config

processes = ['B->K*','B->rho','B->omega','Bs->phi','Bs->K*','B->D*']

def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)

for p in processes:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')

    iname = p + ' BSZ2'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bsz.ff, p, n=2))
    i.set_description("2-parameter BSZ parametrization (see arXiv:1503.05534)")

    iname = p + ' BSZ3'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bsz.ff, p, n=3))
    i.set_description("3-parameter BSZ parametrization (see arXiv:1503.05534)")

    iname = p + ' SSE'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(sse.ff, p, n=2))
    i.set_description("2-parameter simplified series expansion")

    iname = p + ' CLN-IW'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(cln.ff, p, scale=config['renormalization scale']['bvll']))
    i.set_description("CLN parametrization using improved Isgur-Wise relations"
                      " for the tensor form factors")
