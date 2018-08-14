from flavio.physics.bdecays.formfactors.b_v import bsz, sse, cln, clnexp
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config

processes_H2L = ['B->K*', 'B->rho', 'B->omega', 'Bs->phi', 'Bs->K*']  # heavy to light
processes_H2H = ['B->D*', ]  # heavy to heavy


def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)


for p in processes_H2L + processes_H2H:

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

for p in processes_H2H:

    iname = p + ' CLN'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(cln.ff, p, scale=config['renormalization scale']['bvll']))
    i.set_description("CLN parametrization")

    iname = p + ' CLNexp-IW'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(clnexp.ff, p, scale=config['renormalization scale']['bvll']))
    i.set_description("CLN-like parametrization as used by B factories"
                      " and using improved Isgur-Wise relations"
                      " for the tensor form factors")
