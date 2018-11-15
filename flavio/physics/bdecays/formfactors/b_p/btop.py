from flavio.physics.bdecays.formfactors.b_p import bcl, cln, bsz
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config

processes_H2L = ['B->K', 'B->pi']  # heavy to light
processes_H2H = ['B->D', ]  # heavy to heavy


def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)


for p in processes_H2L + processes_H2H:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')

    iname = p + ' BSZ3'
    i = Implementation(name=iname, quantity=quantity,
                       function=ff_function(bsz.ff, p, n=3))
    i.set_description("3-parameter BSZ parametrization (see arXiv:1811.00983).")

    iname = p + ' BCL3'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff, p, n=3))
    i.set_description("3-parameter BCL parametrization (see arXiv:0807.2722).")

    iname = p + ' BCL4'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff, p, n=4))
    i.set_description("4-parameter BCL parametrization (see arXiv:0807.2722).")

    iname = p + ' BCL3-IW'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff_isgurwise, p,
                   scale=config['renormalization scale']['bpll'], n=3))
    i.set_description("3-parameter BCL parametrization using improved Isgur-Wise relation"
                      " for the tensor form factor")

    iname = p + ' BCL4-IW'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff_isgurwise, p,
                   scale=config['renormalization scale']['bpll'], n=4))
    i.set_description("4-parameter BCL parametrization using improved Isgur-Wise relation"
                      " for the tensor form factor")

    iname = p + ' BCL3-IW-t0max'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff_isgurwise, p,
                   scale=config['renormalization scale']['bpll'], n=3, t0='tm'))
    i.set_description("3-parameter BCL parametrization using improved Isgur-Wise relation"
                      r" for the tensor form factor and taking $t_0=t_-$ in the $z$ expansion")

    iname = p + ' BCL4-IW-t0max'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(bcl.ff_isgurwise, p,
                   scale=config['renormalization scale']['bpll'], n=4, t0='tm'))
    i.set_description("4-parameter BCL parametrization using improved Isgur-Wise relation"
                      r" for the tensor form factor and taking $t_0=t_-$ in the $z$ expansion")

for p in processes_H2H:
    iname = p + ' CLN'
    i = Implementation(name=iname, quantity=quantity,
                   function=ff_function(cln.ff, p,
                   scale=config['renormalization scale']['bpll']))
    i.set_description("CLN parametrization based on HQET")
