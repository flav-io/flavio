r"""Hadronic form factors for the $D\to P$ transition"""

from . import bcl
from . import bsz

from flavio.classes import AuxiliaryQuantity, Implementation


processes = ['D->pi', 'D->K']


def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)


for p in processes:
    quantity = p + ' form factor'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
    a.set_description('Hadronic form factor for the ' + p + ' transition')

    iname = p + ' BSZ2'
    i = Implementation(name=iname, quantity=quantity,
                       function=ff_function(bsz.ff, p, n=2))
    i.set_description("2-parameter BSZ parametrization (see arXiv:1811.00983).")

    iname = p + ' BCL2'
    i = Implementation(name=iname, quantity=quantity,
                       function=ff_function(bcl.ff, p, n=2))
    i.set_description("2-parameter BCL parametrization (see arXiv:0807.2722).")


    iname = p + ' BSZ3'
    i = Implementation(name=iname, quantity=quantity,
                       function=ff_function(bsz.ff, p, n=3))
    i.set_description("3-parameter BSZ parametrization (see arXiv:1811.00983).")
