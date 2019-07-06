from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.config import config


def ff(q2, par, B):
    r"""Central value of $B\to \gamma$ form factors 
    
    See hep-ph/0208256.pdf.
    """
    fB = par['f_'+B]
    mB = par['m_'+B]
    name = 'Bs->gamma KM '
    ff = {}
    ff['v'] = par[name+'betav']*fB*mB/(par[name+'deltav']+mB/2*(1-q2/mB**2))
    ff['a'] = par[name+'betaa']*fB*mB/(par[name+'deltaa']+mB/2*(1-q2/mB**2))
    ff['tv'] = par[name+'betatv']*fB*mB/(par[name+'deltatv']+mB/2*(1-q2/mB**2))
    ff['ta'] = par[name+'betata']*fB*mB/(par[name+'deltata']+mB/2*(1-q2/mB**2))
    return ff



quantity = 'Bs->gamma form factor'
a = AuxiliaryQuantity(name=quantity, arguments=['q2'])
a.set_description('Form factor for the Bs->gamma transition')

i = Implementation(name="Bs->gamma KM", quantity=quantity,
                   function = lambda wc_obj, par_dict, q2, B: ff(q2, par_dict, B))
i.set_description("KM parametrization (see hep-ph/0208256.pdf).")
