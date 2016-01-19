from flavio.physics.running import betafunctions
from flavio.physics.running import masses
from scipy.integrate import odeint


# scale at which the MSbar quark masses in the parameter dictionary are
# assumed to be renormalized.
quark_mass_scale = {
'u': 2., 'd': 2., 's': 2., # light quarks at 2 GeV
'c': 'mass', 'b': 'mass', # b and c at the scale = their mass
}

def rg_evolve(initial_condition, derivative, scale_in, scale_out):
    sol = odeint(derivative, initial_condition, [scale_in, scale_out])
    return sol[1]

def get_alpha(par, scale):
    r"""Get the running $\overline{\mathrm{MSbar}}$ $\alpha_s$ and $\alpha_e$
    at the specified scale.
    """
    alpha_initial = [par[('alpha_s')], par[('alpha_e')]]
    scale_initial = par[('mass','Z')]
    if scale == scale_initial:
        # no need to run!
        return dict(zip(('alpha_s','alpha_e'),alpha_initial))
    derivative = lambda x, mu: betafunctions.beta_qcd_qed(x, mu, 5)
    if scale < par[('mass','b')]:
        # if scale is below mb, run only down to mb first!
        solution_mb = rg_evolve(initial_condition=alpha_initial,
              derivative=derivative,
              scale_in=scale_initial,
              scale_out=par[('mass','b')])
        # now set the new initial scale and values at mb and use the 4-flavour RGE
        scale_initial = par[('mass','b')]
        alpha_initial = solution_mb
        derivative = lambda x, mu: betafunctions.beta_qcd_qed(x, mu, 4)
    solution = rg_evolve(initial_condition=alpha_initial,
          derivative=derivative,
          scale_in=scale_initial,
          scale_out=scale)
    return dict(zip(('alpha_s','alpha_e'),solution))



def get_mq(alphas_in, m_in, scale_in, scale_out, nf):
    x_in = [alphas_in, m_in]
    def derivative(x, mu):
        d_alphas = betafunctions.beta_qcd_qed([x[0],0], mu, nf)[0] # only alpha_s
        d_m = masses.gamma_qcd(x[1], x[0], mu, nf)
        return [ d_alphas, d_m ]
    solution = rg_evolve(initial_condition=x_in,
          derivative=derivative,
          scale_in=scale_in,
          scale_out=scale_out)
    return solution[1]

def get_mb(par, scale):
    m = par[('mass','b')]
    alphas = get_alpha(par, m)['alpha_s']
    # FIXME implement correct decoupling and flavour-dependence
    return get_mq(alphas, m, m, scale, 5)

def get_mc(par, scale):
    m = par[('mass','c')]
    alphas = get_alpha(par, m)['alpha_s']
    # FIXME implement correct decoupling and flavour-dependence
    return get_mq(alphas, m, m, scale, 4)

def get_mu(par, scale):
    m = par[('mass','u')]
    alphas = get_alpha(par, 2.)['alpha_s']
    # FIXME implement correct decoupling and flavour-dependence
    return get_mq(alphas, m, 2., scale, 4)

def get_md(par, scale):
    m = par[('mass','d')]
    alphas = get_alpha(par, 2.)['alpha_s']
    # FIXME implement correct decoupling and flavour-dependence
    return get_mq(alphas, m, 2., scale, 4)

def get_ms(par, scale):
    m = par[('mass','u')]
    alphas = get_alpha(par, 2.)['alpha_s']
    # FIXME implement correct decoupling and flavour-dependence
    return get_mq(alphas, m, 2., scale, 4)
