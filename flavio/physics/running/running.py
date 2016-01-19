from flavio.physics.running import betafunctions
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
        return alpha_initial
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

def get_mq(par, flavour, scale):
    mass_initial = par[('mass',flavour)]
    scale_initial = mass_initial if quark_mass_scale[flavour] == 'mass' else quark_mass_scale[flavour]
    alphas_initial = get_alpha(par, scale)
    initial = [alphas_initial, mass_initial]
    def derivative(x, mu):
        d_alphas = betafunctions.beta_qcd_qed(x[0], mu, 5)
        d_m = 0.
        return [ d_alphas, d_m ]
    solution = rg_evolve(initial_condition=initial,
          derivative=derivative,
          scale_in=scale_initial,
          scale_out=scale)
    return solution
