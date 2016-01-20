from flavio.physics.running import betafunctions
from flavio.physics.running import masses
from scipy.integrate import odeint


def rg_evolve(initial_condition, derivative, scale_in, scale_out):
    sol = odeint(derivative, initial_condition, [scale_in, scale_out])
    return sol[1]

def rg_evolve_sm(initial_condition, par, derivative_nf, scale_in, scale_out):
    if scale_in == scale_out:
        # no need to run!
        return initial_condition
    if scale_out < 0.1:
        raise ValueError('RG evolution below the strange threshold not implemented.')
    # quark mass thresholds
    thresholds = {
        3: 0.1,
        4: par[('mass','c')],
        5: par[('mass','b')],
        6: par[('mass','t')],
        }
    if scale_in > scale_out: # running DOWN
        # set initial values and scales
        initial_nf = initial_condition
        scale_in_nf = scale_in
        for nf in (6,5,4,3):
            if scale_in <= thresholds[nf]:
                continue
             # run either to next threshold or to final scale, whichever is closer
            scale_stop = max(thresholds[nf], scale_out)
            sol = rg_evolve(initial_nf, derivative_nf(nf), scale_in_nf, scale_stop)
            if scale_stop == scale_out:
                return sol
            initial_nf = sol
            scale_in_nf = thresholds[nf]
    if scale_in < scale_out: # running UP
        # set initial values and scales
        initial_nf = initial_condition
        scale_in_nf = scale_in
        for nf in (3,4,5,6):
            if nf < 6 and scale_in >= thresholds[nf+1]:
                continue
             # run either to next threshold or to final scale, whichever is closer
            scale_stop = min(thresholds[nf+1], scale_out)
            sol = rg_evolve(initial_nf, derivative_nf(nf), scale_in_nf, scale_stop)
            if scale_stop == scale_out:
                return sol
            initial_nf = sol
            scale_in_nf = thresholds[nf]
    return sol


def get_alpha(par, scale):
    r"""Get the running $\overline{\mathrm{MSbar}}$ $\alpha_s$ and $\alpha_e$
    at the specified scale.
    """
    alpha_in = [par[('alpha_s')], par[('alpha_e')]]
    scale_in = par[('mass','Z')]
    def derivative_nf(nf):
        return lambda x, mu: betafunctions.beta_qcd_qed(x, mu, nf)
    alpha_out = rg_evolve_sm(alpha_in, par, derivative_nf, scale_in, scale)
    return dict(zip(('alpha_s','alpha_e'),alpha_out))

def get_mq(par, m_in, scale_in, scale_out):
    alphas_in = get_alpha(par, scale_in)['alpha_s']
    x_in = [alphas_in, m_in]
    def derivative(x, mu, nf):
        d_alphas = betafunctions.beta_qcd_qed([x[0],0], mu, nf)[0] # only alpha_s
        d_m = masses.gamma_qcd(x[1], x[0], mu, nf)
        return [ d_alphas, d_m ]
    def derivative_nf(nf):
        return lambda x, mu: derivative(x, mu, nf)
    sol = rg_evolve_sm(x_in, par, derivative_nf, scale_in, scale_out)
    return sol[1]


def get_mb(par, scale):
    m = par[('mass','b')]
    return get_mq(par=par, m_in=m, scale_in=m, scale_out=scale)

def get_mc(par, scale):
    m = par[('mass','c')]
    return get_mq(par=par, m_in=m, scale_in=m, scale_out=scale)

def get_mu(par, scale):
    m = par[('mass','u')]
    return get_mq(par=par, m_in=m, scale_in=2.0, scale_out=scale)

def get_md(par, scale):
    m = par[('mass','d')]
    return get_mq(par=par, m_in=m, scale_in=2.0, scale_out=scale)

def get_ms(par, scale):
    m = par[('mass','s')]
    return get_mq(par=par, m_in=m, scale_in=2.0, scale_out=scale)
