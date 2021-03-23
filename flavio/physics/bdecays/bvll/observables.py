"""Functions for exclusive $B\to V\ell^+\ell^-$ decays."""

from math import sqrt, log
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.common import conjugate_par, conjugate_wc
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.config import config
from flavio.physics.running import running
from .amplitudes import *
from flavio.classes import Observable, Prediction
import flavio
import warnings


def dGdq2(J):
    return 3/4. * (2 * J['1s'] + J['1c']) - 1/4. * (2 * J['2s'] + J['2c'])

def dGdq2_ave(J, J_bar):
    return ( dGdq2(J) + dGdq2(J_bar) )/2.

def dGdq2_diff(J, J_bar):
    return ( dGdq2(J) - dGdq2(J_bar) )/2.

# denominator of S_i and A_i observables
def SA_den(J, J_bar):
    return 2*dGdq2_ave(J, J_bar)

# denominator of P_i observables
def P_den(J, J_bar):
    return S_theory_num(J, J_bar, '2s')

def S_theory(J, J_bar, i):
    r"""CP-averaged angular observable $S_i$ in the theory convention."""
    return S_theory_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def S_theory_num(J, J_bar, i):
    return (J[i] + J_bar[i])

def A_theory(J, J_bar, i):
    r"""Angular CP asymmetry $A_i$ in the theory convention."""
    return A_theory_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def A_theory_num(J, J_bar, i):
    return (J[i] - J_bar[i])

def S_experiment(J, J_bar, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_experiment_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def S_experiment_num(J, J_bar, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory_num(J, J_bar, i)
    return S_theory_num(J, J_bar, i)

def A_experiment(J, J_bar, i):
    r"""Angular CP asymmetry $A_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return A_experiment_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def A_experiment_num(J, J_bar, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -A_theory_num(J, J_bar, i)
    return A_theory_num(J, J_bar, i)

def AFB_experiment(J, J_bar):
    r"""Forward-backward asymmetry in the LHCb convention.

    See eq. (C.9) of arXiv:1506.03970v2.
    """
    return AFB_experiment_num(J, J_bar)/SA_den(J, J_bar)

def AFB_experiment_num(J, J_bar):
    return 3/4.*S_experiment_num(J, J_bar, '6s')

def AFB_theory(J, J_bar):
    """Forward-backward asymmetry in the original theory convention.
    """
    return AFB_theory_num(J, J_bar)/SA_den(J, J_bar)

def AFB_theory_num(J, J_bar):
    return 3/4.*S_theory_num(J, J_bar, '6s')

def FL(J, J_bar):
    r"""Longitudinal polarization fraction $F_L$"""
    return FL_num(J, J_bar)/SA_den(J, J_bar)

def FL_num(J, J_bar):
    return -S_theory_num(J, J_bar, '2c')

def FLhat(J, J_bar):
    r"""Modified longitudinal polarization fraction for vanishing lepton masses,
    $\hat F_L$.

    See eq. (32) of arXiv:1510.04239.
    """
    return FLhat_num(J, J_bar)/SA_den(J, J_bar)

def FLhat_num(J, J_bar):
    return -S_theory_num(J, J_bar, '1c')


class BVllObservable(object):
    r"""Base class for $B\to V\ell^+\ell^- observable functions that
    facilitates caching/memoization."""

    def __init__(self, B, V, lep, wc_obj, par):
        """Initialize the class and cache results needed more often."""
        self.B = B
        self.V = V
        self.lep = lep
        self.wc_obj = wc_obj
        self.par = par
        self.par_conjugate = conjugate_par(par)
        self.prefactor = prefactor(None, self.par, B, V)
        self.prefactor_conjugate = prefactor(None, self.par_conjugate, B, V)
        self.scale = config['renormalization scale']['bvll']
        self.label = meson_quark[(B,V)] + lep + lep # e.g. bsmumu, bdtautau
        self.wctot_dict = wctot_dict(wc_obj, self.label, self.scale, par)
        self._ff = {}
        self._wceff = {}
        self._wceff_bar = {}
        self._ha = {}
        self._ha_bar = {}
        self._j = {}
        self._j_bar = {}
        self.ml = par['m_'+lep]
        self.mB = par['m_'+B]
        self.mV = par['m_'+V]
        self.mb = running.get_mb(par, self.scale)

    def ff(self, q2):
        """Get form factors. Cache and only recompute if necessary."""
        if q2 not in self._ff:
            self._ff[q2] = get_ff(q2, self.par, self.B, self.V)
        return self._ff[q2]

    def wceff(self, q2):
        """Get effective WCs. Cache and only recompute if necessary."""
        if q2 not in self._wceff:
            self._wceff[q2] = get_wceff(q2, self.wctot_dict, self.par, self.B, self.V, self.lep, self.scale)
        return self._wceff[q2]

    def wceff_bar(self, q2):
        """Get CP conjugate effective WCs. Cache and only recompute if necessary."""
        if q2 not in self._wceff_bar:
            self._wceff_bar[q2] = get_wceff(q2, conjugate_wc(self.wctot_dict), self.par_conjugate, self.B, self.V, self.lep, self.scale)
        return self._wceff_bar[q2]

    def helicity_amps_ff(self, q2, cp_conjugate):
        """Get helicity amps proportional to FFs. Cache and only recompute if necessary."""
        if not cp_conjugate:
            return angular.helicity_amps_v(q2,
                     self.mB, self.mV, self.mb, 0, self.ml, self.ml,
                     self.ff(q2), self.wceff(q2), self.prefactor)
        else:
            return angular.helicity_amps_v(q2,
                     self.mB, self.mV, self.mb, 0, self.ml, self.ml,
                     self.ff(q2), self.wceff_bar(q2), self.prefactor_conjugate)

    def ha(self, q2):
        """Get full helicity amps. Cache and only recompute if necessary."""
        if q2 not in self._ha:
            self._ha[q2] = add_dict((
                self.helicity_amps_ff(q2, cp_conjugate=False),
                get_ss(q2, self.wc_obj, self.par, self.B, self.V, cp_conjugate=False),
                get_subleading(q2, self.wc_obj, self.par, self.B, self.V, cp_conjugate=False)
                ))
        return self._ha[q2]

    def ha_bar(self, q2):
        """Get CP conjugate full helicity amps. Cache and only recompute if necessary."""
        if q2 not in self._ha_bar:
            self._ha_bar[q2] = add_dict((
                self.helicity_amps_ff(q2, cp_conjugate=True),
                get_ss(q2, self.wc_obj, self.par, self.B, self.V, cp_conjugate=True),
                get_subleading(q2, self.wc_obj, self.par, self.B, self.V, cp_conjugate=True)
                ))
        return self._ha_bar[q2]

    def j(self, q2):
        """Get angular coeffs. Cache and only recompute if necessary."""
        h = self.ha(q2)
        if q2 not in self._j:
            self._j[q2] = angular.angularcoeffs_general_v(h, q2, self.mB, self.mV, self.mb, 0, self.ml, self.ml)
        return self._j[q2]

    def jbar(self, q2):
        """Get CP conjugate angular coeffs. Cache and only recompute if necessary."""
        hbar = self.ha_bar(q2)
        if q2 not in self._j_bar:
            self._j_bar[q2] = angular.angularcoeffs_general_v(hbar, q2, self.mB, self.mV, self.mb, 0, self.ml, self.ml)
        return self._j_bar[q2]

    def jfunc(self, function, q2):
        """Return a function of J and Jbar at one value of q2."""
        return function(self.j(q2), self.jbar(q2))


class BVllObservableDifferential(BVllObservable):
    """Base class for differential observables depending on q2."""
    def __init__(self, q2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q2 = q2
        if q2 < 4 * self.ml**2 or q2 > (self.mB - self.mV)**2:
            self.allowed = False
        else:
            self.allowed = True


class BVllObservableBinned(BVllObservable):
    """Base class for binned observables depending on q2min and q2max."""
    def __init__(self, q2min, q2max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q2min = q2min
        self.q2max = q2max
        self.q2min_allowed = max(4 * self.ml**2, self.q2min)
        self.q2max_allowed = min((self.mB - self.mV)**2, self.q2max)


class BVll_dBRdq2(BVllObservableDifferential):
    """Differential branching ratio"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        if not self.allowed:
            return 0
        tauB = self.par['tau_' + self.B]
        return tauB * self.jfunc(dGdq2_ave, self.q2)


class BVll_obs(BVllObservableDifferential):
    """Differential function of angular coefficients"""
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def __call__(self):
        if not self.allowed:
            return 0
        return self.jfunc(self.func, self.q2)


class BVll_ratio(BVllObservableDifferential):
    """Differential ratio of functions of angular coefficients"""
    def __init__(self, func_num, func_den, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func_num = func_num
        self.func_den = func_den

    def __call__(self):
        if not self.allowed:
            return 0
        num = self.jfunc(self.func_num, self.q2)
        if num == 0:
            return 0
        den = self.jfunc(self.func_den, self.q2)
        return num / den


class BVll_pprime(BVllObservableDifferential):
    r"""Differential $P'$ observables"""
    def __init__(self, func_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func_num = func_num

    @staticmethod
    def func_2s(J, J_bar):
        return S_experiment_num(J, J_bar, '2s')

    @staticmethod
    def func_2c(J, J_bar):
        return S_experiment_num(J, J_bar, '2c')

    def __call__(self):
        if not self.allowed:
            return 0
        num = self.jfunc(self.func_num, self.q2)
        if num == 0:
            return 0
        den_2s = self.jfunc(self.func_2s, self.q2)
        den_2c = self.jfunc(self.func_2c, self.q2)
        den = 2 * sqrt(-den_2s * den_2c)
        return num / den


class BVll_dBRdq2_int(BVllObservableBinned):
    """Binned branching ratio"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsrel = 0.005

    def obs(self, q2):
        tauB = self.par['tau_' + self.B]
        return tauB * self.jfunc(dGdq2_ave, q2)

    def __call__(self):
        if self.q2max_allowed <= self.q2min_allowed:
            return 0
        return nintegrate_pole(self.obs, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel) / (self.q2max - self.q2min)


class BVll_obs_int(BVllObservableBinned):
    """Binned function of angular coefficients"""
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsrel = 0.005
        self.func = func

    def obs(self, q2):
        return self.jfunc(self.func, q2)

    def __call__(self):
        if self.q2max_allowed <= self.q2min_allowed:
            return 0
        return nintegrate_pole(self.obs, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel) / (self.q2max - self.q2min)


class BVll_int_ratio(BVllObservableBinned):
    """Binned ratio of functions if angular coefficients"""
    def __init__(self, func_num, func_den, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsrel = 0.005
        self.func_num = func_num
        self.func_den = func_den

    def obs_num(self, q2):
        return self.jfunc(self.func_num, q2)

    def obs_den(self, q2):
        return self.jfunc(self.func_den, q2)

    def __call__(self):
        if self.q2max_allowed <= self.q2min_allowed:
            return 0
        num = nintegrate_pole(self.obs_num, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        if num == 0:
            return 0
        den = nintegrate_pole(self.obs_den, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        return num / den


class BVll_int_pprime(BVllObservableBinned):
    r"""Binned $P'$ observables"""
    def __init__(self, func_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsrel = 0.005
        self.func_num = func_num

    def obs_num(self, q2):
        return self.jfunc(self.func_num, q2)

    def obs_2s(self, q2):
        return self.jfunc(lambda J, J_bar: S_experiment_num(J, J_bar, '2s'), q2)

    def obs_2c(self, q2):
        return self.jfunc(lambda J, J_bar: S_experiment_num(J, J_bar, '2c'), q2)

    def __call__(self):
        if self.q2max_allowed <= self.q2min_allowed:
            return 0
        num = nintegrate_pole(self.obs_num, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        if num == 0:
            return 0
        den_2s = nintegrate_pole(self.obs_2s, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        den_2c = nintegrate_pole(self.obs_2c, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        den = 2 * sqrt(-den_2s * den_2c)
        return num / den


def nintegrate_pole(function, q2min, q2max, epsrel=0.005):
    # this is a special integration function to treat the presence of the
    # photon pole at low q^2. If q2min is below 0.1 GeV^2, it adds and subtracts
    # the 1/q^2-enhanced pole part to split the integral into a well-behaved part
    # and one that is trivially solved analytically.
    # This leads to a huge speed-up.
    if q2min <= 0.1 and q2min > 0:
        q20 = q2min
        f_q20 = function(q20)
        int_a = flavio.math.integrate.nintegrate(lambda q2: function(q2)-f_q20*q20/q2, q2min, q2max, epsrel=epsrel)
        int_b = f_q20*q20 * log(q2max/q2min)
        return int_a + int_b
    else:
        return flavio.math.integrate.nintegrate(function, q2min, q2max)



# Functions returning functions needed for Prediction instances

def bvll_obs_int_ratio_leptonflavour(func, B, V, l1, l2):
    def fct(wc_obj, par, q2min, q2max):
        # ignore QCDF warnings for LFU ratios!
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The QCDF corrections should not be trusted .*")
            numobj = BVll_obs_int(func, q2min, q2max, B, V, l1, wc_obj, par)
            numobj.epsrel = 0.0005
            num = numobj()
            if num == 0:
                return 0
            denobj = BVll_obs_int(func, q2min, q2max, B, V, l2, wc_obj, par)
            denobj.epsrel = 0.0005
            den = denobj()
            return num / den
    return fct

def bvll_obs_ratio_leptonflavour(func, B, V, l1, l2):
    def fct(wc_obj, par, q2):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The QCDF corrections should not be trusted .*")
            num = BVll_obs(func, q2, B, V, l1, wc_obj, par)()
            if num == 0:
                return 0
            den = BVll_obs(func, q2, B, V, l2, wc_obj, par)()
            return num / den
    return fct


# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_observables = {
'ACP': {'func_num': dGdq2_diff, 'tex': r'A_\text{CP}', 'desc': 'Direct CP asymmetry'},
'AFB': {'func_num': AFB_experiment_num, 'tex': r'A_\text{FB}', 'desc': 'forward-backward asymmetry'},
'FL': {'func_num': FL_num, 'tex': r'F_L', 'desc': 'longitudinal polarization fraction'},
'S3': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 3), 'tex': r'S_3', 'desc': 'CP-averaged angular observable'},
'S4': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 4), 'tex': r'S_4', 'desc': 'CP-averaged angular observable'},
'S5': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 5), 'tex': r'S_5', 'desc': 'CP-averaged angular observable'},
'S6c': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, '6c'), 'tex': r'S_6^c', 'desc': 'CP-averaged angular observable'},
'S7': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 7), 'tex': r'S_7', 'desc': 'CP-averaged angular observable'},
'S8': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 8), 'tex': r'S_8', 'desc': 'CP-averaged angular observable'},
'S9': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 9), 'tex': r'S_9', 'desc': 'CP-averaged angular observable'},
'A3': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 3), 'tex': r'A_3', 'desc': 'Angular CP asymmetry'},
'A4': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 4), 'tex': r'A_4', 'desc': 'Angular CP asymmetry'},
'A5': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 5), 'tex': r'A_5', 'desc': 'Angular CP asymmetry'},
'A6s': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, '6s'), 'tex': r'A_6^s', 'desc': 'Angular CP asymmetry'},
'A7': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 7), 'tex': r'A_7', 'desc': 'Angular CP asymmetry'},
'A8': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 8), 'tex': r'A_8', 'desc': 'Angular CP asymmetry'},
'A9': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 9), 'tex': r'A_9', 'desc': 'Angular CP asymmetry'},
}
# for the P observables, the convention of LHCb is used. This differs by a
# sign in P_2 and P_3 from the convention in arXiv:1303.5794
_observables_p = {
'P1': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 3)/2., 'tex': r'P_1', 'desc': "CP-averaged \"optimized\" angular observable"},
'P2': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, '6s')/8., 'tex': r'P_2', 'desc': "CP-averaged \"optimized\" angular observable"},
'P3': {'func_num': lambda J, J_bar: -S_experiment_num(J, J_bar, 9)/4., 'tex': r'P_3', 'desc': "CP-averaged \"optimized\" angular observable"},
'ATIm': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 9)/2., 'tex': r'A_T^\text{Im}', 'desc': "Transverse CP asymmetry"},
}
_observables_pprime = {
'P4p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 4), 'tex': r'P_4^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
'P5p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 5), 'tex': r'P_5^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
# yes, P6p depends on J_7, not J_6. Don't ask why.
'P6p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 7), 'tex': r'P_6^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
'P8p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 8), 'tex': r'P_8^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
}
_hadr = {
'B0->K*': {'tex': r"B^0\to K^{\ast 0}", 'B': 'B0', 'V': 'K*0', },
'B+->K*': {'tex': r"B^+\to K^{\ast +}", 'B': 'B+', 'V': 'K*+', },
}

def make_metadata_binned(M, l, obs, obsdict):
    _process_tex = _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+r"^-"
    _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"
    B = _hadr[M]['B']
    V = _hadr[M]['V']
    _obs_name = "<" + obs + ">("+M+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
    _obs.set_description('Binned ' + obsdict['desc'] + r" in $" + _process_tex + r"$")
    _obs.tex = r"$\langle " + obsdict['tex'] + r"\rangle(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs

def make_metadata_differential(M, l, obs, obsdict):
    _process_tex = _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+r"^-"
    _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"
    B = _hadr[M]['B']
    V = _hadr[M]['V']
    _obs_name = obs + "("+M+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2'])
    _obs.set_description(obsdict['desc'][0].capitalize() + obsdict['desc'][1:] + r" in $" + _process_tex + r"$")
    _obs.tex = r"$" + obsdict['tex'] + r"(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs

def make_obs(M, l, obs, obsdict):
    B = _hadr[M]['B']
    V = _hadr[M]['V']
    func_num = obsdict['func_num']

    # binned angular observables
    _obs = make_metadata_binned(M, l, obs, obsdict)
    func = lambda wc_obj, par, q2min, q2max: BVll_int_ratio(func_num, SA_den, q2min, q2max, B, V, l, wc_obj, par)()
    Prediction(_obs.name, func)

    # differential angular observables
    _obs = make_metadata_differential(M, l, obs, obsdict)
    func = lambda wc_obj, par, q2: BVll_ratio(func_num, SA_den, q2, B, V, l, wc_obj, par)()
    Prediction(_obs.name, func)


def make_obs_p(M, l, obs, obsdict):
    B = _hadr[M]['B']
    V = _hadr[M]['V']
    func_num = obsdict['func_num']

    # binned "optimized" angular observables P
    _obs = make_metadata_binned(M, l, obs, obsdict)
    func = lambda wc_obj, par, q2min, q2max: BVll_int_ratio(func_num, P_den, q2min, q2max, B, V, l, wc_obj, par)()
    Prediction(_obs.name, func)

    # differential "optimized"  angular observables
    _obs = make_metadata_differential(M, l, obs, obsdict)
    func = lambda wc_obj, par, q2: BVll_ratio(func_num, P_den, q2, B, V, l, wc_obj, par)()
    Prediction(_obs.name, func)


def make_obs_pprime(M, l, obs, obsdict):
    B = _hadr[M]['B']
    V = _hadr[M]['V']
    func_num = obsdict['func_num']

    # binned "optimized"  angular observables
    _obs = make_metadata_binned(M, l, obs, obsdict)
    func = lambda wc_obj, par, q2min, q2max: BVll_int_pprime(func_num, q2min, q2max, B, V, l, wc_obj, par)()
    Prediction(_obs.name, func)

    # differential "optimized"  angular observables
    _obs = make_metadata_differential(M, l, obs, obsdict)
    func = lambda wc_obj, par, q2: BVll_pprime(func_num, q2, B, V, l, wc_obj, par)()
    Prediction(_obs.name, func)


def make_obs_br(M, l):
    """Make observable instances for branching ratios"""
    _process_tex = _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+r"^-"
    _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"
    B = _hadr[M]['B']
    V = _hadr[M]['V']

    # binned branching ratio
    _obs_name = "<dBR/dq2>("+M+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
    _obs.set_description(r"Binned differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    func = lambda wc_obj, par, q2min, q2max: BVll_dBRdq2_int(q2min, q2max, B, V, l, wc_obj, par)()
    Prediction(_obs_name, func)

    # differential branching ratio
    _obs_name = "dBR/dq2("+M+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2'])
    _obs.set_description(r"Differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    func = lambda wc_obj, par, q2: BVll_dBRdq2(q2, B, V, l, wc_obj, par)()
    Prediction(_obs_name, func)


def make_obs_lfur(M, l):
    """Make observable instances for lepton flavour ratios"""
    # binned ratio of BRs
    _obs_name = "<R"+l[0]+l[1]+">("+M+"ll)"
    _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
    _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
    _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
    for li in l:
        # add taxonomy for both processes (e.g. B->Vee and B->Vmumu)
        _obs.add_taxonomy(r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _hadr[M]['tex'] +_tex[li]+r"^+"+_tex[li]+r"^-$")
    Prediction(_obs_name, bvll_obs_int_ratio_leptonflavour(dGdq2_ave, _hadr[M]['B'], _hadr[M]['V'], *l))

    # differential ratio of BRs
    _obs_name = "R"+l[0]+l[1]+"("+M+"ll)"
    _obs = Observable(name=_obs_name, arguments=['q2'])
    _obs.set_description(r"Ratio of differential branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
    _obs.tex = r"$R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} (" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
    for li in l:
        # add taxonomy for both processes (e.g. B->Vee and B->Vmumu)
        _obs.add_taxonomy(r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _hadr[M]['tex'] +_tex[li]+r"^+"+_tex[li]+r"^-$")
    func = lambda wc_obj, par, q2: BVll_dBRdq2(q2, B, V, l, wc_obj, par)()
    Prediction(_obs_name, bvll_obs_ratio_leptonflavour(dGdq2_ave, _hadr[M]['B'], _hadr[M]['V'], *l))

# loop over all cases
for l in ['e', 'mu', 'tau']:
    for M in _hadr.keys():
        for obs, obsdict in _observables.items():
            # angular obs
            make_obs(M, l, obs, obsdict)
        for obs, obsdict in _observables_p.items():
            # P obs
            make_obs_p(M, l, obs, obsdict)
        for obs, obsdict in _observables_pprime.items():
            # P' obs
            make_obs_pprime(M, l, obs, obsdict)
        # BRs
        make_obs_br(M, l)

# lepton flavour ratios
for l in [('mu','e'), ('tau','mu'),]:
    for M in _hadr.keys():
        make_obs_lfur(M, l)


# define LFU differences D_P4p,5p and D_AFB
def diff(x, y):
    return x-y


for D_obs in ['P4p', 'P5p', 'AFB']:
    if D_obs in ['AFB']:
        tex = _observables[D_obs]['tex']
    else:
        tex = _observables_pprime[D_obs]['tex']
    obs = Observable.from_function('Dmue_{}(B0->K*ll)'.format(D_obs),
                                    ['{}(B0->K*mumu)'.format(D_obs), '{}(B0->K*ee)'.format(D_obs)],
                                    diff)
    obs.set_description(r"Difference of angular observable ${}$ in $B^0\to K^{{\ast 0}}\mu^+\mu^-$ and $B^0\to K^{{\ast 0}}e^+e^-$".format(tex))
    obs.tex = r"$D_{{{}}}^{{\mu e}}(B^0\to K^{{\ast 0}}\ell^+\ell^-)$".format(tex)
    obs.add_taxonomy(r"Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $B^0\to K^{\ast 0}e^+e^-$")
    obs.add_taxonomy(r"Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $B^0\to K^{\ast 0}\mu^+\mu^-$")

    obs = Observable.from_function('<Dmue_{}>(B0->K*ll)'.format(D_obs),
                                    ['<{}>(B0->K*mumu)'.format(D_obs), '<{}>(B0->K*ee)'.format(D_obs)],
                                    diff)
    obs.set_description(r"Binned difference of angular observable ${}$ in $B^0\to K^{{\ast 0}}\mu^+\mu^-$ and $B^0\to K^{{\ast 0}}e^+e^-$".format(tex))
    obs.tex = r"$\langle D_{{{}}}^{{\mu e}} \rangle(B^0\to K^{{\ast 0}}\ell^+\ell^-)$".format(tex)
    obs.add_taxonomy(r"Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $B^0\to K^{\ast 0}e^+e^-$")
    obs.add_taxonomy(r"Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $B^0\to K^{\ast 0}\mu^+\mu^-$")
