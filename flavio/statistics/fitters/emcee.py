"""Interface to the `emcee` ensemble Monte Carlo sampler.

See http://dan.iel.fm./emcee for details on `emcee`.
"""


import numpy as np
import flavio

try:
    import emcee
except:
    pass

class emceeScan(object):
    """Ensemble Monte Carlo sampler using the `emcee` package.

    Methods:

    - run: run the sampler
    - result: get a flat array of the sampled points of all walkers
    - save_result: save the result to a `.npy` file

    Important attributes:

    - mc: the `emcee.EnsembleSampler` instance

    """

    def __init__(self, fit, nwalkers=None, **kwargs):
        """Initialize the emceeScan.

        Parameters:

        - fit: an instance of `flavio.statistics.fits.BayesianFit`
        - nwalkers (optional): number of walkers. Defaults to ten times the number
          of dimensions.

        Additional keyword argumements will be passed to
        `emcee.EnsembleSampler`.
        """

        assert isinstance(fit, flavio.statistics.fits.BayesianFit), "emcee fit object must be an instance of BayesianFit"
        self.fit = fit

        self.dimension = len(fit.get_random_start)
        if nwalkers is None:
            self.nwalkers = self.dimension * 10
        else:
            self.nwalkers = nwalkers
        self.start = None
        self.mc = emcee.EnsembleSampler(nwalkers=self.nwalkers,
                                        dim=self.dimension,
                                        lnpostfn=self.fit.log_target,
                                        **kwargs)

    def _get_random_good(self):
        """Generate random starting points until a point with finite
        probability is found."""
        good = False
        i = 0
        while not good:
            x = self.fit.get_random_start
            good = np.isfinite(self.fit.log_target(x))
            i += 1
            if(i == 1000):
                raise ValueError("Could not find enough starting values with finite probability. "
                                 " Try reducing the starting priors for the Wilson coefficients.")
        return x

    def initialize_sampler(self):
        """Initialize the emcee.EnsembleSampler instance."""
        self.start = [self._get_random_good() for i in range(self.nwalkers)]

    def run(self, steps, burnin=1000, **kwargs):
        """Run the sampler.

        Parameters:

        - steps: number of steps per walker
        - burnin (optional): number of steps for burn-in (samples will not be
          retained); defaults to 1000

        Note that the total number of samples will be `steps * nwalkers`!
        """
        if self.start is None:
            self.initialize_sampler()
        pos = self.start
        if burnin > 0:
            pos, prob, state = self.mc.run_mcmc(self.start, burnin, **kwargs)
        self.mc.reset()
        self.mc.run_mcmc(pos, steps, **kwargs)

    @property
    def result(self):
        """Return a flat array of the samples."""
        return self.mc.flatchain[:]

    def save_result(self, file):
        """Save the samples to a `.npy` file."""
        res = self.result
        np.save(file, res)
