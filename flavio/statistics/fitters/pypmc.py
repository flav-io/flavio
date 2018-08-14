"""Interface to the `pypmc` Markov Chain Monte Carlo package."""

import numpy as np
import flavio

try:
    import pypmc
except:
    pass

class pypmcScan(object):
    """Interface to adaptive Markov Chain Monte Carlo using the `pypmc` package.

    Methods:
    - run: run the sampler
    - find_burnin: attempt to automatically determine the burn-in period by
      excluding points at the beginning of the chain with much worse likelihood
      than the end of the chain
    - result: get a flat array of the sampled points
    - save_result: save the result to a `.npy` file
    """

    def __init__(self, fit, **kwargs):
        """Initialize the pypmcScan instance.

        Parameters:

        - fit: an instance of `flavio.statistics.fits.BayesianFit`

        Additional keyword argumements will be passed to
        `markov_chain.AdaptiveMarkovChain`.
        """

        assert isinstance(fit, flavio.statistics.fits.BayesianFit), "PyPMC fit object must be an instance of BayesianFit"
        self.fit = fit
        # start value is a random vector
        self.start = fit.get_random_start
        self.dimension = len(self.start)
        # for the initial proposal distribution, generate N random samples
        # and compute the covariance
        N = max(50, 2*self.dimension)
        _initial_covariance = np.cov(np.array([fit.get_random for i in range(N)]).T)
        try:
            self._initial_proposal = pypmc.density.gauss.LocalGauss(_initial_covariance)
        except:
            # if this fails for some reason, discard the correlation
            self._initial_proposal = pypmc.density.gauss.LocalGauss(
                            np.eye(self.dimension)*np.diag(_initial_covariance))

        self.mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(target=self.fit.log_target,
                                                                 proposal=self._initial_proposal,
                                                                 start=self.start,
                                                                 save_target_values=True,
                                                                 **kwargs)

    def run(self, steps, burnin=1000, adapt=500):
        """Run the sampler.

        Parameters:

        - steps: number of steps per walker
        - burnin (optional): number of steps for burn-in (samples will not be
          retained); defaults to 1000
        - adapt (optional): number of steps after which to adapt the proposal
          distribution. Defaults to 500.
        """
        done = 0
        self.mc.run( burnin )
        self.mc.clear()
        while done < steps:
            todo = min(steps-done, adapt)
            accepted = self.mc.run( todo )
            done += todo
            self.mc.adapt()

    @property
    def find_burnin(self):
        """Return the index of the first sample that has a log-likelihood
        bigger than the mean minus three standard deviations of the distribution
        of log-likelihoods in the last 10% of the chain."""
        target = self.mc.target_values[:] # this contains all log likelihood values of the chain
        target_end = target[-len(target)//10:] # take the last 10% of the chain
        target_end_mn = np.mean(target_end) # mean of the last 10%
        target_end_std = np.std(target_end) # std. deviation of the last 10%
        # find the index of the first entry where the log likelihood is greater
        # than the mean of the last 10% - 3 standard deviations
        first_good = np.argmax(target > target_end_mn - 3*target_end_std)
        return first_good

    @property
    def result(self):
        """Return an array of the samples obtained, where points with low
        likelihood at the beginning of the chain have been omitted (using the
        `find_burnin` method)."""
        return self.mc.samples[:][self.find_burnin:]

    def save_result(self, file):
        """Save the samples obtained to a `.npy` file."""
        res = self.result
        np.save(file, res)
