import numpy as np
import flavio

try:
    import pypmc
except:
    pass

class pypmcScan(object):
    """
    """

    def __init__(self, fit, **kwargs):

        assert isinstance(fit, flavio.statistics.fits.BayesianFit), "PyPMC fit object must be an instance of BayesianFit"
        self.fit = fit
        # start value is a random vector
        self.start = fit.get_random
        self.dimension = len(self.start)
        # generate another random vector to guess an initial step size
        _initial_sigma = np.absolute(self.start - fit.get_random)
        self._initial_proposal = pypmc.density.gauss.LocalGauss(_initial_sigma * np.eye(self.dimension))

        self.mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(target=self.fit.log_target,
                                                                 proposal=self._initial_proposal,
                                                                 start=self.start,
                                                                 save_target_values=True,
                                                                 **kwargs)

    def run(self, steps, burnin=1000, adapt=500):
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
        return self.mc.samples[:][self.find_burnin:]

    def save_result(self, file):
        res = self.result
        np.save(file, res)
