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

        assert isinstance(fit, flavio.statistics.classes.BayesianFit), "PyPMC fit object must be an instance of BayesianFit"
        self.fit = fit
        # start value is a random vector
        self.start = fit.get_random
        self.dimension = len(self.start)
        # generate another random vector to guess an initial step size
        _initial_sigma = np.absolute(self.start - fit.get_random)
        self._initial_proposal = pypmc.density.gauss.LocalGauss(_initial_sigma * np.eye(self.dimension))

        self.mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(target=self.fit.log_likelihood,
                                                                 proposal=self._initial_proposal,
                                                                 start=self.start,
                                                                 save_target_values=True,
                                                                 **kwargs)

    def run(self, steps, burnin=1000, adapt=500):
        done = 0
        self.mc.run( burnin )
        self.mc.clear()
        while done < steps:
            todo = min(steps-done, 500)
            accepted = self.mc.run( todo )
            done += todo
            self.mc.adapt()

    @property
    def result(self):
        return self.mc.history[:]
