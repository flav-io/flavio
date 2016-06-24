import numpy as np
import flavio

try:
    import emcee
except:
    pass

class emceeScan(object):
    """
    """

    def __init__(self, fit, nwalkers=None, **kwargs):

        assert isinstance(fit, flavio.statistics.fits.BayesianFit), "emcee fit object must be an instance of BayesianFit"
        self.fit = fit

        self.dimension = len(fit.get_random)
        if nwalkers is None:
            self.nwalkers = self.dimension * 10
        else:
            self.nwalkers = nwalkers
        self.start = [fit.get_random for i in range(self.nwalkers)]


        self.mc = emcee.EnsembleSampler(nwalkers=self.nwalkers,
                                        dim=self.dimension,
                                        lnpostfn=self.fit.log_target,
                                        **kwargs)

    def run(self, steps, burnin=1000):
        pos = self.start
        if burnin > 0:
            pos, prob, state = self.mc.run_mcmc(self.start, burnin)
        self.mc.reset()
        self.mc.run_mcmc(pos, steps)

    @property
    def result(self):
        return self.mc.flatchain[:]

    def save_result(self, file):
        res = self.result
        np.save(file, res)
