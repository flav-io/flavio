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

        self.dimension = len(fit.get_random_start)
        if nwalkers is None:
            self.nwalkers = self.dimension * 10
        else:
            self.nwalkers = nwalkers
        def get_random_good():
            # iterate until a random point with finite probability is found
            good = False
            i = 0
            while not good:
                x = fit.get_random_start
                good = np.isfinite(fit.log_target(x))
                i += 1
                if(i == 1000):
                    raise ValueError("Could not find enough starting values with finite probability. "
                                     " Try reducing the starting priors for the Wilson coefficients.")
            return x
        self.start = [get_random_good() for i in range(self.nwalkers)]


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
