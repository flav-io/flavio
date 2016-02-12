import flavio
from flavio.statistics.fitters.pypmc import pypmcScan

fit = flavio.statistics.BayesianFit(
name=                 'Bayesian SM CKM fit',
constraints=          flavio.default_parameters,
wc_obj=               flavio.WilsonCoefficients(),
fit_parameters=       ['Vus', 'Vub', 'Vcb', 'gamma'],
nuisance_parameters=  ['bag_B0_1', 'bag_Bs_1', 'bag_K0_1', 'f_B0', 'f_Bs', 'f_K0', 'eta_tt_K0', 'eta_cc_K0', 'eta_ct_K0', 'kappa_epsilon', 'DeltaM_K0',],
fit_coefficients=     [],
measurements=         ['HFAG osc summer 2015', 'HFAG UT summer 2015', 'PDG kaon CPV'],
exclude_observables=  []
)

scan = pypmcScan(fit)

scan.run(steps=100000, burnin=0, adapt=500)
scan.save_result(fit.name)
