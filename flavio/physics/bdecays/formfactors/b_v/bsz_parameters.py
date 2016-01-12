import numpy as np
import json
import pkgutil

FFs = ["A0","A1","A12","V","T1","T2","T23"]
ai = ["a0","a1","a2"]
ff_a  = [(ff,a) for ff in FFs for a in ai]
a_ff_string  = [a + '_' + ff for ff in FFs for a in ai]

def get_ffpar(filename):
    f = pkgutil.get_data('flavio.physics', filename)
    data = json.loads(f.decode('utf-8'))
    central = np.array([data['central'][ff][a] for ff, a in ff_a])
    unc = np.array([data['uncertainty'][ff][a] for ff, a in ff_a])
    corr = np.array([[data['correlation'][ff1 + ff2][a1 + a2] for ff1, a1 in ff_a] for ff2, a2 in ff_a])
    return [central, unc, corr]

def ffpar_dict(filename, process):
    par = {}
    central = get_ffpar(filename)[0]
    for i, name in enumerate(a_ff_string):
        par[('formfactor', process, name)] = central[i]
    return par


ffpar_lcsr = {}
ffpar_lcsr.update(ffpar_dict('data/arXiv-1503-05534v1/BKstar_LCSR.json', 'B->K*'))
ffpar_lcsr.update(ffpar_dict('data/arXiv-1503-05534v1/Bomega_LCSR.json', 'B->omega'))
ffpar_lcsr.update(ffpar_dict('data/arXiv-1503-05534v1/Brho_LCSR.json', 'B->rho'))
ffpar_lcsr.update(ffpar_dict('data/arXiv-1503-05534v1/BsKstar_LCSR.json', 'Bs->K*'))
ffpar_lcsr.update(ffpar_dict('data/arXiv-1503-05534v1/Bsphi_LCSR.json', 'Bs->phi'))

ffpar_combined = {}
ffpar_combined.update(ffpar_dict('data/arXiv-1503-05534v1/BKstar_LCSR-Lattice.json', 'B->K*'))
ffpar_combined.update(ffpar_dict('data/arXiv-1503-05534v1/BsKstar_LCSR-Lattice.json', 'Bs->K*'))
ffpar_combined.update(ffpar_dict('data/arXiv-1503-05534v1/Bsphi_LCSR-Lattice.json', 'Bs->phi'))
