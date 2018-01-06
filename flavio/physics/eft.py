"""Classes for effective field theory (EFT) Wilson coefficients"""

from flavio.config import config
import wcxf
import wetrunner

# List of all Wilson coefficients in the Standard basis

# A "sector" is a set of WCs relevant for a particular class of processes
# (e.g. b->sll) that closes under renormalization.
# Individual WCs can appear in more than one set.

coefficients = {}

_fcnc = ['bs', 'bd', 'sd', ]
_ll = ['ee', 'mumu', 'tautau']
_lilj = ['emu', 'mue', 'etau', 'taue', 'mutau', 'taumu']
_lnu = ['enue', 'munumu', 'taunutau',
        'enumu', 'enutau', 'munue', 'munutau', 'taunue', 'taunumu']
_fccc = ['bc', 'bu', 'su', 'du', 'dc', 'sc', ]
_nunu = ['nuenue', 'numunumu', 'nutaunutau']
_nuinuj = ['nuenumu', 'numunue', 'nuenutau', 'nutaunue', 'numunutau', 'nutaunumu']

# DeltaF=2 operators
for qq in _fcnc:
    # DeltaF=2 operators
    coefficients[qq + qq] = [ 'CVLL_'+qq+qq, 'CSLL_'+qq+qq, 'CTLL_'+qq+qq,
                        'CVRR_'+qq+qq, 'CSRR_'+qq+qq, 'CTRR_'+qq+qq,
                        'CVLR_'+qq+qq, 'CSLR_'+qq+qq, ]

    # DeltaF=1 operators
    for ll in _ll:
        coefficients[qq + ll] = [ 'C1_'+qq, 'C2_'+qq, # current-current
                            'C3_'+qq, 'C4_'+qq, 'C5_'+qq, 'C6_'+qq, # QCD penguins
                            'C7_'+qq, 'C8_'+qq, # dipoles
                            'C9_'+qq+ll, 'C10_'+qq+ll, # semi-leptonic
                            'C3Q_'+qq, 'C4Q_'+qq, 'C5Q_'+qq, 'C6Q_'+qq, 'Cb_'+qq, # EW penguins
                            # and everything with flipped chirality ...
                            'C1p_'+qq, 'C2p_'+qq,
                            'C3p_'+qq, 'C4p_'+qq, 'C5p_'+qq, 'C6p_'+qq,
                            'C7p_'+qq, 'C8p_'+qq,
                            'C9p_'+qq+ll, 'C10p_'+qq+ll,
                            'C3Qp_'+qq, 'C4Qp_'+qq, 'C5Qp_'+qq, 'C6Qp_'+qq, 'Cbp_'+qq,
                            # scalar and pseudoscalar
                            'CS_'+qq+ll, 'CP_'+qq+ll,
                            'CSp_'+qq+ll, 'CPp_'+qq+ll, ]

    # DeltaF=1 decays with same-flavour neutrinos in the final state
    for ll in _nunu:
        coefficients[qq + ll] = [ 'CL_'+qq+ll, 'CR_'+qq+ll, ]

    # DeltaF=1 decays with differently flavoured neutrinos in the final state
    for ll in _nuinuj:
        coefficients[qq + ll] = [ 'CL_'+qq+ll, 'CR_'+qq+ll, ]

    # DeltaF=1 LFV decays
    for ll in _lilj:
        coefficients[qq + ll] = [ 'C9_'+qq+ll, 'C10_'+qq+ll, # semi-leptonic
                            'C9p_'+qq+ll, 'C10p_'+qq+ll,
                            'CS_'+qq+ll, 'CP_'+qq+ll, # scalar and pseudoscalar
                            'CSp_'+qq+ll, 'CPp_'+qq+ll, ]

    # tree-level weak decays
    for qq in _fccc:
        for ll in _lnu:
            coefficients[qq + ll] = [ 'CVL_'+qq+ll, 'CSR_'+qq+ll, 'CT_'+qq+ll,
                                'CVR_'+qq+ll, 'CSL_'+qq+ll, ]


# sector names prior to v0.27 translated to WCxf sector names
sectors_flavio2wcxf ={
 'bcenue': 'cbenu',
 'bcenumu': 'cbenu',
 'bcenutau': 'cbenu',
 'bcmunue': 'cbmunu',
 'bcmunumu': 'cbmunu',
 'bcmunutau': 'cbmunu',
 'bctaunue': 'cbtaunu',
 'bctaunumu': 'cbtaunu',
 'bctaunutau': 'cbtaunu',
 'bdbd': 'dbdb',
 'bdee': 'db',
 'bdemu': 'dbemu',
 'bdetau': 'dbetau',
 'bdmue': 'dbmue',
 'bdmumu': 'db',
 'bdmutau': 'dbmutau',
 'bdnuenue': 'dbnunu',
 'bdnuenumu': 'dbnunu',
 'bdnuenutau': 'dbnunu',
 'bdnumunue': 'dbnunu',
 'bdnumunumu': 'dbnunu',
 'bdnumunutau': 'dbnunu',
 'bdnutaunue': 'dbnunu',
 'bdnutaunumu': 'dbnunu',
 'bdnutaunutau': 'dbnunu',
 'bdtaue': 'dbtaue',
 'bdtaumu': 'dbtaumu',
 'bdtautau': 'db',
 'bsbs': 'sbsb',
 'bsee': 'sb',
 'bsemu': 'sbemu',
 'bsetau': 'sbetau',
 'bsmue': 'sbmue',
 'bsmumu': 'sb',
 'bsmutau': 'sbmutau',
 'bsnuenue': 'sbnunu',
 'bsnuenumu': 'sbnunu',
 'bsnuenutau': 'sbnunu',
 'bsnumunue': 'sbnunu',
 'bsnumunumu': 'sbnunu',
 'bsnumunutau': 'sbnunu',
 'bsnutaunue': 'sbnunu',
 'bsnutaunumu': 'sbnunu',
 'bsnutaunutau': 'sbnunu',
 'bstaue': 'sbtaue',
 'bstaumu': 'sbtaumu',
 'bstautau': 'sb',
 'buenue': 'ubenu',
 'buenumu': 'ubenu',
 'buenutau': 'ubenu',
 'bumunue': 'ubmunu',
 'bumunumu': 'ubmunu',
 'bumunutau': 'ubmunu',
 'butaunue': 'ubtaunu',
 'butaunumu': 'ubtaunu',
 'butaunutau': 'ubtaunu',
 'dcenue': 'cdenu',
 'dcenumu': 'cdenu',
 'dcenutau': 'cdenu',
 'dcmunue': 'cdmunu',
 'dcmunumu': 'cdmunu',
 'dcmunutau': 'cdmunu',
 'dctaunue': 'cdtaunu',
 'dctaunumu': 'cdtaunu',
 'dctaunutau': 'cdtaunu',
 'duenue': 'udenu',
 'duenumu': 'udenu',
 'duenutau': 'udenu',
 'dumunue': 'udmunu',
 'dumunumu': 'udmunu',
 'dumunutau': 'udmunu',
 'dutaunue': 'udtaunu',
 'dutaunumu': 'udtaunu',
 'dutaunutau': 'udtaunu',
 'scenue': 'csenu',
 'scenumu': 'csenu',
 'scenutau': 'csenu',
 'scmunue': 'csmunu',
 'scmunumu': 'csmunu',
 'scmunutau': 'csmunu',
 'sctaunue': 'cstaunu',
 'sctaunumu': 'cstaunu',
 'sctaunutau': 'cstaunu',
 'sdnuenue': 'sdnunu',
 'sdnuenumu': 'sdnunu',
 'sdnuenutau': 'sdnunu',
 'sdnumunue': 'sdnunu',
 'sdnumunumu': 'sdnunu',
 'sdnumunutau': 'sdnunu',
 'sdnutaunue': 'sdnunu',
 'sdnutaunumu': 'sdnunu',
 'sdnutaunutau': 'sdnunu',
 'sdsd': 'sdsd',
 'suenue': 'usenu',
 'suenumu': 'usenu',
 'suenutau': 'usenu',
 'sumunue': 'usmunu',
 'sumunumu': 'usmunu',
 'sumunutau': 'usmunu',
 'sutaunue': 'ustaunu',
 'sutaunumu': 'ustaunu',
 'sutaunutau': 'ustaunu'}


class WilsonCoefficients(object):
    """Class representing a point in the EFT parameter space and giving
    access to RG evolution.

    Note that all Wilson coefficient values refer to new physics contributions
    only, i.e. they vanish in the SM.

    Methods:

    - set_initial: set the initial values of Wilson coefficients at some scale
    - get_wc: get the values of the Wilson coefficients at some scale
    - set_initial_wcxf: set the initial values from a wcxf.WC instance
    - get_wc_wcxf: get the values of the Wilson coefficients at some scale
      as a wcxf.WC instance
    """
    def __init__(self):
        self._initial = {}
        self._initial_wcxf = None
        self._cache = {}
        self.coefficients = coefficients
        self.all_wc = [c for v in self.coefficients.values() for c in v] # list of all coeffs


    def set_initial(self, wc_dict, scale, eft='WET', basis='flavio'):
        """Set initial values of Wilson coefficients.

        Parameters:

        - wc_dict: dictionary where keys are Wilson coefficient name strings and
          values are Wilson coefficient NP contribution values
        - scale: $\overline{\text{MS}}$ renormalization scale
        """
        if basis != 'flavio':
            wc = wcxf.WC(eft, basis, scale, wcxf.WC.dict2values(wc_dict))
            self.set_initial_wcxf(wc)
        all_wcs = wcxf.Basis[eft, basis].all_wcs
        for name in wc_dict:
            if name not in all_wcs:
                raise KeyError("Wilson coefficient {} not known in basis ({}, {})".format(name, eft, basis))
        self._initial = {'scale': scale, 'eft': eft, 'values': wc_dict}

    def set_initial_wcxf(self, wc):
        """Set initial values of Wilson coefficients from a WCxf WC instance.

        If the instance is given in a basis other than the flavio basis,
        the translation is performed automatically, if implemented in the
        `wcxf` package."""
        if not isinstance(wc, wcxf.WC):
            raise ValueError("`wc` should be an instance of `wcxf.WC`")
        if wc.eft not in ['WET', ]:
            raise NotImplementedError("Matching from a different EFT is currently not implemented.")
        if wc.basis == 'flavio':
            wc_dict = wc.dict
        else:
            wc_trans = wc.translate('flavio')
            wc_dict = wc_trans.dict
        self.set_initial(wc_dict, wc.scale)

    @property
    def get_initial_wcxf(self):
        """Return a wcxf.WC instance in the flavio basis containing the initial
        values of the Wilson coefficients."""
        if not self._initial:
            return None  # SM case
        if self._initial_wcxf is None:
            self._initial_wcxf = wcxf.WC(eft=self._initial['eft'],
                                         basis='flavio',
                                         scale=self._initial['scale'],
                                         values=wcxf.WC.dict2values(self._initial['values']))
        return self._initial_wcxf

    def run_wcxf(self, wc, eft, scale, sectors=None):
        """Run a set of Wilson coefficients (in the form of a `wcxf.WC`
        instance) to a different scale (and possibly different EFT)
        and return them as `wcxf.WC` instance in the flavio basis."""
        if wc.basis == 'flavio' and wc.eft == eft and scale == wc.scale:
            return wc  # nothing to do
        wr = wetrunner.WET(wc.translate('Bern'))
        if eft == wc.eft:  # just run
            return wr.run(scale, sectors=sectors).translate('flavio')
        elif eft == 'WET-4' and wc.eft == 'WET':  # match at mb
            mb = config['RGE thresholds']['mb']
            wc_mb = wr.run(mb, sectors=sectors).match('WET-4', 'Bern')
            wr4 = wetrunner.WET(wc_mb)
            return wr4.run(scale, sectors=sectors).translate('flavio')
        elif eft == 'WET-3' and wc.eft == 'WET-4':  # match at mc
            mc = config['RGE thresholds']['mc']
            wc_mc = wr.run(mc, sectors=sectors).match('WET-3', 'Bern')
            wr3 = wetrunner.WET(wc_mc)
            return wr3.run(scale, sectors=sectors).translate('flavio')
        elif eft == 'WET-3' and wc.eft == 'WET':  # match at mb and mc
            mb = config['RGE thresholds']['mb']
            mc = config['RGE thresholds']['mc']
            wc_mb = wr.run(mb, sectors=sectors).match('WET-4', 'Bern')
            wr4 = wetrunner.WET(wc_mb)
            wc_mc = wr4.run(scale, sectors=sectors).match('WET-3', 'Bern')
            wr3 = wetrunner.WET(wc_mc)
            return wr3.run(scale, sectors=sectors).translate('flavio')
        else:
            raise ValueError("Invalid input")

    def get_wc(self, sector, scale, par, eft='WET', basis='flavio', nf_out=None):
        """Get the values of the Wilson coefficients belonging to a specific
        sector (e.g. `bsmumu`) at a given scale.

        Returns a dictionary of WC values.

        Parameters:

        - sector: string name of the sector as defined in the WCxf EFT instance
        - scale: $\overline{\text{MS}}$ renormalization scale
        - par: dictionary of parameters
        - eft: name of the EFT at the output scale
        - basis: name of the output basis
        """
        # nf_out is only present to preserve backwards compatibility
        if nf_out == 5:
            eft = 'WET'
        elif nf_out == 4:
            eft = 'WET-4'
        elif nf_out == 3:
            eft = 'WET-3'
        elif nf_out is not None:
            raise ValueError("Invalid value: nf_out=".format(nf_out))
        # check if already there in cache
        wc_cached = self._get_from_cache(sector, scale, eft, basis)
        if wc_cached is not None:
            return wc_cached
        basis = wcxf.Basis[eft, basis]
        if sector not in basis.sectors:
            wcxf_sector = sectors_flavio2wcxf[sector]
        else:
            wcxf_sector = sector
        coeffs = basis.sectors[wcxf_sector].keys()
        wc_sm = {k: 0 for k in coeffs}
        if not self._initial:
            return wc_sm
        wc_out = self.run_wcxf(self.get_initial_wcxf, eft, scale, sectors=(wcxf_sector,))
        wc_out_dict = wc_sm  # initialize with zeros
        wc_out_dict.update(wc_out.dict)  # overwrite non-zero entries
        self._set_cache(wcxf_sector, scale, eft, basis, wc_out_dict)
        return wc_out_dict

    def _get_from_cache(self, sector, scale, eft, basis):
        """Try to load a set of Wilson coefficients from the cache, else return
        None."""
        try:
            return self._cache[eft][scale][basis][sector]
        except KeyError:
            return None

    def _set_cache(self, sector, scale, eft, basis, wc_out_dict):
        if eft not in self._cache:
            self._cache[eft] = {scale: {basis: {sector: wc_out_dict}}}
        elif scale not in self._cache[eft]:
            self._cache[eft][scale] = {basis: {sector: wc_out_dict}}
        elif basis not in self._cache[eft][scale]:
            self._cache[eft][scale][basis] = {sector: wc_out_dict}
        else:
            self._cache[eft][scale][basis][sector] = wc_out_dict

# this global variable is simply an instance that is not meant to be modifed -
# i.e., a Standard Model Wilson coefficient instance. This is useful since it
# allows caching SM intermediate results where needed.
_wc_sm = WilsonCoefficients()
