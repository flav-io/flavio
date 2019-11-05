"""Classes for effective field theory (EFT) Wilson coefficients"""

from wilson import wcxf
import wilson


# sector names prior to v0.27 translated to WCxf sector names
sectors_flavio2wcxf = {
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
 'bdemu': 'dbmue',
 'bdetau': 'dbtaue',
 'bdmue': 'dbemu',
 'bdmumu': 'db',
 'bdmutau': 'dbtaumu',
 'bdnuenue': 'dbnunu',
 'bdnuenumu': 'dbnunu',
 'bdnuenutau': 'dbnunu',
 'bdnumunue': 'dbnunu',
 'bdnumunumu': 'dbnunu',
 'bdnumunutau': 'dbnunu',
 'bdnutaunue': 'dbnunu',
 'bdnutaunumu': 'dbnunu',
 'bdnutaunutau': 'dbnunu',
 'bdtaue': 'dbetau',
 'bdtaumu': 'dbmutau',
 'bdtautau': 'db',
 'bsbs': 'sbsb',
 'bsee': 'sb',
 'bsemu': 'sbmue',
 'bsetau': 'sbtaue',
 'bsmue': 'sbemu',
 'bsmumu': 'sb',
 'bsmutau': 'sbtaumu',
 'bsnuenue': 'sbnunu',
 'bsnuenumu': 'sbnunu',
 'bsnuenutau': 'sbnunu',
 'bsnumunue': 'sbnunu',
 'bsnumunumu': 'sbnunu',
 'bsnumunutau': 'sbnunu',
 'bsnutaunue': 'sbnunu',
 'bsnutaunumu': 'sbnunu',
 'bsnutaunutau': 'sbnunu',
 'bstaue': 'sbetau',
 'bstaumu': 'sbmutau',
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
 'sdee': 'sd',
 'sdmumu': 'sd',
 'sdtautau': 'sd',
 'sdsd': 'sdsd',
 'ucuc': 'cucu',
 'suenue': 'usenu',
 'suenumu': 'usenu',
 'suenutau': 'usenu',
 'sumunue': 'usmunu',
 'sumunumu': 'usmunu',
 'sumunutau': 'usmunu',
 'sutaunue': 'ustaunu',
 'sutaunumu': 'ustaunu',
 'sutaunutau': 'ustaunu'}


class WilsonCoefficients(wilson.Wilson):
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
        self.wc = None
        self._options = {}

    def set_initial(self, wc_dict, scale, eft='WET', basis='flavio'):
        """Set initial values of Wilson coefficients.

        Parameters:

        - wc_dict: dictionary where keys are Wilson coefficient name strings and
          values are Wilson coefficient NP contribution values
        - scale: $\overline{\text{MS}}$ renormalization scale
        """
        super().__init__(wcdict=wc_dict, scale=scale, eft=eft, basis=basis)

    def set_initial_wcxf(self, wc):
        """Set initial values of Wilson coefficients from a WCxf WC instance.

        If the instance is given in a basis other than the flavio basis,
        the translation is performed automatically, if implemented in the
        `wcxf` package."""
        super().__init__(wcdict=wc.dict, scale=wc.scale, eft=wc.eft, basis=wc.basis)

    @property
    def get_initial_wcxf(self):
        """Return a wcxf.WC instance in the flavio basis containing the initial
        values of the Wilson coefficients."""
        if self.wc is None:
            raise ValueError("Need to set initial values first.")
        return self.wc

    @classmethod
    def from_wilson(cls, w, par_dict):
        if w is None:
            return None
        if isinstance(w, cls):
            return w
        fwc = cls()
        fwc.set_initial_wcxf(w.wc)
        fwc._cache = w._cache
        fwc._options = w._options
        _ckm_options = {k: par_dict[k] for k in ['Vus', 'Vcb', 'Vub', 'delta']}
        if fwc.get_option('parameters') != _ckm_options:
            fwc.set_option('parameters', _ckm_options)
        return fwc

    def run_wcxf(*args, **kwargs):
        raise ValueError("The method run_wcxf has been removed. Please use the match_run method of wilson.Wilson instead.")

    def get_wcxf(self, sector, scale, par, eft='WET', basis='flavio', nf_out=None):
        """Get the values of the Wilson coefficients belonging to a specific
        sector (e.g. `bsmumu`) at a given scale.

        Returns a WCxf.WC instance.

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
        if sector == 'all':
            mr_sectors = 'all'
        else:
            # translate from legacy flavio to wcxf sector if necessary
            wcxf_sector = sectors_flavio2wcxf.get(sector, sector)
            mr_sectors = (wcxf_sector,)
        if not self.wc:
            return wcxf.WC(eft=eft, basis=basis, scale=scale, values={})
        return self.match_run(scale=scale, eft=eft, basis=basis, sectors=mr_sectors)

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
        wcxf_basis = wcxf.Basis[eft, basis]
        if sector == 'all':
            coeffs = wcxf_basis.all_wcs
        else:
            # translate from legacy flavio to wcxf sector if necessary
            wcxf_sector = sectors_flavio2wcxf.get(sector, sector)
            coeffs = wcxf_basis.sectors[wcxf_sector].keys()
        wc_sm = dict.fromkeys(coeffs, 0)
        if not self.wc or not any(self.wc.values.values()):
            return wc_sm
        wc_out = self.get_wcxf(sector, scale, par, eft, basis, nf_out)
        wc_out_dict = wc_sm  # initialize with zeros
        wc_out_dict.update(wc_out.dict)  # overwrite non-zero entries
        return wc_out_dict


# this global variable is simply an instance that is not meant to be modifed -
# i.e., a Standard Model Wilson coefficient instance.
_wc_sm = WilsonCoefficients()
