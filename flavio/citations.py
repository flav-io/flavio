# Inspired by pybamm.citations
# https://github.com/pybamm-team/PyBaMM

from multiprocessing import Array
import flavio
import sys
from itertools import compress
import ctypes
import yaml
import pkgutil


class CitationScope:

    def __enter__(self):
        self._citations_global = flavio.citations
        flavio.citations = Citations()
        return flavio.citations

    def __exit__(self, type, value, traceback):
        flavio.citations = self._citations_global


class Citations:

    collect = CitationScope
    __all_citations = set(yaml.safe_load(
        pkgutil.get_data('flavio', 'data/citations.yml')
    ))
    __name__ = __name__

    def __init__(self, initial_citations=[]):
        self._initial_citations = set(initial_citations)
        self._all_citations = {
            k:i for i,k in
            enumerate(sorted(self.__all_citations | self._initial_citations))
        }
        self._array = Array(ctypes.c_bool, len(self._all_citations))
        for inspire_key in self._initial_citations:
            self.register(inspire_key)

    def __iter__(self):
        for citation in self.set:
            yield citation

    def __str__(self):
        return ",".join(self.set)

    @property
    def set(self):
        return set(compress(sorted(self._all_citations.keys()), self._array))

    @property
    def string(self):
        return str(self)

    def register(self, inspire_key):
        """Register a paper to be cited. The intended use is that this method
        should be called only when the referenced functionality is actually being used.
        Parameters
        ----------
        inspire_key : str
            The INSPIRE texkey for the paper to be cited
        """
        try:
            self._array[self._all_citations[inspire_key]] = True
        except KeyError:
            from flavio.util import get_datapath
            yaml_path = get_datapath('flavio', 'data/citations.yml')
            raise KeyError(
                f'The inspire key must be contained in {yaml_path}. '
                f'The key `{inspire_key}` was not found there.'
            )


    def clear(self):
        """Clear the list of cited papers (including any default citations)."""
        self._array[:] = [False]*len(self._array)

    def reset(self):
        """Reset the list of cited papers back to only the default ones."""
        self.clear()
        for inspire_key in self._initial_citations:
            self.register(inspire_key)


sys.modules[__name__] = Citations(["Straub:2018kue"])
