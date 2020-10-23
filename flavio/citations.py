"""Citation class for handling theory calculation citations"""

import flavio


class Citations:

    """Entry point to citations management.
    This object may be used to record Bibtex citation information and then register that
    a particular citation is relevant for a particular implementation.

    Examples
    --------
    >>> import flavio
    >>> flavio.print_citations("citations.tex")
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        "Reset citations to default only (only for testing purposes)"
        # Initialize empty papers to cite
        self._papers_to_cite = set()

    def register(self, key):
        """Register a paper to be cited. The intended use is that this method
        should be called only when the referenced functionality is actually being used.

        Parameters
        ----------
        key : str
            The INSPIRE key for the paper to be cited
        """
        self._papers_to_cite.add(key)

    def print(self, filename=None):
        """Print all citations that were used for calculating theory predictions.

        Parameters
        ----------
        filename : str, optional
            Filename in which to print citations, in the form \cite{paper1,paper2}.
            If None, the citation list is returned as a string of the form "paper1,paper2".
        """
        citation_list = ",".join(self._papers_to_cite)
        if filename is None:
            return citation_list
        else:
            citation_text = "\cite{" + citation_list + "}"
            with open(filename, "w") as f:
                f.write(citation_text)


def print_citations(filename=None):
    "See `flavio.citations.print`"
    return flavio.citations.print(filename)


citations = Citations()
# Register the flavio paper
citations.register("Straub:2018kue")