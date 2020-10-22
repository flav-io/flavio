"""Citation class for handling theory calculation citations"""

import flavio


class Citations:

    """Entry point to citations management.
    This object may be used to record Bibtex citation information and then register that
    a particular citation is relevant for a particular implementation.

    Examples
    --------
    >>> import flavio
    >>> pybamm.print_citations("citations.tex")
    """

    def __init__(self):
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
            Filename to which to print citations. If None, citations are printed to the
            terminal.
        """
        citations = "\cite{" + ",".join(self._papers_to_cite) + "}"
        if filename is None:
            print(citations)
        else:
            with open(filename, "w") as f:
                f.write(citations)


def print_citations(filename=None):
    "See `flavio.citations.print`"
    flavio.citations.print(filename)


citations = Citations()
# Register the flavio paper
citations.register("Straub:2018kue")