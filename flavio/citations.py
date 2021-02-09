"""Citation class for handling theory calculation citations"""
# citations.py is based on code from the PyBaMM project

# Copyright (c) 2018-2020, the PyBaMM team.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import flavio


class Citations:

    """Entry point to citations management.
    This object may be used to register a citation is relevant for a particular
    implementation. It can then also then print out a list of papers to be cited.

    Examples
    --------
    >>> import flavio
    >>> flavio.print_citations("citations.tex")
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        "Reset citations to empty"
        self._papers_to_cite = set()

    def register(self, key):
        """Register a paper to be cited. The intended use is that this method
        should be called only when the referenced functionality is actually being used.

        Parameters
        ----------
        key : str
            The INSPIRE texkey for the paper to be cited
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
    "See `flavio.default_citations.print`"
    return flavio.default_citations.print(filename)


default_citations = Citations()
# Register the flavio paper
default_citations.register("Straub:2018kue")
