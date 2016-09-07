"""Common functions for physics."""

from collections import Counter

def conjugate_par(par_dict):
    """Given a dictionary of parameter values, return the dictionary where
    all CP-odd parameters have flipped sign.

    This assumes that the only CP-odd parameters are `gamma` or `delta` (the
    CKM phase in the Wolfenstein or standard parametrization)."""
    cp_odd = ['gamma', 'delta']
    return {k: -v if k in cp_odd else v for k, v in par_dict.items()}

def conjugate_wc(wc_dict):
    """Given a dictionary of Wilson coefficients, return the dictionary where
    all coefficients are CP conjugated (which simply amounts to complex
    conjugation)."""
    return {k: v.conjugate() for k, v in wc_dict.items()}

def add_dict(dicts):
    """Add dictionaries.

    This will add the two dictionaries

    `A = {'a':1, 'b':2, 'c':3}`
    `B = {'b':3, 'c':4, 'd':5}`

    into

    `A + B {'a':1, 'b':5, 'c':7, 'd':5}`

    but works for an arbitrary number of dictionaries."""
    # start with the first dict
    res = Counter(dicts[0])
    # successively add all other dicts
    for d in dicts[1:]: res.update(Counter(d))
    return res


def lambda_K(a, b, c):
    r"""Källén function $\lambda$.

    $\lambda(a,b,c) = a^2 + b^2 + c^2 - 2 (ab + bc + ac)$
    """
    z = a**2 + b**2 + c**2 - 2 * (a * b + b * c + a * c)
    if z < 0:
        # to avoid sqrt(-1e-16) type errors due to numerical inaccuracies
        return 0
    else:
        return z
