r"""Constants for unit conversion."""

from math import pi

# exact definition of constants in SI as of 2019
e_SI = 1.602176634e-19  # e in C
h_SI = 6.62607015e-34  # h in Js
c_SI = 299792458  # c in m/s

# SI prefixes
T = 1e12
G = 1e9
M = 1e6
k = 1e3
milli = 1e-3  # not to confuse with meter!
micro = 1e-6
n = 1e-9
p = 1e-12
f = 1e-15

# Natural Units

GeV = 1
eV = GeV / G
keV = k * eV
MeV = M * eV
TeV = T * eV

hbar = 1
h = hbar / (2 * pi)

# Joule
J = eV / e_SI
# second
s = 1 / (J * h_SI / (2 * pi))
# meter
meter = s / c_SI  # not to confuse with milli!

ms = milli * s
ns = n * s
ps = p * s

mm = milli * meter
nm = n * meter
pm = p * meter
fm = f * meter
