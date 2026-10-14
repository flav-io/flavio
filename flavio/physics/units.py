r"""Constants for unit conversion."""

from math import pi

# exact definition of dimensionless conversion factors for constants in SI as of 2019
e_SI = 1.602176634e-19  # e/C
h_SI = 6.62607015e-34  # h/(Js)
c_SI = 299792458  # c/(m/s)

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
hbar = 1
c = 1

eV = GeV / G
keV = k * eV
MeV = M * eV
TeV = T * eV

h = (2 * pi) * hbar

# Joule (eV / (e/C) = V * C = J)
J = eV / e_SI

# second (h / (h/(Js)) / J = Js / J = s)
s = h / h_SI / J

# meter (c / (c/(m/s)) * s = m/s * s = m)
meter = c / c_SI * s # not to confuse with milli!

ms = milli * s
ns = n * s
ps = p * s

mm = milli * meter
nm = n * meter
pm = p * meter
fm = f * meter
