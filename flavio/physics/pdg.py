import csv
import pkgutil
import pprint


def read_pdg_masswidth(filename):
    """Read the PDG mass and width table and return a dictionary.

    Parameters
    ----------
    filname : string
        Path to the PDG data file, e.g. 'data/pdg/mass_width_2015.mcd'

    Returns
    -------
    particles : dict
        A dictionary where the keys are the particle names with the charge
        appended, e.g. 'gamma0' for the photon, 't+2/3' for the top quark.
        The value of the dictionary is again a dictionary with the following
        keys:
        - 'id': PDG particle ID
        - 'mass': list with the mass, postitive and negative error in GeV
        - 'width': list with the width, postitive and negative error in GeV
        - 'name': same as the key
    """
    data = pkgutil.get_data('flavio.physics', filename)
    lines = data.decode('utf-8').splitlines()
    particles_by_name = {}
    for line in lines:
        if  line.strip()[0] == '*':
            continue
        mass = ((line[33:51]),(line[52:60]),(line[61:69]))
        mass = [float(m) for m in mass]
        width = ((line[70:88]),(line[89:97]),(line[98:106]))
        if  width[0].strip() == '':
            width = (0,0,0)
        else:
            width = [float(w) for w in width]
        ids = line[0:32].split()
        charges = line[107:128].split()[1].split(',')
        if len(ids) != len(charges):
            raise ValueError()
        for i in range(len(ids)):
            particle = {}
            particle['id'] = ids[i].strip()
            particle['mass']  = mass
            particle['width'] = width
            particle['name'] = line[107:128].split()[0]
            particle['name'] =  particle['name'] + charges[i].strip()
            particles_by_name[particle['name']] = particle
    return particles_by_name
