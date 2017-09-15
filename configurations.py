import numpy as np
from ase import Atoms
from itertools import combinations
from copy import deepcopy

def build_lattice( points, motif, lattice_constant ):
    "A function for creating very simple lattices"
    motifs_list = []
    # A lattice with two points
    if points == 2:
        for point in range(points):
            this_motif = motif.copy()
            if point == 0:
                motifs_list.append(this_motif)
            else:
                this_motif.translate([lattice_constant,0.0,0.0])
                motifs_list.append(this_motif)
    # A lattice with three points
    if points == 3:
        for point in range(points):
            this_motif = motif.copy()
            if point == 0:
                motifs_list.append(this_motif)
            elif point == 1:
                this_motif.translate([-lattice_constant/2.0,-lattice_constant*np.sin(np.pi/3.0),0.0])
                motifs_list.append(this_motif)
            else:
                this_motif.translate([lattice_constant/2.0,-lattice_constant*np.sin(np.pi/3.0),0.0])
                motifs_list.append(this_motif)
    if points == 4:
        for point in range(points):
            this_motif = motif.copy()
            if point == 0:
                motifs_list.append(this_motif)
            elif point == 1:
                this_motif.translate([lattice_constant,0.0,0.0])
                motifs_list.append(this_motif)
            elif point == 2:
                this_motif.translate([0.0,-lattice_constant,0.0])
                motifs_list.append(this_motif)
            elif point == 3:
                this_motif.translate([lattice_constant,-lattice_constant,0.0])
                motifs_list.append(this_motif)
    return motifs_list

def create_configurations( lattice, axis,  angle ):
    """A function for generating all possible combinations of a list
    of configurations. The result is a dictionary with the label 'A'
    being used to mark positions on the lattice where no rotation has
    been applied and the label 'C' to indicate that this point has undergone
    a rotation of 180.0 about the z-axis.
    """
    les_confs = []
    conf_names = []
    for L in range(len(lattice)+1):
        for subset in combinations(range(len(lattice)), L):
            name = len(lattice) * ['A']
            lattice_copy = deepcopy(lattice)
            conf = Atoms()
            if len(subset) == 0:       
                for item in lattice_copy:
                    conf.extend(item)
            elif len(subset) == len(lattice_copy):
                for item in lattice_copy:
                    item.rotate(v=axis,a= angle * (np.pi/180.), center='COM')
                    conf.extend(item)
                    name = len(lattice) * ['C']
            else:
                for index in subset:
                    lattice_copy[index-1].rotate(v=axis,a=angle * (np.pi/180.), center='COM')
                    name[index] = 'C'
                for item in lattice_copy:
                    conf.extend(item)
            les_confs.append(conf)
            conf_names.append(''.join(name))
    return dict(zip(conf_names,les_confs))


