import numpy as np
from ase import Atoms
from ase.build import graphene_nanoribbon
from itertools import combinations
from copy import deepcopy
import sys

def too_close( mol, cutoff):
    # decide whether the number of atom distances below cutoff are
    # equal to the number of atoms.
    return not np.sum(mol.get_all_distances() < cutoff) == len(mol)

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
            x_shift = abs(this_motif.get_center_of_mass()[0] - np.min(this_motif.positions.T[0]))
            if point == 0:
                motifs_list.append(this_motif)
            elif point == 1:
                this_motif.translate([-(lattice_constant+x_shift),0.0,0.0])
                motifs_list.append(this_motif)
            else:
                this_motif.translate([lattice_constant+x_shift,0.0,0.0])
                motifs_list.append(this_motif)
    if points == 4:
        for point in range(points):
            this_motif = motif.copy()
            x_shift = abs(this_motif.get_center_of_mass()[0] - np.min(this_motif.positions.T[0]))
            y_shift = abs(this_motif.get_center_of_mass()[1] - np.min(this_motif.positions.T[1]))
            if point == 0:
                motifs_list.append(this_motif)
            elif point == 1:
                this_motif.translate([lattice_constant+x_shift,0.0,0.0])
                motifs_list.append(this_motif)
            elif point == 2:
                this_motif.translate([0.0,-(lattice_constant+y_shift),0.0])
                motifs_list.append(this_motif)
            elif point == 3:
                this_motif.translate([lattice_constant+x_shift,-(lattice_constant+y_shift),0.0])
                motifs_list.append(this_motif)
    if points == 6:
        for point in range(points):
            this_motif = motif.copy()
            x_shift = abs(this_motif.get_center_of_mass()[0] - np.min(this_motif.positions.T[0]))
            y_shift = abs(this_motif.get_center_of_mass()[1] - np.min(this_motif.positions.T[1]))
            if point == 0:
                motifs_list.append(this_motif)
            elif point == 1:
                this_motif.translate([-(lattice_constant+x_shift),0.0,0.0])
                motifs_list.append(this_motif)
            elif point == 2:
                this_motif.translate([lattice_constant+x_shift,0.0,0.0])
                motifs_list.append(this_motif)
            elif point == 3:
                this_motif.translate([0.0,-(lattice_constant+y_shift),0.0])
                motifs_list.append(this_motif)
            elif point == 4:
                this_motif.translate([lattice_constant+x_shift,-(lattice_constant+y_shift),0.0])
                motifs_list.append(this_motif)
            elif point == 5:
                this_motif.translate([-(lattice_constant+x_shift),-(lattice_constant+y_shift),0.0])
                motifs_list.append(this_motif)
    return motifs_list

def create_all_configurations( lattice, axis,  angle ):
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
            for index in subset:
                lattice_copy[index-1].rotate(v=axis,a=angle * (np.pi/180.), center='COM')
                name[index] = 'C'
            for item in lattice_copy:
                conf.extend(item)
            les_confs.append(conf)
            conf_names.append(''.join(name))
    for conf in les_confs: 
        if too_close(conf,1.0):
            conf.write('conf.xyz')
            print(conf.get_all_distances())
            print('Some atoms were found to be too close when building the configurations')
            sys.exit()
    return dict(zip(conf_names,les_confs))

def create_configurations( lattice, axis,  angle ):
    """A function for generating all possible combinations of a list
    of configurations. The result is a dictionary with the label 'A'
    being used to mark positions on the lattice where no rotation has
    been applied and the label 'C' to indicate that this point has undergone
    a rotation of 180.0 about the z-axis.
    """
    les_confs = []
    conf_names = []
    for i in [0,1]:
        for index in range(i,len(lattice),2):
            print(index)
            name = len(lattice) * ['A']
            lattice_copy = deepcopy(lattice)
            conf = Atoms()
            lattice_copy[index].rotate(v=axis,a=angle * (np.pi/180.), center='COM')
            name[index] = 'C'
        print(name)
        for item in lattice_copy:
            conf.extend(item)
        les_confs.append(conf)
        conf_names.append(''.join(name))
    for conf in les_confs: 
        if too_close(conf,1.0):
            conf.write('conf.xyz')
            print(conf.get_all_distances())
            print('Some atoms were found to be too close when building the configurations')
            sys.exit()
    return dict(zip(conf_names,les_confs))

def generate_electrode(configuration, lattice_constant, z_len):
    x_len = np.ceil((np.max(configuration.get_positions().T[0]) - np.min(configuration.get_positions().T[0])
                    + lattice_constant / 2.0 )
                    / graphene_nanoribbon(1,1,sheet=True).get_cell()[0][0])
    y_len = np.ceil((np.max(configuration.get_positions().T[1]) - np.min(configuration.get_positions().T[1])
                    + lattice_constant / 2.0)
                    / graphene_nanoribbon(1,1,sheet=True).get_cell()[2][2])
    sheet = graphene_nanoribbon(int(x_len),int(y_len),sheet=True)
    sheet.rotate(v='x',a=np.pi/2.)
    sheet.set_cell([sheet.cell[0][0],sheet.cell[2][2],z_len])
    sheet.center()
    return sheet

def translate_ions( molec, z_vect, direction ):
    my_list = []
    for shift in z_vect:
        dummy = molec.copy()
        if direction == 0:
            dummy.translate([float(shift),0.0,0.0])
        elif direction == 1:
            dummy.translate([0.0,float(shift),0.0])
        else:
            dummy.translate([0.0,0.0,float(shift)])
        my_list.append(dummy)
    return my_list

