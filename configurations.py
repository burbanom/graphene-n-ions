import numpy as np
from ase import Atoms
from ase.build import graphene_nanoribbon
from itertools import combinations
from copy import deepcopy
import sys

def mol_setup( molecule, z_translation, axes, angles, cell ):
    my_mol = deepcopy(molecule)
    my_mol.set_cell(cell)
    my_mol.center()
    for axis, angle in zip( axes, angles):
        my_mol.rotate(a= angle,v=axis,center='COP')
    my_mol.translate([0.0,0.0, z_translation])
    return my_mol

def too_close( mol, cutoff):
    # decide whether the number of atom distances below cutoff are
    # equal to the number of atoms.
    return not np.sum(mol.get_all_distances() < cutoff) == len(mol)

def build_lattice( points, motif, lattice_constant ):
    "A function for creating very simple lattices"
    motifs_list = []
    # A lattice with one point
    if points == 1:
        motifs_list.append(motif)
    # A lattice with two points
    if points == 2:
        for point in range(points):
            this_motif = motif.copy()
            if point == 0:
                motifs_list.append(this_motif)
            else:
                this_motif.translate([lattice_constant,0.0,0.0])
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
                lattice_copy[index-1].rotate(a=angle ,v=axis, center='COP')
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
        name = len(lattice) * ['A']
        lattice_copy = deepcopy(lattice)
        for index in range(i,len(lattice),2):
            lattice_copy[index].rotate(a=angle , v=axis,center='COP')
            name[index] = 'C'
        conf = Atoms()
        for item in lattice_copy:
            conf.extend(item)
        les_confs.append(conf)
        conf_names.append(''.join(name))
    for conf in les_confs: 
        if too_close(conf,1.0):
            conf.write('too_close.cif')
            print(conf.get_all_distances())
            print('Some atoms were found to be too close when building the configurations')
            sys.exit()
    return dict(zip(conf_names,les_confs))

def generate_electrode(configuration, lattice_constant, z_len, x_in = None, y_in = None):

    x_test = np.ceil((np.max(configuration.get_positions().T[0]) - np.min(configuration.get_positions().T[0])
                    + lattice_constant / 2.0 )
                    / graphene_nanoribbon(1,1,sheet=True).get_cell()[0][0])
    y_test = np.ceil((np.max(configuration.get_positions().T[1]) - np.min(configuration.get_positions().T[1])
                    + lattice_constant / 2.0)
                    / graphene_nanoribbon(1,1,sheet=True).get_cell()[2][2])

    if x_in is not None:
        x_in = np.ceil(x_in / graphene_nanoribbon(1,1,sheet=True).get_cell()[0][0])
        if x_in >= x_test:
            x_len = x_in
        else:
            x_len = x_test
    else:
        x_len = x_test

    if y_in is not None:
        y_in = np.ceil(y_in / graphene_nanoribbon(1,1,sheet=True).get_cell()[2][2])
        if y_in >= y_test:
            y_len = y_in
        else:
            y_len = y_test
    else:
        y_len = y_test

    sheet = graphene_nanoribbon(int(x_len),int(y_len),sheet=True)
    sheet.rotate(a=90.0, v='x')
    sheet.set_cell([sheet.cell[0][0],sheet.cell[2][2],z_len])
    sheet.center()
    return sheet

def translate_ions( lattice, shifts_list, direction ):
    my_list = []
    my_names = []
    molec = Atoms()
    for point in lattice:
        molec.extend(point)
    for shift in shifts_list:
        dummy = molec.copy()
        if direction == 'x':
            dummy.translate((float(shift),0.0,0.0))
            label = 'x-'
        elif direction == 'y':
            dummy.translate((0.0,float(shift),0.0))
            label = 'y-'
        elif direction == 'z':
            dummy.translate((0.0,0.0,float(shift)))
            label = 'z-'
        else:
            print('ERROR: Translate directions can be x, y or z only.')
            sys.exit()

        my_list.append(dummy)
        my_names.append(label+str(shift))

    return dict(zip(my_names,my_list))

