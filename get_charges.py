#!/usr/bin/env python3
import cp2k_calc
import numpy as np
import graphene_n_ions
from ase.io import read, write
import os

charges = cp2k_calc.hirshfeld_charges('eng.out')
coords_file = [ x for x in os.listdir() if x.endswith('.xyz') ][0]
coords = read(coords_file)
x_coords = coords.positions.T[0]
y_coords = coords.positions.T[1]
z_coords = coords.positions.T[2]
n_electrode = len([z for z in z_coords if z == z_coords[-1]])
np.savetxt('charges.dat',charges[-n_electrode:])
graphene_n_ions.plot_charges( x_coords[-n_electrode:], y_coords[-n_electrode:], charges[-n_electrode:] )

