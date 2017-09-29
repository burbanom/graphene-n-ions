#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
from scipy import linalg
from ase.io import read, write
from ase import Atoms
import pandas as pd
import ase
import os, re, sys
from yml_options import read_options
from file_utils import *
from configurations import * 
from copy import deepcopy
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from cp2k_calc import Cp2k_calc

def parse_commandline_arguments():
    import argparse
    parser = argparse.ArgumentParser( description = 'cp2k calculator' )
    parser.add_argument( '--ncores', '-nc', metavar = 'N', type=int, required = True, help='set the number of cores to use for this calculation' )
    parser.add_argument( '--debug', '-d', action='store_true', required = False, help='Do not run calcs, but write input files instead.' )

    return parser.parse_args()


def plot_charges( x_coords, y_coords, charges, folder = './', points = None):
    plt.figure(figsize=(7,7))
    cm = plt.cm.get_cmap('seismic')
    plt.scatter( x_coords, y_coords, c=charges, vmin=np.min(charges), vmax=np.max(charges), s=55, cmap=cm)
    plt.colorbar(shrink=1.00)
    if points is not None:
        for point in points:
            plt.scatter(point[0],point[1], marker=u'*', s=75)
    plt.ylabel(r'position along $y$-axis ($\AA$)')
    plt.xlabel(r'position along $x$-axis ($\AA$)')
    #plt.axes().set_aspect('equal')
    plt.axes().set_aspect(aspect = np.max(y_coords)/np.max(x_coords), adjustable='box', anchor = 'C')
    plt.xlim([0.0,np.max(x_coords)])
    plt.ylim([0.0,np.max(y_coords)])
    plt.savefig(folder+'charges.pdf',dpi=1000)
    return

if __name__ == '__main__':

    args = parse_commandline_arguments()
    ncores = args.ncores
    debug = args.debug
    run_options = read_options('options.yml')
    ############################################################################
    # calculation parameters
    try:
        jobname = run_options['calculation']['jobname']
    except:
        jobname = eng
    try:
        coords_folder = run_options['calculation']['coords_folder']
    except:
        coords_folder = './' 
    try:
        coords_file = run_options['calculation']['coords_file']
    except:
        coords_file = 'BMIM-PF6.xyz'
    #########################################################################
    try:
        calc_mode = run_options['calculation']['calc_modes']
        if 'flips' in run_options['calculation']['calc_modes'].keys():
            calc_mode = 'flips'
            try:
                flips_vector = run_options['calculation']['calc_modes']['flips']['vector']
                flips_rotation_angle = run_options['calculation']['calc_modes']['flips']['rotation_angle']
            except:
                flips_vector = x 
                flips_rotation_angle = 180.0
        elif 'translation' in run_options['calculation']['calc_modes'].keys():
            calc_mode = 'translation'
            try:
                trans_direct = run_options['calculation']['calc_modes']['translation']['direction']
                trans_begin = run_options['calculation']['calc_modes']['translation']['begin']
                trans_end = run_options['calculation']['calc_modes']['translation']['end']
                trans_step = run_options['calculation']['calc_modes']['translation']['step']
            except:
                trans_direct = 'z' 
                trans_begin = 0.0 
                trans_end = 10.0 
                trans_step = 1.0 
        else:
            calc_mode = 'single'
    except:
        calc_mode = 'single'
    #########################################################################
    try:
        periodicity = run_options['calculation']['box']['periodicity']
    except:
        periodicity = 2
    try:
        x_len = run_options['calculation']['box']['x_len']
    except:
        x_len = None
    try:
        y_len = run_options['calculation']['box']['y_len']
    except:
        y_len = None 
    try:
        z_len = run_options['calculation']['box']['z_len']
    except:
        z_len = 40.0
    try: 
        add_electrode = run_options['calculation']['box']['add_electrode']
    except:
        add_electrode = False
    #########################################################################
    try:
        charge = run_options['calculation']['charge']
    except:
        charge = 0
    try:
        r_shift = run_options['calculation']['r_shift']
    except:
        r_shift = 0.0
    else:
        degrees = 0.0
    #########################################################################
    if 'l_pairs' in run_options['calculation'].keys():
        l_pairs = run_options['calculation']['l_pairs']['number']

        try:
            l_shift = run_options['calculation']['l_pairs']['initial_shift']
        except:
            l_shift = 0.0

        if l_pairs >= 1:
            l_pair_separation = run_options['calculation']['l_pairs']['pair_separation']
        else:
            l_pair_separation = 0

        try:
            l_init_rot_axes = run_options['calculation']['l_pairs']['initial_rotations']['axes']
            l_init_rot_angles = run_options['calculation']['l_pairs']['initial_rotations']['angles']
        except:
            l_init_rot_axes = []
            l_init_rot_angles = [] 
    else:
        l_pairs = 0
        l_pair_separation = 0. 

    #########################################################################
    if 'r_pairs' in run_options['calculation'].keys():
        r_pairs = run_options['calculation']['r_pairs']['number']

        try:
            r_shift = run_options['calculation']['r_pairs']['initial_shift']
        except:
            r_shift = 0.0

        if r_pairs >= 1:
            r_pair_separation = run_options['calculation']['r_pairs']['pair_separation']
        else:
            r_pair_separation = 0

        try:
            r_init_rot_axes = run_options['calculation']['r_pairs']['initial_rotations']['axes']
            r_init_rot_angles = run_options['calculation']['r_pairs']['initial_rotations']['angles']
        except:
            r_init_rot_axes = []
            r_init_rot_angles = [] 
    else:
        r_pairs = 0
        r_pair_separation = 0. 

    # XC parameters
    try:
        vdW = run_options['XC']['vdW']
    except:
        vdW = True 
    # scf parameters
    try:
        mgrid = run_options['scf']['mgrid']
    except:
        mgrid = 280 
    try:
        eps_scf = run_options['scf']['eps_scf']
    except:
        eps_scf = 1.0E-05 
    try:
        diagonalize = run_options['scf']['diagonalize']
    except:
        diagonalize = False 
    # basis set
    try: 
        basis = run_options['kind']['basis']
    except:
        basis = 'DZVP-MOLOPT-GTH'
    try:
        spin_polarized = run_options['dft']['spin_polarized']
    except:
        spin_polarized = False

    bmim_pf6 = read(coords_folder+'/'+coords_file)

    pf6 = bmim_pf6[0:7]; bmim = bmim_pf6[7:32]; electrode = bmim_pf6[32:]
    pair = pf6+bmim

    eq_dist_ion_pair_electrode = abs(electrode.get_center_of_mass()-pair.get_center_of_mass())[2]
    electrode.center()
    eq_dist_ion_pair = linalg.norm(bmim.get_center_of_mass() - pf6.get_center_of_mass())

    electrode = generate_electrode(pair, lattice_constant=l_pairs*l_pair_separation, z_len=z_len, x_in=x_len, y_in=y_len)
    cell = electrode.cell

    if l_pairs >= 1:
        l_pair = mol_setup(pair,-eq_dist_ion_pair_electrode,l_init_rot_axes,l_init_rot_angles,cell=cell)
        l_lattice = build_lattice(l_pairs, motif=l_pair,lattice_constant=l_pair_separation)

    if r_pairs >=1:
        r_pair = mol_setup(pair,eq_dist_ion_pair_electrode,r_init_rot_axes,r_init_rot_angles,cell=cell)
        r_lattice = build_lattice(r_pairs, motif=r_pair,lattice_constant=r_pair_separation)

    if calc_mode == 'flips':

        if l_pairs: 
            l_confs = create_configurations(l_lattice,flips_vector,flips_rotation_angle)
            for key in l_confs.keys():
               cell = electrode.cell
               l_confs[key].set_cell(cell)
               l_confs[key].center(axis=(0,1))
               if l_shift != 0.0:
                   l_confs[key].center(axis=2)
                   l_confs[key].translate([0.0,0.0,l_shift])

        if r_pairs: 
            r_confs = create_configurations(r_lattice,flips_vector,flips_rotation_angle)

            for key in r_confs.keys():
               r_confs[key].set_cell(cell)
               r_confs[key].center(axis=(0,1))
               if r_shift != 0.0:
                   r_confs[key].center(axis=2)
                   r_confs[key].translate([0.0,0.0,r_shift])

    elif calc_mode == 'translation':

        if l_pairs: 
            translate_range = np.arange(trans_begin,trans_end+trans_step,trans_step)
            l_confs = translate_ions(l_lattice,translate_range,trans_direct)
            for key in l_confs.keys():
               cell = electrode.cell
               l_confs[key].set_cell(cell)
               if l_shift != 0.0:
                   l_confs[key].center(axis=2)
                   l_confs[key].translate([0.0,0.0,l_shift])

        if r_pairs: 
            r_confs = create_configurations(r_lattice,'z',180.0)

            for key in r_confs.keys():
               r_confs[key].set_cell(cell)
               r_confs[key].center(axis=(0,1))
               if r_shift != 0.0:
                   r_confs[key].center(axis=2)
                   r_confs[key].translate([0.0,0.0,r_shift])



    if l_pairs:
        indices = l_confs.keys()
    else:
        indices = ['configuration']

    if r_pairs:
        columns = r_confs.keys()
    else:
        columns = ['energy']

    energies = pd.DataFrame(columns=columns,index=indices)

    if add_electrode:
        added_MOs = len(electrode)
    else:
        added_MOs = 20 

    for  index in indices:
        for  col in columns:

            calc = Cp2k_calc( jobname=jobname, ncores=ncores, mgrid=mgrid, 
                    eps_scf=eps_scf, charge=charge, periodicity=periodicity, vdW=vdW, 
                    basis=basis, diagonalize=diagonalize, spin_polarized=spin_polarized, 
                    added_MOs=added_MOs, debug=debug )

            box = Atoms()
            box.set_cell(cell)

            if l_pairs:
                box.extend(l_confs[index])

            if r_pairs:
                box.extend(r_confs[col])

            if add_electrode:
                box.extend(electrode)

            if too_close(box,1.0):
                box.write('box.cif')
                print('Some atoms are too close')
                sys.exit()


            dir_name = jobname + '-L-' + str(index) + '-R-' + str(col)
            energies[col][index] = calc.run_calc(dir_name, box)  
            if add_electrode:
                try:
                    charges = calc.get_charges( -len(electrode) )
                    np.savetxt(dir_name+'/'+'charges.dat',charges)
                    plot_charges( electrode.get_positions().T[0], electrode.get_positions().T[1], charges=charges, folder=dir_name+'/' )
                except:
                    continue
            energies.to_csv('results.csv')

    with open("options.yml") as f0:
        with open("results.csv", "a") as f1:
            for line in f0:
                f1.write('# '+ line)
