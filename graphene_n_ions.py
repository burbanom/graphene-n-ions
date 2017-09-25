#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
from scipy import linalg
from ase.io import read, write
from ase import Atoms
import pandas as pd
import ase
import os, re, sys
import shutil
from yml_options import read_options
from file_utils import *
from configurations import * 
from copy import deepcopy
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
#sys.path.insert(0,'../pycp2k')
#sys.path.insert(0,'../python-utils')
from pycp2k import CP2K

def parse_commandline_arguments():
    import argparse
    parser = argparse.ArgumentParser( description = 'cp2k calculator' )
    parser.add_argument( '--ncores', '-nc', metavar = 'N', type=int, required = True, help='set the number of cores to use for this calculation' )
    parser.add_argument( '--debug', '-d', action='store_true', required = False, help='Do not run calcs, but write input files instead.' )

    return parser.parse_args()

def run_calc( my_dir, calc, box, debug = False ):
    # This function will run the calculation intitially using the 
    # default FULL_KINETIC and 2PNT settings. If these fail, other
    # linesearches are performed before changing the preconditioner.
    # ranOK is used to break out of the loops once the calculation has
    # finished without errors.
    import subprocess
    eng_string = "ENERGY| Total FORCE_EVAL"
    if os.path.exists(my_dir):
        shutil.rmtree(my_dir)
    os.makedirs(my_dir)
    os.chdir(my_dir)
    calc.create_cell(SUBSYS,box)
    calc.create_coord(SUBSYS,box)
    box.write(my_dir+'.cif')
    box.write(my_dir+'.xyz')
    result = np.nan 
    ranOK = False
    try:
        if debug:
            calc.write_input_file()
        else:
            calc.run()
            ranOK = True
    except subprocess.CalledProcessError:
        ranOK = False 
    #######################################################
    if ranOK:
        result = return_value(calc.output_path,eng_string)
    else:
        result = np.nan
    os.chdir(root_dir)
    return result 

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
        l_shift = run_options['calculation']['l_shift']
    except:
        l_shift = 0.0
    try:
        r_shift = run_options['calculation']['r_shift']
    except:
        r_shift = 0.0
    if 'r_rotate' in run_options['calculation'].keys():
        degrees = run_options['calculation']['r_rotate']['degrees']
        axis = run_options['calculation']['r_rotate']['axis']
    else:
        degrees = 0.0
    if 'l_pairs' in run_options['calculation'].keys():
        l_pairs = run_options['calculation']['l_pairs']['number']
        separation = run_options['calculation']['l_pairs']['separation']
        vector = run_options['calculation']['l_pairs']['vector']
        rotation_angle = run_options['calculation']['l_pairs']['rotation_angle']
        try:
            initial_rot_axes = run_options['calculation']['l_pairs']['initial_rotations']['axes']
            initial_rot_angles = run_options['calculation']['l_pairs']['initial_rotations']['angles']
        except:
            initial_rot_axes = []
            initial_rot_angles = [] 
    else:
        l_pairs = 0
        separation = 0. 
        vector = None
        rotation_angle = 0.0

    try:
        r_pairs = run_options['calculation']['r_pairs']['number']
    except:
        r_pairs = 0 

    if r_pairs:
        if 'r_move_range' in run_options['calculation']['r_pairs'].keys():
            r_translate = True
            try:
                r_move_range_b = run_options['calculation']['r_pairs']['r_move_range']['begin']
            except:
                r_move_range_b = 0.0 
            try:
                r_move_range_e = run_options['calculation']['r_pairs']['r_move_range']['end']
            except:
                r_move_range_e = 0.0 
            try:
                r_move_range_s = run_options['calculation']['r_pairs']['r_move_range']['step']
            except:
                r_move_range_s = 1.0 
            try:
                r_move_direction = run_options['calculation']['r_pairs']['r_move_range']['direction']
            except:
                r_move_direction = 'z' 
        else:
            r_translate = False

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

    ############################################################################
    root_dir = os.getcwd()
    calc = CP2K()
    calc.project_name = jobname
    calc.mpi_n_processes = args.ncores
    calc.working_directory = "./"
    CP2K_INPUT = calc.CP2K_INPUT
    GLOBAL = CP2K_INPUT.GLOBAL
    FORCE_EVAL = CP2K_INPUT.FORCE_EVAL_add()
    SUBSYS = FORCE_EVAL.SUBSYS
    MOTION = CP2K_INPUT.MOTION
    GLOBAL.Extended_fft_lengths = True

    FORCE_EVAL.Method = 'QS'

    FORCE_EVAL.DFT.Basis_set_file_name = 'BASIS_MOLOPT'
    FORCE_EVAL.DFT.Potential_file_name = 'GTH_POTENTIALS'

    FORCE_EVAL.DFT.MGRID.Cutoff = mgrid
    FORCE_EVAL.DFT.QS.Method = 'GPW'
    FORCE_EVAL.DFT.SCF.Scf_guess = 'ATOMIC'
    FORCE_EVAL.DFT.SCF.Eps_scf = eps_scf 

    if not vdW:
        FORCE_EVAL.DFT.XC.XC_FUNCTIONAL.PBE.Parametrization = 'ORIG'
    else:
        FORCE_EVAL.DFT.XC.XC_FUNCTIONAL.PBE.Parametrization = 'revPBE'
        FORCE_EVAL.DFT.XC.XC_FUNCTIONAL.PBE.Scale_c = 0.0
        FORCE_EVAL.DFT.XC.XC_FUNCTIONAL.VWN.Scale_c = 1.0
        FORCE_EVAL.DFT.XC.VDW_POTENTIAL.Dispersion_functional = 'NON_LOCAL'
        NON_LOCAL = FORCE_EVAL.DFT.XC.VDW_POTENTIAL.NON_LOCAL_add()
        NON_LOCAL.Type = 'DRSLL'
        NON_LOCAL.Kernel_file_name = 'vdW_kernel_table.dat'
        NON_LOCAL.Cutoff =  20.0 

    FORCE_EVAL.DFT.Uks = spin_polarized

    KIND = SUBSYS.KIND_add("H")
    KIND.Basis_set = basis 
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("C")
    KIND.Basis_set = basis 
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("N")
    KIND.Basis_set = basis 
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("P")
    KIND.Basis_set = basis 
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("F")
    KIND.Basis_set = basis 
    KIND.Potential = 'GTH-PBE'

    GLOBAL.Run_type = 'ENERGY'



    bmim_pf6 = read(coords_folder+'/'+coords_file)

    pf6 = bmim_pf6[0:7]; bmim = bmim_pf6[7:32]; electrode = bmim_pf6[32:]
    pair = pf6+bmim

    for axis, angle in zip(initial_rot_axes,initial_rot_angles):
        pair.rotate(v=axis,a= angle * (np.pi/ 180.),center='COP')

    eq_dist_ion_pair_electrode = abs(electrode.get_center_of_mass()-pair.get_center_of_mass())[2]
    electrode.center()
    pair.center()
    pair.translate([0.0,0.0,-eq_dist_ion_pair_electrode])
    #eq_dist_ion_pair = linalg.norm(bmim.get_center_of_mass() - pf6.get_center_of_mass())

    if diagonalize:
        FORCE_EVAL.DFT.SCF.Max_scf = 300
        FORCE_EVAL.DFT.SCF.DIAGONALIZATION.Algorithm = 'STANDARD'
        FORCE_EVAL.DFT.SCF.Added_mos = [len(electrode), len(electrode)]
        FORCE_EVAL.DFT.SCF.SMEAR.Method = 'FERMI_DIRAC'
        FORCE_EVAL.DFT.SCF.SMEAR.Electronic_temperature = 300.0
        FORCE_EVAL.DFT.SCF.MIXING.Method = 'BROYDEN_MIXING'
        FORCE_EVAL.DFT.SCF.MIXING.Alpha = 0.2
        FORCE_EVAL.DFT.SCF.MIXING.Beta = 1.5
        FORCE_EVAL.DFT.SCF.MIXING.Nbroyden = 8
    else:
        FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
        FORCE_EVAL.DFT.SCF.OT.Preconditioner = 'FULL_KINETIC' 
        FORCE_EVAL.DFT.SCF.OT.Linesearch = '2PNT'  
        FORCE_EVAL.DFT.SCF.Max_scf = 15
        FORCE_EVAL.DFT.SCF.OUTER_SCF.Max_scf = 20
        FORCE_EVAL.DFT.SCF.OUTER_SCF.Eps_scf = eps_scf

    if l_pairs >= 1:
        generated_electrode = False
        cell = bmim_pf6.cell
        lattice = build_lattice(l_pairs, motif=pair,lattice_constant=separation)
        l_confs = create_configurations(lattice,vector,rotation_angle)
        ######
        for key in l_confs.keys():
            while not generated_electrode:
                electrode = generate_electrode(l_confs[key], lattice_constant=separation, x_in=x_len, y_in=y_len, z_len=z_len)
                cell = electrode.cell
                generated_electrode = True
            l_confs[key].set_cell(cell)
            l_confs[key].center(axis=(0,1))
            if l_shift != 0.0:
                l_confs[key].center(axis=2)
                l_confs[key].translate([0.0,0.0,l_shift])
                if add_electrode and too_close(l_confs[key]+electrode,1.0):
                    (l_confs[key]+electrode).write('lhs_'+key+'_positions.vasp')
                    print('LHS ions are too close to the electrode')
                    sys.exit()

    if r_pairs >=1:
        lattice = build_lattice(r_pairs, motif=pair,lattice_constant=separation)
        r_confs = create_configurations(lattice,vector,rotation_angle)
        for key in r_confs.keys():
            r_confs[key].set_cell(cell)
            r_confs[key].rotate('y',a=np.pi,center='COP')
            r_confs[key].center(axis=(0,1,2))
            r_confs[key].translate([0.0,0.0,eq_dist_ion_pair_electrode])
            if r_shift != 0.0:
                r_confs[key].center(axis=2)
                r_confs[key].translate([0.0,0.0,r_shift])
                if add_electrode and too_close(r_confs[key]+electrode,1.0):
                    (r_confs[key]+electrode).write('rhs_'+key+'_positions.vasp')
                    print('RHS ions are too close to the electrode')
                    sys.exit()

    columns = l_confs.keys()
    if r_pairs:
        indices = r_confs.keys()
    else:
        indices = ['energy']
    energies = pd.DataFrame(columns=columns,index=indices)

    for  col in columns:
        for  index in indices:
            box = l_confs[col] 
            if r_pairs:
                box.extend(r_confs[index])
            if add_electrode:
                box.extend(electrode)
            FORCE_EVAL.DFT.Charge = charge
            if periodicity == 3:
                FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
                FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
                box.pbc = [True,True,True]
            elif periodicity == 2:
                FORCE_EVAL.DFT.POISSON.Periodic = 'XY'
                FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'
                box.pbc = [True,True,False]
            elif periodicity == 0:
                FORCE_EVAL.DFT.POISSON.Periodic = 'NONE'
                FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'
                box.pbc = [False,False,False]

            dir_name = calc.project_name + '-L-' + str(col) + '-R-' + str(index)
            energies[col][index] = run_calc(dir_name, calc, box=box, debug=debug)  
            if add_electrode:
                try:
                    charges = hirshfeld_charges(dir_name+'/'+calc.project_name+'.out')[-len(electrode):]
                    np.savetxt(dir_name+'/'+'charges.dat',charges)
                    plot_charges( electrode.get_positions().T[0], electrode.get_positions().T[1], charges=charges, folder=dir_name+'/' )
                except:
                    continue
            energies.to_csv('results.csv')

    with open("options.yml") as f0:
        with open("results.csv", "a") as f1:
            for line in f0:
                f1.write('# '+ line)
