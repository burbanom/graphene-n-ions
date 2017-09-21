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
from file_utils import return_value
from configurations import * 
from copy import deepcopy
import sys
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
#    for scf in ['OT','DIAG']:
#        if ranOK: break
#        if scf == 'OT': 
#            FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
#            FORCE_EVAL.DFT.SCF.OT.Preconditioner = 'FULL_KINETIC' 
#            FORCE_EVAL.DFT.SCF.OT.Linesearch = '2PNT'  
#            FORCE_EVAL.DFT.SCF.Max_scf = 15
#            FORCE_EVAL.DFT.SCF.OUTER_SCF.Max_scf = 20
#            FORCE_EVAL.DFT.SCF.OUTER_SCF.Eps_scf = eps_scf
#        elif scf == 'DIAG':
#            FORCE_EVAL.DFT.SCF.Max_scf = 300
#            FORCE_EVAL.DFT.SCF.DIAGONALIZATION.Algorithm = 'STANDARD'
#            FORCE_EVAL.DFT.SCF.Added_mos = [len(electrode), len(electrode)]
#            FORCE_EVAL.DFT.SCF.SMEAR.Method = 'FERMI_DIRAC'
#            FORCE_EVAL.DFT.SCF.SMEAR.Electronic_temperature = 300.0
#            FORCE_EVAL.DFT.SCF.MIXING.Method = 'BROYDEN_MIXING'
#            FORCE_EVAL.DFT.SCF.MIXING.Alpha = 0.2
#            FORCE_EVAL.DFT.SCF.MIXING.Beta = 1.5
#            FORCE_EVAL.DFT.SCF.MIXING.NBroyden = 8
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

if __name__ == '__main__':

    args = parse_commandline_arguments()
    debug = args.debug
    run_options = read_options('options.yml')
    # calculation parameters
    jobname = run_options['calculation']['jobname']
    coords = run_options['calculation']['coords']
    periodicity = run_options['calculation']['periodicity']
    bigbox = run_options['calculation']['bigbox']
    charge = run_options['calculation']['charge']
    lshift = run_options['calculation']['lshift']
    rshift = run_options['calculation']['rshift']
    if 'r_rotate' in run_options['calculation'].keys():
        degrees = run_options['calculation']['r_rotate']['degrees']
        axis = run_options['calculation']['r_rotate']['axis']
    else:
        degrees = 0.0
    z_len = run_options['calculation']['z_len']
    if 'add_electrode' in run_options['calculation'].keys():
        add_electrode = run_options['calculation']['add_electrode']
    else:
        add_electrode = False
    if 'tailored_box' in run_options['calculation'].keys():
        tailored_box = run_options['calculation']['tailored_box']
    else:
        tailored_box = False
    if 'pairs' in run_options['calculation'].keys():
        pairs = run_options['calculation']['pairs']['number']
        separation = run_options['calculation']['pairs']['separation']
        vector = run_options['calculation']['pairs']['vector']
        rotation_angle = run_options['calculation']['pairs']['rotation_angle']
    else:
        pairs = 0
        separation = 0. 
        vector = None
        rotation_angle = 0.0
    if 'l_ions' in run_options['calculation'].keys():
        l_ions = run_options['calculation']['l_ions']
    if 'r_ions' in run_options['calculation'].keys():
        r_ions = run_options['calculation']['r_ions']
    move_range_b = run_options['calculation']['move_range']['begin']
    move_range_e = run_options['calculation']['move_range']['end']
    move_range_s = run_options['calculation']['move_range']['step']
    move_direction = run_options['calculation']['move_direction']

    # XC parameters
    vdW = run_options['XC']['vdW']
    # scf parameters
    mgrid = run_options['scf']['mgrid']
    eps_scf = run_options['scf']['eps_scf']
    diagonalize = run_options['scf']['diagonalize']
    # basis set
    if 'kind' in run_options.keys() and 'basis' in run_options['kind'].keys():
        basis = run_options['kind']['basis']
    else:
        basis = 'DZVP-MOLOPT-GTH'

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

    FORCE_EVAL.DFT.Uks = True

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



    if bigbox: 
        bmim_pf6_opt = read(coords+'/opt_coords_bigbox.xyz')
        bmim_pf6_opt.cell = [44.20800018, 42.54000092, z_len]
    else:
        bmim_pf6_opt = read(coords+'/opt_coords.xyz')
        bmim_pf6_opt.cell = [22.104, 21.27, z_len]

    pf6 = bmim_pf6_opt[0:7]; bmim = bmim_pf6_opt[7:32]; electrode = bmim_pf6_opt[32:]
    pair = pf6+bmim
    pair.rotate('y',a=np.pi/2,center='COM')
    pair.rotate('x',a=np.pi/9.,center='COM')
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

    if pairs >= 1:
        generated_electrode = False
        cell = bmim_pf6_opt.cell
        lattice = build_lattice(pairs, motif=pair,lattice_constant=separation)
        l_confs = create_configurations(lattice,vector,rotation_angle)
        ######
        for key in l_confs.keys():
            if tailored_box:
                while not generated_electrode:
                    electrode = generate_electrode(l_confs[key], lattice_constant=separation, z_len=z_len)
                    cell = electrode.cell
                    generated_electrode = True
            l_confs[key].set_cell(cell)
            l_confs[key].center(axis=(0,1))
            if lshift != 0.0:
                l_confs[key].center(axis=2)
                l_confs[key].translate([0.0,0.0,lshift])
                if add_electrode and too_close(l_confs[key]+electrode,1.0):
                    (l_confs[key]+electrode).write('lhs_'+key+'_positions.vasp')
                    print('LHS ions are too close to the electrode')
                    sys.exit()

        lattice = build_lattice(pairs, motif=pair,lattice_constant=separation)
        r_confs = create_configurations(lattice,vector,rotation_angle)
        for key in r_confs.keys():
            r_confs[key].set_cell(cell)
            r_confs[key].rotate('y',a=np.pi,center='COP')
            r_confs[key].center(axis=(0,1,2))
            r_confs[key].translate([0.0,0.0,eq_dist_ion_pair_electrode])
            if rshift != 0.0:
                r_confs[key].center(axis=2)
                r_confs[key].translate([0.0,0.0,rshift])
                if add_electrode and too_close(r_confs[key]+electrode,1.0):
                    (r_confs[key]+electrode).write('rhs_'+key+'_positions.vasp')
                    print('RHS ions are too close to the electrode')
                    sys.exit()

    energies = pd.DataFrame(columns=l_confs.keys(),index=r_confs.keys())

    for  l_key in l_confs.keys():
        for  r_key in r_confs.keys():
            box = l_confs[l_key] + r_confs[r_key] 
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

            dir_name = calc.project_name + '-L-' + str(l_key) + '-R-' + str(r_key)
            energies[l_key][r_key] = run_calc(dir_name, calc, box=box, debug=debug)  
            energies.to_csv('results.csv')

    with open("options.yml") as f0:
        with open("results.csv", "a") as f1:
            for line in f0:
                f1.write('# '+ line)
