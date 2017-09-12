from __future__ import print_function
import numpy as np
from scipy import linalg
from ase.io import xyz
from ase import Atoms
import pandas as pd
import ase
from pycp2k import CP2K
import os, re, sys
import fnmatch
import shutil
from options import read_options

def parse_commandline_arguments():
    import argparse
    parser = argparse.ArgumentParser( description = 'cp2k calculator' )
    parser.add_argument( '--ncores', '-nc', metavar = 'N', type=int, required = True, help='set the number of cores to use for this calculation' )
    parser.add_argument( '--debug', '-d', metavar = 'B', type=bool, required = False, default = False, help='Do not run calcs, but write input files instead.' )

    return parser.parse_args()

def clean_files(path,pattern):
    all_files = os.listdir(path)
    filtered = fnmatch.filter(all_files,pattern+"*")
    for element in filtered:
        os.remove(os.path.join(path,element))

def return_value(filename,pattern):
    import mmap
    if type(pattern) is str:
        pattern = pattern.encode()
    with open(filename, "r") as fin:
        # memory-map the file, size 0 means whole file
        m = mmap.mmap(fin.fileno(), 0, prot=mmap.PROT_READ)
        #                             prot argument is *nix only
        i = m.rfind(pattern)
        try:
            m.seek(i)             # seek to the location
        except ValueError:
            return np.nan
        line = m.readline()   # read to the end of the line
    return float(line.split()[-1])

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

def too_close( mol, cutoff):
    # decide whether the number of atom distances below cutoff are
    # equal to the number of atoms.
    return not np.sum(mol.get_all_distances() < cutoff) == len(mol)

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
        calc.run()
        ranOK = True
        #break
    except subprocess.CalledProcessError:
        ranOK = False 
        #continue
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
    yshift = run_options['calculation']['yshift']
    zlen = run_options['calculation']['zlen']
    add_electrode = run_options['calculation']['add_electrode']
    l_ions = run_options['calculation']['l_ions']
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
        bmim_pf6_opt = ase.io.read(coords+'/opt_coords_bigbox.xyz')
        bmim_pf6_opt.cell = [44.20800018, 42.54000092, zlen]
    else:
        bmim_pf6_opt = ase.io.read(coords+'/opt_coords.xyz')
        bmim_pf6_opt.cell = [22.104, 21.27, zlen]

    lhs = Atoms(); rhs = Atoms()
    pf6 = bmim_pf6_opt[0:7]; bmim = bmim_pf6_opt[7:32]; electrode = bmim_pf6_opt[32:]

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
        #FORCE_EVAL.DFT.QS.Map_consistent = 'TRUE'
        #FORCE_EVAL.DFT.QS.Extrapolation = 'ASPC'
        #FORCE_EVAL.DFT.QS.Extrapolation_order = 3 
        #FORCE_EVAL.DFT.QS.Eps_default = 1.0E-10
        #FORCE_EVAL.DFT.QS.Eps_pgf_orb = 1.0E-07

    for ion in l_ions:
        if ion == 'A':
            lhs.extend(pf6)
        elif ion == 'C':
            lhs.extend(bmim)

    for ion in r_ions:
        if ion == 'A':
            rhs.extend(pf6)
        elif ion == 'C':
            rhs.extend(bmim)

    lhs.set_cell(bmim_pf6_opt.cell)
    rhs.set_cell(bmim_pf6_opt.cell)
    rhs.rotate(v = 'y', a= - 2 * np.pi, center='COM' )
    rhs.translate([0.0,0.0,2*abs(zlen/2.0-rhs.get_center_of_mass()[2])])

    lhs.center(axis=(0,1))
    rhs.center(axis=(0,1))

    if lshift != 0.0:
        lhs.center(axis=2,about=electrode.get_center_of_mass())
        lhs.translate([0.0,0.0,lshift])
        if add_electrode and too_close(lhs+electrode,1.0):
            print('LHS ions are too close to the electrode')
            sys.exit()

    if rshift != 0.0:
        rhs.center(axis=2,about=electrode.get_center_of_mass())
        rhs.translate([0.0,0.0,rshift])
        if add_electrode and too_close(rhs+electrode,1.0):
            print('RHS ions are too close to the electrode')
            sys.exit()

    if abs(yshift) >= bmim_pf6_opt.get_cell_lengths_and_angles()[2] / 2.0:
        print('yshift is too large for this box size')
        sys.exit()

    lhs.translate([0.0,yshift/2.0,0.0])
    rhs.translate([0.0,-yshift/2.0,0.0])

    translate_range = np.arange(move_range_b,move_range_e+move_range_s,move_range_s)
    rhs_list = translate_ions( rhs, translate_range, move_direction )

    energies = pd.DataFrame(columns=['energies'], index=translate_range)

    for index, conf in enumerate(translate_range):
        box = lhs + rhs_list[index]
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

        dir_name = calc.project_name + '-' + str(conf)
        energies['energies'][conf] = run_calc(dir_name, calc, box, debug)  
        energies.to_csv('results.csv')

    with open("options.yml") as f0:
        with open("results.csv", "a") as f1:
            for line in f0:
                f1.write('# '+ line)
