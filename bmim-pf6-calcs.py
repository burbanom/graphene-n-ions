from __future__ import print_function
import numpy as np
from scipy import linalg
from ase.io import xyz
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
    parser.add_argument( '--debug', '-n', metavar = 'B', type=bool, required = False, default = False, help='Do not run calcs, but write input files instead.' )

    return parser.parse_args()

def clean_files(path,pattern):
    all_files = os.listdir(path)
    filtered = fnmatch.filter(all_files,pattern+"*")
    for element in filtered:
        os.remove(os.path.join(path,element))

def return_value(filename,pattern):
    import mmap
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
    return line.split()[-1]

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

def run_calc( my_dir, calc, box, debug=False ):
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
    for scf in ['OT','DIAG']:
        if ranOK: break
        if scf == 'OT': 
            FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
            FORCE_EVAL.DFT.SCF.OT.Preconditioner = 'FULL_KINETIC' 
            FORCE_EVAL.DFT.SCF.OT.Linesearch = '2PNT'  
            FORCE_EVAL.DFT.SCF.Max_scf = 20
            FORCE_EVAL.DFT.SCF.OUTER_SCF.Max_scf = 16
            FORCE_EVAL.DFT.SCF.OUTER_SCF.Eps_scf = 1.0E-05
        elif scf == 'DIAG':
            FORCE_EVAL.DFT.SCF.Max_scf = 300
            FORCE_EVAL.DFT.SCF.DIAGONALIZATION.Algorithm = 'STANDARD'
            FORCE_EVAL.DFT.SCF.Added_mos = [len(electrodes), len(electrodes)]
            FORCE_EVAL.DFT.SCF.SMEAR.Method = 'FERMI_DIRAC'
            FORCE_EVAL.DFT.SCF.SMEAR.Electronic_temperature = 300.0
            FORCE_EVAL.DFT.SCF.MIXING.Method = 'BROYDEN_MIXING'
            FORCE_EVAL.DFT.SCF.MIXING.Alpha = 0.2
            FORCE_EVAL.DFT.SCF.MIXING.Beta = 1.5
            FORCE_EVAL.DFT.SCF.MIXING.NBROYDEN = 8
        try:
            calc.run()
            ranOK = True
            break
        #except subprocess.CalledProcessError, e:
        except subprocess.CalledProcessError:
            ranOK = False 
            #FORCE_EVAL.DFT.SCF.Scf_guess = 'RESTART'
            continue
    result = return_value(calc.output_path,eng_string)
    os.chdir(root_dir)
    return result 

if __name__ == '__main__':

    args = parse_commandline_arguments()
    run_options = read_options('options.yml')
    # calculation parameters
    jobname = run_options['calculation']['jobname']
    threeD = run_options['calculation']['3D']
    bigbox = run_options['calculation']['bigbox']
    neutral = run_options['calculation']['neutral']
    lshift = run_options['calculation']['lshift']
    rshift = run_options['calculation']['rshift']
    yshift = run_options['calculation']['rshift']
    zlen = run_options['calculation']['rshift']
    add_electrode = run_options['calculation']['add_electrode']
    l_ions = run_options['calculation']['l_ions']
    r_ions = run_options['calculation']['r_ions']
    move_range = run_options['calculation']['move_range']
    move_direction = run_options['calculation']['move_range']

    # XC parameters
    vdW = run_options['XC']['vdW']
    with open("options.yml") as f0:
        with open("results.csv", "w") as f1:
            for line in f0:
                f1.write('# '+ line)
    results = open('results.csv', 'a')

    root_dir = os.getcwd()
    calc = CP2K()
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
#
#    FORCE_EVAL.DFT.MGRID.Cutoff = args.mgrid
#    FORCE_EVAL.DFT.QS.Method = 'GPW'
#    #FORCE_EVAL.DFT.QS.Map_consistent = 'TRUE'
#    #FORCE_EVAL.DFT.QS.Extrapolation = 'ASPC'
#    #FORCE_EVAL.DFT.QS.Extrapolation_order = 3 
#    #FORCE_EVAL.DFT.QS.Eps_default = 1.0E-10
#    #FORCE_EVAL.DFT.QS.Eps_pgf_orb = 1.0E-07
#    FORCE_EVAL.DFT.SCF.Scf_guess = 'ATOMIC'
#    FORCE_EVAL.DFT.SCF.Eps_scf = 1.0E-05
#
#    FORCE_EVAL.DFT.XC.XC_FUNCTIONAL.Section_parameters = "PBE"
#    FORCE_EVAL.DFT.Uks = True
#
#    KIND = SUBSYS.KIND_add("H")
#    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
#    KIND.Potential = 'GTH-PBE'
#
#    KIND = SUBSYS.KIND_add("C")
#    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
#    KIND.Potential = 'GTH-PBE'
#
#    KIND = SUBSYS.KIND_add("N")
#    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
#    KIND.Potential = 'GTH-PBE'
#
#    KIND = SUBSYS.KIND_add("P")
#    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
#    KIND.Potential = 'GTH-PBE'
#
#    KIND = SUBSYS.KIND_add("F")
#    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
#    KIND.Potential = 'GTH-PBE'
#
#    GLOBAL.Run_type = 'ENERGY'
#
#
#
#    if bigbox: 
#        bmim_pf6_opt = ase.io.read('/gpfshome/mds/staff/mburbano/bmim-pf6/after-optimization/single-ions/opt_coords_bigbox.xyz')
#        bmim_pf6_opt.cell = [44.20800018, 42.54000092, zlen]
#    else:
#        bmim_pf6_opt = ase.io.read('/gpfshome/mds/staff/mburbano/bmim-pf6/after-optimization/single-ions/opt_coords.xyz')
#        bmim_pf6_opt.cell = [22.104, 21.27, zlen]
#
#    pf6 = bmim_pf6_opt[0:7]; bmim = bmim_pf6_opt[7:32]; electrode = bmim_pf6_opt[32:]
#    pf6_L = pf6.copy()
#    bmim_L = bmim.copy()
#    pf6_R = pf6.copy()
#    bmim_R = bmim.copy()
#    shift_vector = pf6.get_center_of_mass() - bmim.get_center_of_mass()
#    electrode_COM = electrode.get_center_of_mass()
#
#    if lshift != 0.0:
#        pf6_L = pf6_L.center(about=electrode.get_center_of_mass())
#        pf6_L = pf6.translate([0.0,0.0,lshift])
#        if too_close(pf6_L+electrode,1.0):
#            print('pf6_L is to close to the electrode')
#            sys.exit()
#
#    bmim_L = bmim.translate([0.0,0.0,lshift])
#    bmim_L = bmim_L.center(about=electrode.get_center_of_mass())
#    if too_close(bmim_L+electrode,1.0):
#        print('bmim_L is to close to the electrode')
#        sys.exit()
#
#    pf6_R = pf6.translate([0.0,0.0,rshift])
#    if too_close(pf6_R+electrode,1.0):
#        print('pf6_R is to close to the electrode')
#        sys.exit()
#
#    bmim_R = bmim.translate([0.0,0.0,rshift])
#    if too_close(bmim_R+electrode,1.0):
#        print('bmim_R is to close to the electrode')
#        sys.exit()
#
#    try:
#        z_translate_range = np.arange(zrange[0],zrange[1]+zrange[2],zrange[2])
#        energies = pd.DataFrame(columns=run_types, index=z_translate_range)
#        my_range = z_translate_range
#    except NameError:
#        pass
#    try:
#        y_translate_range = np.arange(yrange[0],yrange[1]+yrange[2],yrange[2])
#        energies = pd.DataFrame(columns=run_types, index=y_translate_range)
#        my_range = y_translate_range
#    except NameError:
#        print( 'No ranges were attributed' )
#
#    try:
#        pf6_list_L = z_translate( pf6_L, - z_translate_range )
#        bmim_list_L = z_translate( bmim_L, - z_translate_range )
#        ###################################################################
#        pf6_R.rotate(v = 'y', a= - 2 * np.pi, center='COM' )#; pf6_R.wrap()
#        pf6_R.translate([0.0,0.0,2 * abs(pf6_COM[2] - 50.0) ])
#        pf6_R_o = pf6_R.copy()
#        pf6_list_R = z_translate( pf6_R, z_translate_range )
#        ###################################################################
#        pf6_R_o.translate([0.0,- shift_vector[1], 0.0])
#        pf6_list_R_o = z_translate( pf6_R_o, z_translate_range )
#        ###################################################################
#        pf6_R_c = pf6_closer.copy()
#        pf6_R_c.rotate(v = 'y', a= - np.pi, center='COM' )#; pf6_R.wrap()
#        pf6_R_c.translate([0.0,0.0,2 * 3.5 ])
#        pf6_R_o_c = pf6_R_c.copy()
#        pf6_list_R_c = z_translate( pf6_R_c, z_translate_range )
#        ###################################################################
#        pf6_R_o_c.translate([0.0,- shift_vector[1], 0.0])
#        pf6_list_R_o_c = z_translate( pf6_R_o_c, z_translate_range )
#        ###################################################################
#        bmim_R.rotate(v = 'y', a= - np.pi, center='COM' )#; bmim_R.wrap()
#        bmim_R.translate([0.0,0.0,2 * abs(bmim_COM[2] - 50.0)])
#        bmim_R_o = bmim_R.copy()
#        bmim_list_R = z_translate( bmim_R, z_translate_range )
#        ###################################################################
#        bmim_R_o.translate([0.0, shift_vector[1], 0.0])
#        bmim_list_R_o = z_translate( bmim_R_o, z_translate_range )
#        ###################################################################
#    except NameError:
#        pass
#    try:
#        pf6_R_y = pf6.copy()
#        pf6_R_y.rotate(v = 'y', a= - np.pi, center='COM' )#; pf6_R.wrap()
#        pf6_R_y.translate([0.0,0.0,2 * abs(pf6_COM[2] - 50.0) ])
#        pf6_list_R_y = y_translate( pf6_R_y, y_translate_range )
#    except NameError:
#        pass
#    ###################################################################
#
#    for column in run_types:
#        for index, conf in enumerate(my_range):
#            # with electrode
#            if column == 'a_w_electrode':
#                box = pf6_list_L[index] + electrode
#                if not neutral: 
#                    FORCE_EVAL.DFT.Charge = -1 
#            elif column == 'c_w_electrode':
#                box = bmim_list_L[index] + electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = +1 
#            elif column == 'aa_w_electrode_f':
#                box = pf6 + pf6_list_R[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'aa_w_electrode_o':
#                box = pf6 + pf6_list_R_o[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'aa_w_electrode_f_c':
#                box = pf6_closer + pf6_list_R_c[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'aa_w_electrode_o_c':
#                box = pf6_closer + pf6_list_R_o_c[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'aa_wo_electrode_f_c':
#                box = pf6_closer + pf6_list_R_c[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'aa_wo_electrode_o_c':
#                box = pf6_closer + pf6_list_R_o_c[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'cc_w_electrode_f':
#                box = bmim + bmim_list_R[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = +2
#            elif column == 'cc_w_electrode_o':
#                box = bmim + bmim_list_R_o[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = +2
#            elif column == 'ac_w_electrode_f':
#                box = pf6 + bmim_list_R[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = 0
#            elif column == 'ac_w_electrode_o':
#                box = pf6 + bmim_list_R_o[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = 0
#            elif column == 'aa_wo_electrode_f':
#                box = pf6 + pf6_list_R[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'aa_wo_electrode_o':
#                box = pf6 + pf6_list_R_o[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'cc_wo_electrode_f':
#                box = bmim + bmim_list_R[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = +2
#            elif column == 'cc_wo_electrode_o':
#                box = bmim + bmim_list_R_o[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = +2
#            elif column == 'ac_wo_electrode_f':
#                box = pf6 + bmim_list_R[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = 0
#            elif column == 'ac_wo_electrode_o':
#                box = pf6 + bmim_list_R_o[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = 0
#            elif column == 'aa_w_electrode_y':
#                box = pf6 + pf6_list_R_y[index] +  electrode
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#            elif column == 'aa_wo_electrode_y':
#                box = pf6 + pf6_list_R_y[index]
#                if not neutral:
#                    FORCE_EVAL.DFT.Charge = -2
#
#            if threeD:
#                FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
#                FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
#                box.pbc = [True,True,True]
#            else:
#                FORCE_EVAL.DFT.POISSON.Periodic = 'XY'
#                FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'
#                box.pbc = [True,True,False]
#
#            dir_name = calc.project_name + '-' + column + str(conf)
#            energies[column][conf] = run_calc(dir_name, calc, box)  
#            energies.to_csv(results)
    results.close()
