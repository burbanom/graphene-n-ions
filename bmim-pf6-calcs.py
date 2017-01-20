from __future__ import print_function
import numpy as np
from scipy import linalg
from ase.io import xyz
import pandas as pd
import ase
from pycp2k import CP2K
import os, re
import mmap
import fnmatch
import shutil

def parse_commandline_arguments():
    import argparse
    parser = argparse.ArgumentParser( description = 'cp2k calculator' )
    parser.add_argument( '--ncores', '-nc', metavar = 'N', type=int, required = True, help='set the number of cores to use for this calculation' )
    #parser.add_argument( '--path', '-p', metavar = 'S', type=str, required = True, help='Path/folder for the calculations' )
    parser.add_argument( '--jobname', '-j', metavar = 'S', type=str, required = True, help='Job name for this run' )
    parser.add_argument( '--neutral', '-n', metavar = 'B', type=bool, required = False, default = False, help='Impose charge of 0 on the system?' )
    parser.add_argument( "--zrange", "-zr", nargs="+", type=float, required = True, help="Starting, end point and step of z translation.")
    parser.add_argument( '--run_types', '-rt', nargs='+', help='a_w_electrode c_w_electrode aa_w_electrode_f aa_w_electrode_o cc_w_electrode_f cc_w_electrode_o ac_w_electrode_f ac_w_electrode_o aa_wo_electrode_f aa_wo_electrode_o cc_wo_electrode_f cc_wo_electrode_o ac_wo_electrode_f ac_wo_electrode_o', required = True )

    return parser.parse_args()

def clean_files(path,pattern):
    all_files = os.listdir(path)
    filtered = fnmatch.filter(all_files,pattern+"*")
    for element in filtered:
        os.remove(os.path.join(path,element))

def return_value(filename,pattern):
    with open(filename, "r") as fin:
        # memory-map the file, size 0 means whole file
        m = mmap.mmap(fin.fileno(), 0, prot=mmap.PROT_READ)
                                    # prot argument is *nix only
        i = m.rfind(pattern)
        m.seek(i)             # seek to the location
        line = m.readline()   # read to the end of the line
    return line.split()[-1]

def z_translate( molec, z_vect ):
    my_list = []
    for shift in z_vect:
        dummy = molec.copy()
        dummy.translate([0.0,0.0,float(shift)])
        my_list.append(dummy)
    return my_list

if __name__ == '__main__':

    args = parse_commandline_arguments()
    root_dir = os.getcwd()
    calc = CP2K()
    calc.mpi_n_processes = args.ncores
    run_types = args.run_types
    zrange = args.zrange
    neutral = args.neutral
    if neutral:
        calc.project_name = args.jobname + '-neutral'
    else:
        calc.project_name = args.jobname
    calc.working_directory = "./"
    CP2K_INPUT = calc.CP2K_INPUT
    GLOBAL = CP2K_INPUT.GLOBAL
    FORCE_EVAL = CP2K_INPUT.FORCE_EVAL_add()
    SUBSYS = FORCE_EVAL.SUBSYS
    MOTION = CP2K_INPUT.MOTION
    GLOBAL.Extended_fft_lengths = True

    FORCE_EVAL.Method = 'QS'

    FORCE_EVAL.DFT.Basis_set_file_name = '/gpfs1l/gpfshome/mds/staff/mburbano/bmim-pf6/BASIS_MOLOPT'
    FORCE_EVAL.DFT.Potential_file_name = '/gpfs1l/gpfshome/mds/staff/mburbano/bmim-pf6/GTH_POTENTIALS'

    FORCE_EVAL.DFT.MGRID.Cutoff = 280
    FORCE_EVAL.DFT.QS.Method = 'GPW'
    FORCE_EVAL.DFT.QS.Map_consistent = 'TRUE'
    FORCE_EVAL.DFT.QS.Extrapolation = 'ASPC'
    FORCE_EVAL.DFT.QS.Extrapolation_order = 3 
    FORCE_EVAL.DFT.QS.Eps_default = 1.0E-10
    #FORCE_EVAL.DFT.QS.Eps_pgf_orb = 1.0E-07
    FORCE_EVAL.DFT.SCF.Scf_guess = 'ATOMIC'
    FORCE_EVAL.DFT.SCF.Eps_scf = 1.0E-05
    FORCE_EVAL.DFT.SCF.Max_scf = 80
    FORCE_EVAL.DFT.SCF.OUTER_SCF.Max_scf = 2 
    FORCE_EVAL.DFT.SCF.OUTER_SCF.Eps_scf = 1.0E-05
    FORCE_EVAL.DFT.POISSON.Periodic = 'XY'
    FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'
    #FORCE_EVAL.DFT.SCF.OUTER_SCF.Eps_scf = 1.0E-05
    #FORCE_EVAL.DFT.SCF.OUTER_SCF.Max_scf = 12
    FORCE_EVAL.DFT.XC.XC_FUNCTIONAL.Section_parameters = "PBE"
    FORCE_EVAL.DFT.Uks = True

    KIND = SUBSYS.KIND_add("H")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-BLYP-q1'

    KIND = SUBSYS.KIND_add("C")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-BLYP-q4'

    KIND = SUBSYS.KIND_add("N")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-BLYP-q5'

    KIND = SUBSYS.KIND_add("P")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-BLYP-q5'

    KIND = SUBSYS.KIND_add("F")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-BLYP-q7'

    GLOBAL.Run_type = 'ENERGY'



    bmim_pf6_opt = ase.io.read('/gpfshome/mds/staff/mburbano/bmim-pf6/after-optimization/single-ions/opt_coords.xyz')

    bmim_pf6_opt.cell = [22.104, 21.27, 100.]
    bmim_pf6_opt.pbc = [True,True,False]
    pf6 = bmim_pf6_opt[0:7]; bmim = bmim_pf6_opt[7:32]; electrodes = bmim_pf6_opt[32:]
    pf6_COM = pf6.get_center_of_mass()
    bmim_COM = bmim.get_center_of_mass()
    shift_vector = pf6_COM - bmim_COM
    z_translate_range = np.arange(zrange[0],zrange[1]+zrange[2],zrange[2])

    def run_calc( my_dir, box ):
        import subprocess
        eng_string = "ENERGY| Total FORCE_EVAL"
        if os.path.exists(my_dir):
            shutil.rmtree(my_dir)
        os.makedirs(my_dir)
        os.chdir(my_dir)
        calc.create_cell(SUBSYS,box)
        calc.create_coord(SUBSYS,box)
        try:
            FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
            FORCE_EVAL.DFT.SCF.OT.Preconditioner = 'FULL_ALL'
            FORCE_EVAL.DFT.SCF.Scf_guess = 'ATOMIC'
            FORCE_EVAL.DFT.SCF.OT.Linesearch = '3PNT'
            calc.run()
            result = return_value(calc.output_path,eng_string)
        except subprocess.CalledProcessError, e:
            FORCE_EVAL.DFT.SCF.Scf_guess = 'RESTART'
            FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
            FORCE_EVAL.DFT.SCF.OT.Preconditioner = 'FULL_ALL'
            FORCE_EVAL.DFT.SCF.OT.Linesearch = '3PNT'
            calc.run()
            result = return_value(calc.output_path,eng_string)
        except subprocess.CalledProcessError, e:
            FORCE_EVAL.DFT.SCF.Scf_guess = 'ATOMIC'
            FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
            FORCE_EVAL.DFT.SCF.OT.Preconditioner = 'FULL_ALL'
            FORCE_EVAL.DFT.SCF.OT.Linesearch = 'GOLD'
            calc.run()
            result = return_value(calc.output_path,eng_string)
        except subprocess.CalledProcessError, e:
            print( 'CG did not converge' )
            result = np.nan 
        os.chdir(root_dir)
        return result 

    pf6_list_L = z_translate( pf6, - z_translate_range )
    bmim_list_L = z_translate( bmim, - z_translate_range )
    ###################################################################
    pf6_R = pf6.copy()
    pf6_R.rotate(v = 'y', a= - np.pi, center='COM' )#; pf6_R.wrap()
    pf6_R.translate([0.0,0.0,2 * abs(pf6_COM[2] - 50.0) ])
    pf6_R_o = pf6_R.copy()
    pf6_list_R = z_translate( pf6_R, z_translate_range )
    ###################################################################
    pf6_R_o.translate([0.0,- shift_vector[1], 0.0])
    pf6_list_R_o = z_translate( pf6_R_o, z_translate_range )
    ###################################################################
    bmim_R = bmim.copy()
    bmim_R.rotate(v = 'y', a= - np.pi, center='COM' )#; bmim_R.wrap()
    bmim_R.translate([0.0,0.0,2 * abs(bmim_COM[2] - 50.0)])
    bmim_R_o = bmim_R.copy()
    bmim_list_R = z_translate( bmim_R, z_translate_range )
    ###################################################################
    bmim_R_o.translate([0.0, shift_vector[1], 0.0])
    bmim_list_R_o = z_translate( bmim_R_o, z_translate_range )

    my_confs = z_translate_range
    columns = run_types 
    energies = pd.DataFrame(columns=columns, index=my_confs)
    for column in columns:
        for index, conf in enumerate(z_translate_range):
            # with electrodes
            if column == 'a_w_electrode':
                box = pf6_list_L[index] + electrodes
                if not neutral: 
                    FORCE_EVAL.DFT.Charge = -1 
            elif column == 'c_w_electrode':
                box = bmim_list_L[index] + electrodes
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +1 
            elif column == 'aa_w_electrode_f':
                box = pf6 + pf6_list_R[index] +  electrodes
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
            elif column == 'aa_w_electrode_o':
                box = pf6 + pf6_list_R_o[index] +  electrodes
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
            elif column == 'cc_w_electrode_f':
                box = bmim + bmim_list_R[index] +  electrodes
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +2
            elif column == 'cc_w_electrode_o':
                box = bmim + bmim_list_R_o[index] +  electrodes
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +2
            elif column == 'ac_w_electrode_f':
                box = pf6 + bmim_list_R[index] +  electrodes
                if not neutral:
                    FORCE_EVAL.DFT.Charge = 0
            elif column == 'ac_w_electrode_o':
                box = pf6 + bmim_list_R_o[index] +  electrodes
                if not neutral:
                    FORCE_EVAL.DFT.Charge = 0
            elif column == 'aa_wo_electrode_f':
                box = pf6 + pf6_list_R[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
            elif column == 'aa_wo_electrode_o':
                box = pf6 + pf6_list_R_o[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
            elif column == 'cc_wo_electrode_f':
                box = bmim + bmim_list_R[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +2
            elif column == 'cc_wo_electrode_o':
                box = bmim + bmim_list_R_o[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +2
            elif column == 'ac_wo_electrode_f':
                box = pf6 + bmim_list_R[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = 0
            elif column == 'ac_wo_electrode_o':
                box = pf6 + bmim_list_R_o[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = 0

            dir_name = calc.project_name + '-' + column + str(conf)
            energies[column][conf] = run_calc(dir_name, box)  
            energies.to_csv(root_dir + '/' + calc.project_name+'.csv')
