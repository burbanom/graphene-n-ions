from __future__ import print_function
import numpy as np
from scipy import linalg
from ase.io import xyz
import pandas as pd
import ase
from pycp2k import CP2K
import os, re
import fnmatch
import shutil

def parse_commandline_arguments():
    import argparse
    parser = argparse.ArgumentParser( description = 'cp2k calculator' )
    parser.add_argument( '--ncores', '-nc', metavar = 'N', type=int, required = True, help='set the number of cores to use for this calculation' )
    parser.add_argument( '--mgrid', '-mg', metavar = 'N', type=int, required = False, default = 280, help='MGRID cutoff' )
    #parser.add_argument( '--path', '-p', metavar = 'S', type=str, required = True, help='Path/folder for the calculations' )
    parser.add_argument( '--jobname', '-j', metavar = 'S', type=str, required = True, help='Job name for this run' )
    parser.add_argument( '--neutral', '-n', metavar = 'B', type=bool, required = False, default = False, help='Impose charge of 0 on the system?' )
    parser.add_argument( '--threeD', '-d3', metavar = 'B', type=bool, required = False, default = False, help='3D periodic system' )
    parser.add_argument( '--bigbox', '-bb', metavar = 'B', type=bool, required = False, default = False, help='Perform calcs on large box.' )
    parser.add_argument( '--diag', '-d', metavar = 'B', type=bool, required = False, default = False, help='Perform diagonalization.' )
    parser.add_argument( '--vdW', '-vdW', metavar = 'B', type=bool, required = False, default = False, help='Perform PBE-vdW DF.' )
    parser.add_argument( "--zrange", "-zr", nargs="+", type=float, required = False, help="Starting, end point and step of z translation.")
    parser.add_argument( "--yrange", "-yr", nargs="+", type=float, required = False, help="Starting, end point and step of y translation.")
    parser.add_argument( "--lshift", "-ls", metavar="F", type=float, required = True, help="Initial shift for lhs ion from centre of the box")
    parser.add_argument( "--rshift", "-rs", metavar="F", type=float, required = True, help="Initial shift for rhs ion from centre of the box")
    parser.add_argument( '--run_types', '-rt', nargs='+', help='a_w_electrode c_w_electrode aa_w_electrode_f aa_w_electrode_o aa_w_electrode_f_c aa_w_electrode_o_c cc_w_electrode_f cc_w_electrode_o ac_w_electrode_f ac_w_electrode_o aa_wo_electrode_f aa_wo_electrode_o cc_wo_electrode_f cc_wo_electrode_o ac_wo_electrode_f ac_wo_electrode_o aa_w_electrode_y aa_wo_electrode_y', required = True )

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

def z_translate( molec, z_vect ):
    my_list = []
    for shift in z_vect:
        dummy = molec.copy()
        dummy.translate([0.0,0.0,float(shift)])
        my_list.append(dummy)
    return my_list

def y_translate( molec, y_vect ):
    my_list = []
    for shift in y_vect:
        dummy = molec.copy()
        dummy.translate([0.0,float(shift),0.0])
        my_list.append(dummy)
    return my_list

if __name__ == '__main__':

    args = parse_commandline_arguments()
    root_dir = os.getcwd()
    calc = CP2K()
    calc.mpi_n_processes = args.ncores
    run_types = args.run_types
    threeD = args.threeD
    bigbox = args.bigbox
    neutral = args.neutral
    lshift = args.lshift
    rshift = args.rshift

    if args.zrange is not None:
        zrange = args.zrange
    if args.yrange is not None:
        yrange = args.yrange
    if threeD:
        calc.project_name = args.jobname + '-3D'
    if bigbox:
        calc.project_name = args.jobname + '-bigbox'
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

    FORCE_EVAL.DFT.Basis_set_file_name = 'BASIS_MOLOPT'
    FORCE_EVAL.DFT.Potential_file_name = 'GTH_POTENTIALS'

    FORCE_EVAL.DFT.MGRID.Cutoff = args.mgrid
    FORCE_EVAL.DFT.QS.Method = 'GPW'
    #FORCE_EVAL.DFT.QS.Map_consistent = 'TRUE'
    #FORCE_EVAL.DFT.QS.Extrapolation = 'ASPC'
    #FORCE_EVAL.DFT.QS.Extrapolation_order = 3 
    #FORCE_EVAL.DFT.QS.Eps_default = 1.0E-10
    #FORCE_EVAL.DFT.QS.Eps_pgf_orb = 1.0E-07
    FORCE_EVAL.DFT.SCF.Scf_guess = 'ATOMIC'
    FORCE_EVAL.DFT.SCF.Eps_scf = 1.0E-05
    FORCE_EVAL.DFT.SCF.Max_scf = 40
    FORCE_EVAL.DFT.SCF.OUTER_SCF.Max_scf = 8
    FORCE_EVAL.DFT.SCF.OUTER_SCF.Eps_scf = 1.0E-05
    FORCE_EVAL.DFT.POISSON.Periodic = 'XY'
    FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'
    FORCE_EVAL.DFT.XC.XC_FUNCTIONAL.Section_parameters = "PBE"
    FORCE_EVAL.DFT.Uks = True

    KIND = SUBSYS.KIND_add("H")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("C")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("N")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("P")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-PBE'

    KIND = SUBSYS.KIND_add("F")
    KIND.Basis_set = 'DZVP-MOLOPT-GTH'
    KIND.Potential = 'GTH-PBE'

    GLOBAL.Run_type = 'ENERGY'



    if bigbox: 
        bmim_pf6_opt = ase.io.read('/gpfshome/mds/staff/mburbano/bmim-pf6/after-optimization/single-ions/opt_coords_bigbox.xyz')
        bmim_pf6_opt.cell = [44.20800018, 42.54000092, 100.]
    else:
        bmim_pf6_opt = ase.io.read('/gpfshome/mds/staff/mburbano/bmim-pf6/after-optimization/single-ions/opt_coords.xyz')
        bmim_pf6_opt.cell = [22.104, 21.27, 100.]

    bmim_pf6_opt.pbc = [True,True,False]
    pf6 = bmim_pf6_opt[0:7]; bmim = bmim_pf6_opt[7:32]; electrode = bmim_pf6_opt[32:]
    pf6_COM = pf6.get_center_of_mass()
    bmim_COM = bmim.get_center_of_mass()
    electrode_COM = electrode.get_center_of_mass()
    xy_shift = electrode_COM - (bmim+pf6).get_center_of_mass()
    ##### Testing the effect of putting the ions closer to the wall ########
    pf6_closer = pf6.copy(); bmim_closer = bmim.copy()
    pf6_closer.translate([xy_shift[0],xy_shift[1],50.0 - pf6_COM[2]]); pf6_closer.translate([0.0,0.0,-3.5])
    bmim_closer.translate([xy_shift[0],xy_shift[1],50.0 - bmim_COM[2]]); bmim_closer.translate([0.0,0.0,-3.5])
    ########################################################################

    shift_vector = pf6_COM - bmim_COM

    try:
        z_translate_range = np.arange(zrange[0],zrange[1]+zrange[2],zrange[2])
        energies = pd.DataFrame(columns=run_types, index=z_translate_range)
        my_range = z_translate_range
    except NameError:
        pass
    try:
        y_translate_range = np.arange(yrange[0],yrange[1]+yrange[2],yrange[2])
        energies = pd.DataFrame(columns=run_types, index=y_translate_range)
        my_range = y_translate_range
    except NameError:
        print( 'No ranges were attributed' )

    def run_calc( my_dir, box ):
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
        #for preconditioner in ['FULL_KINETIC','FULL_ALL','FULL_SINGLE','FULL_SINGLE_INVERSE','FULL_S_INVERSE']:
        #    if ranOK: break
        #    for linesearch in ['2PNT','3PNT', 'GOLD']:
        #        FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
        #        FORCE_EVAL.DFT.SCF.OT.Preconditioner = preconditioner
        #        FORCE_EVAL.DFT.SCF.OT.Linesearch = linesearch
        #        try:
        #            calc.run()
        #            ranOK = True
        #            break
        #        #except subprocess.CalledProcessError, e:
        #        except subprocess.CalledProcessError:
        #            ranOK = False 
        #            #FORCE_EVAL.DFT.SCF.Scf_guess = 'RESTART'
        #            continue
            if ranOK: break
            for linesearch in ['2PNT','3PNT', 'GOLD']:
                FORCE_EVAL.DFT.SCF.OT.Minimizer = 'CG'
                FORCE_EVAL.DFT.SCF.OT.Preconditioner = 'FULL_KINETIC'
                FORCE_EVAL.DFT.SCF.OT.Linesearch = '2PNT' 
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

    pf6_R = pf6.copy()
    bmim_R = bmim.copy()
    try:
        pf6_list_L = z_translate( pf6, - z_translate_range )
        bmim_list_L = z_translate( bmim, - z_translate_range )
        ###################################################################
        pf6_R.rotate(v = 'y', a= - np.pi, center='COM' )#; pf6_R.wrap()
        pf6_R.translate([0.0,0.0,2 * abs(pf6_COM[2] - 50.0) ])
        pf6_R_o = pf6_R.copy()
        pf6_list_R = z_translate( pf6_R, z_translate_range )
        ###################################################################
        pf6_R_o.translate([0.0,- shift_vector[1], 0.0])
        pf6_list_R_o = z_translate( pf6_R_o, z_translate_range )
        ###################################################################
        pf6_R_c = pf6_closer.copy()
        pf6_R_c.rotate(v = 'y', a= - np.pi, center='COM' )#; pf6_R.wrap()
        pf6_R_c.translate([0.0,0.0,2 * 3.5 ])
        pf6_R_o_c = pf6_R_c.copy()
        pf6_list_R_c = z_translate( pf6_R_c, z_translate_range )
        ###################################################################
        pf6_R_o_c.translate([0.0,- shift_vector[1], 0.0])
        pf6_list_R_o_c = z_translate( pf6_R_o_c, z_translate_range )
        ###################################################################
        bmim_R.rotate(v = 'y', a= - np.pi, center='COM' )#; bmim_R.wrap()
        bmim_R.translate([0.0,0.0,2 * abs(bmim_COM[2] - 50.0)])
        bmim_R_o = bmim_R.copy()
        bmim_list_R = z_translate( bmim_R, z_translate_range )
        ###################################################################
        bmim_R_o.translate([0.0, shift_vector[1], 0.0])
        bmim_list_R_o = z_translate( bmim_R_o, z_translate_range )
        ###################################################################
    except NameError:
        pass
    try:
        pf6_R_y = pf6.copy()
        pf6_R_y.rotate(v = 'y', a= - np.pi, center='COM' )#; pf6_R.wrap()
        pf6_R_y.translate([0.0,0.0,2 * abs(pf6_COM[2] - 50.0) ])
        pf6_list_R_y = y_translate( pf6_R_y, y_translate_range )
    except NameError:
        pass
    ###################################################################

    for column in run_types:
        for index, conf in enumerate(my_range):
            # with electrode
            if column == 'a_w_electrode':
                box = pf6_list_L[index] + electrode
                if not neutral: 
                    FORCE_EVAL.DFT.Charge = -1 
            elif column == 'c_w_electrode':
                box = bmim_list_L[index] + electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +1 
            elif column == 'aa_w_electrode_f':
                box = pf6 + pf6_list_R[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
            elif column == 'aa_w_electrode_o':
                box = pf6 + pf6_list_R_o[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
            elif column == 'aa_w_electrode_f_c':
                box = pf6_closer + pf6_list_R_c[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
                if threeD:
                    FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
                    FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
                    #box.cell = [bigbox*22.104,bigbox * 21.27, 100.]
                    box.pbc = [True,True,True]
            elif column == 'aa_w_electrode_o_c':
                box = pf6_closer + pf6_list_R_o_c[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
                if threeD:
                    FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
                    FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
                    #box.cell = [bigbox*22.104,bigbox * 21.27, 100.]
                    box.pbc = [True,True,True]
            elif column == 'aa_wo_electrode_f_c':
                box = pf6_closer + pf6_list_R_c[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
                if threeD:
                    FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
                    FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
                    #box.cell = [bigbox*22.104,bigbox * 21.27, 100.]
                    box.pbc = [True,True,True]
            elif column == 'aa_wo_electrode_o_c':
                box = pf6_closer + pf6_list_R_o_c[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
                if threeD:
                    FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
                    FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
                    #box.cell = [bigbox*22.104,bigbox * 21.27, 100.]
                    box.pbc = [True,True,True]
            elif column == 'cc_w_electrode_f':
                box = bmim + bmim_list_R[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +2
            elif column == 'cc_w_electrode_o':
                box = bmim + bmim_list_R_o[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = +2
            elif column == 'ac_w_electrode_f':
                box = pf6 + bmim_list_R[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = 0
            elif column == 'ac_w_electrode_o':
                box = pf6 + bmim_list_R_o[index] +  electrode
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
            elif column == 'aa_w_electrode_y':
                box = pf6 + pf6_list_R_y[index] +  electrode
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2
            elif column == 'aa_wo_electrode_y':
                box = pf6 + pf6_list_R_y[index]
                if not neutral:
                    FORCE_EVAL.DFT.Charge = -2

            dir_name = calc.project_name + '-' + column + str(conf)
            energies[column][conf] = run_calc(dir_name, box)  
            energies.to_csv(root_dir + '/' + calc.project_name+'.csv')
