#sys.path.insert(0,'../pycp2k')
#sys.path.insert(0,'../python-utils')
import os
from ase import Atoms
import numpy as np
from ase.io import read, write
from pycp2k import CP2K
from file_utils import *
import shutil
import subprocess

class Cp2k_calc:

    def __init__( self, jobname, ncores = 1, mgrid = 280, eps_scf = 1E-05, charge = 0, periodicity = 2, 
            vdW = True, basis = 'DZVP-MOLOPT-GTH', diagonalize = False, spin_polarized = False, 
            added_MOs = 20, debug = False ):

        ############################################################################
        self.root_dir = os.getcwd()
        self.calc = CP2K()
        self.calc.project_name = jobname
        self.calc.mpi_n_processes = ncores
        self.calc.working_directory = "./"
        self.debug = debug
        ############################################################################

        CP2K_INPUT = self.calc.CP2K_INPUT
        GLOBAL = CP2K_INPUT.GLOBAL
        FORCE_EVAL = CP2K_INPUT.FORCE_EVAL_add()
        self.SUBSYS = FORCE_EVAL.SUBSYS
        MOTION = CP2K_INPUT.MOTION
        GLOBAL.Extended_fft_lengths = True

        FORCE_EVAL.Method = 'QS'

        FORCE_EVAL.DFT.Basis_set_file_name = 'BASIS_MOLOPT'
        FORCE_EVAL.DFT.Potential_file_name = 'GTH_POTENTIALS'

        FORCE_EVAL.DFT.MGRID.Cutoff = mgrid
        FORCE_EVAL.DFT.QS.Method = 'GPW'
        FORCE_EVAL.DFT.SCF.Scf_guess = 'ATOMIC'
        FORCE_EVAL.DFT.SCF.Eps_scf = eps_scf 

        FORCE_EVAL.DFT.Charge = charge

        if periodicity == 3:
            FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
            FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
        elif periodicity == 2:
            FORCE_EVAL.DFT.POISSON.Periodic = 'XY'
            FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'
        elif periodicity == 0:
            FORCE_EVAL.DFT.POISSON.Periodic = 'NONE'
            FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'

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

        KIND = self.SUBSYS.KIND_add("H")
        KIND.Basis_set = basis 
        KIND.Potential = 'GTH-PBE'

        KIND = self.SUBSYS.KIND_add("C")
        KIND.Basis_set = basis 
        KIND.Potential = 'GTH-PBE'

        KIND = self.SUBSYS.KIND_add("N")
        KIND.Basis_set = basis 
        KIND.Potential = 'GTH-PBE'

        KIND = self.SUBSYS.KIND_add("P")
        KIND.Basis_set = basis 
        KIND.Potential = 'GTH-PBE'

        KIND = self.SUBSYS.KIND_add("F")
        KIND.Basis_set = basis 
        KIND.Potential = 'GTH-PBE'

        GLOBAL.Run_type = 'ENERGY'

        if diagonalize:
            FORCE_EVAL.DFT.SCF.Max_scf = 300
            FORCE_EVAL.DFT.SCF.DIAGONALIZATION.Algorithm = 'STANDARD'
            FORCE_EVAL.DFT.SCF.Added_mos = [added_MOs, added_MOs]
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

        return

    def run_calc( self, my_dir, box ):
        # This function will run the calculation intitially using the 
        # default FULL_KINETIC and 2PNT settings. If these fail, other
        # linesearches are performed before changing the preconditioner.
        # ranOK is used to break out of the loops once the calculation has
        # finished without errors.
        eng_string = "ENERGY| Total FORCE_EVAL"
        if os.path.exists(my_dir):
            shutil.rmtree(my_dir)
        os.makedirs(my_dir)
        os.chdir(my_dir)
        self.calc.create_cell(self.SUBSYS,box)
        self.calc.create_coord(self.SUBSYS,box)
        box.write(my_dir+'.cif')
        box.write(my_dir+'.xyz')
        result = np.nan 
        ranOK = False
        try:
            if self.debug:
                self.calc.write_input_file()
            else:
                self.calc.run()
                ranOK = True
        except subprocess.CalledProcessError:
            ranOK = False 
        #######################################################
        if ranOK:
            result = return_value(self.calc.output_path,eng_string)
        else:
            result = np.nan
        os.chdir(self.root_dir)
        return result 

