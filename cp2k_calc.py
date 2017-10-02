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
from collections import OrderedDict

def return_value(filename,pattern,val_posn = -1):
    value = np.nan
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
            return value
        line = m.readline()   # read to the end of the line
        value = float(line.split()[val_posn])
    return value 

def hirshfeld_charges( infile, uks = False ):
    if uks:
        header = r'\#Atom  Element  Kind  Ref Charge     Population       Spin moment  Net charge'
        startat = 7
    else:
        header = r'\#Atom  Element  Kind  Ref Charge     Population                    Net charge'
        startat = 5
    footer = r'Total Charge'
    f = open(infile,'r')
    data = f.read()
    x = re.findall(header+'(.*?)'+footer,data,re.DOTALL)[0].split()
    return np.array(x[startat::startat+1],dtype=np.float64)

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
        self.periodicity = periodicity
        self.basis = basis
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
        # Changing number of wfn copies that are written
        FORCE_EVAL.DFT.SCF.PRINT.RESTART.Backup_copies = 0

        FORCE_EVAL.DFT.Charge = charge

        if self.periodicity == 3:
            FORCE_EVAL.DFT.POISSON.Periodic = 'XYZ'
            FORCE_EVAL.DFT.POISSON.Poisson_solver = 'PERIODIC'
        elif self.periodicity == 2:
            FORCE_EVAL.DFT.POISSON.Periodic = 'XY'
            FORCE_EVAL.DFT.POISSON.Poisson_solver = 'ANALYTIC'
        elif self.periodicity == 0:
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

        GLOBAL.Run_type = 'ENERGY'

        if diagonalize:
            FORCE_EVAL.DFT.SCF.Max_scf = 600
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
            FORCE_EVAL.DFT.SCF.Max_scf = 20
            FORCE_EVAL.DFT.SCF.OUTER_SCF.Max_scf = 50
            FORCE_EVAL.DFT.SCF.OUTER_SCF.Eps_scf = eps_scf

        return

    def run_calc( self, my_dir, box ):
        # This function will run the calculation intitially using the 
        # default FULL_KINETIC and 2PNT settings. If these fail, other
        # linesearches are performed before changing the preconditioner.
        # ranOK is used to break out of the loops once the calculation has
        # finished without errors.
        self.my_dir = my_dir
        self.box = box
        eng_string = "ENERGY| Total FORCE_EVAL"
        if os.path.exists(self.my_dir):
            shutil.rmtree(self.my_dir)
        os.makedirs(self.my_dir)
        os.chdir(self.my_dir)


        for kind in list(dict.fromkeys(box.get_chemical_symbols())):
            KIND = self.SUBSYS.KIND_add(kind)
            KIND.Basis_set = self.basis 
            KIND.Potential = 'GTH-PBE'

        if self.periodicity == 3:
            self.box.set_pbc((True,True,True))
        elif self.periodicity == 2:
            self.box.set_pbc((True,True,False))
        elif self.periodicity == 0:
            self.box.set_pbc((False,False,False))

        self.box.write(self.my_dir+'.cif')
        self.box.write(self.my_dir+'.xyz')
        result = np.nan 
        ranOK = False

        self.calc.create_cell(self.SUBSYS,self.box)
        self.calc.create_coord(self.SUBSYS,self.box)

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

    def get_charges( self, how_many = 0 ):
        os.chdir(self.my_dir)
        try:
            charges = hirshfeld_charges(self.calc.output_path)[how_many:]
        except:
            charges = np.zeros(abs(how_many))
            print('Unable to get charges')
        os.chdir(self.root_dir)
        return charges 
