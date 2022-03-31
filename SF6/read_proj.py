from DFTtools.DFT.output_parser import SCFout
from torch_geometric.data import Data
from e3nn.o3 import Irreps
import torch 
import numpy as np
import re

class projectedBand:
    def __init__(self, filename: str):
        self.atomicOrbits=None    
        self.allK=None            
        self.E_forEachK=None      
        self.allCoeff=None        
        self.numOrbits=None       
        with open(filename, 'r') as f:
            self._readoutput(f)

    def read_states(self, f):
        atoms = {}
        states= []
        aline = f.readline()
        while "state" not in aline:
            aline = f.readline()
        counts = 0
        while "state" in aline:
            line=aline.lstrip().rstrip()
            tmp=re.split('\(|\)',line)

            atomic_index=int(tmp[0].split()[-1]) - 1 # start from zero
            
            parts = re.findall(r"[lm]=\s*\d", tmp[3])
            l = int(parts[0].split("=")[1])
            m = int(parts[1].split("=")[1])
            atoms.setdefault(atomic_index, {}).setdefault(l, {})[m] = counts # also start from zero, different from file
            states.append(f"{atomic_index}-{l}-{m}")
            counts += 1
            aline = f.readline()
        self.state_index = atoms
        self.states = states

    def read_coefficients(self, f, nbnd, nkstot):
        kpoints = []
        # each k point is a dictionary with a list of energies and corresponding coefficients
        aline = f.readline()
        k_done = 0
        while k_done < nkstot:
            if " k =" in aline:
                xyz = np.array(aline.split("=")[1].split(), dtype = float)
                nbnd_done = 0
                energies = []
                coefficients = []
                while nbnd_done < nbnd:
                    energies.append( float(f.readline().split()[4]) )
                    nbnd_done += 1
                    coefficient = []
                    aline = f.readline()
                    while "Coe =" in aline:
                        string = aline.split("=")[1].split("*")[0]
                        real = float(string.split("+")[0])
                        comp = float(string.split("+")[1].replace("i", ""))
                        coefficient.append(real + comp * 1j)
                        aline = f.readline()
                    coefficients.append(np.array(coefficient, dtype = complex))
                kpoints.append({"k": xyz, "energy": energies, "coefficients": np.vstack(coefficients)})
                k_done += 1
            # 
            aline = f.readline()
        
        self.kpoint_coefficients = kpoints

    def _readoutput(self, f):
        aline = f.readline()
        while "Problem Sizes" not in aline:
            aline = f.readline()
        aline = f.readline()
        nbnd = int(f.readline().split("=")[1])
        nkstot = int(f.readline().split("=")[1])

        while "Atomic states used for projection" not in aline:
            aline = f.readline()
        self.read_states(f)

        aline = f.readline()
        while "Printing coefficients" not in aline:
            aline = f.readline()
        self.read_coefficients(f, nbnd, nkstot)

    def readOutput_old(self,filename):
        # this reads the Project.out file to get the coefficients
        output=open(filename,'r')
        data=output.readlines()
        output.close()
        self.atomicOrbits=[] #this is the "state #   1: atom   1 (Sn ), wfc  1 (l=2 m= 1)" orbits
        self.numOrbits=0
        self.allK=[]
        self.E_forEachK=[]
        self.allCoeff=[]
        i=0
        readMark="normal"
        
        #readMark is one of the ['normal','state','projwfc']
        while i < len(data):
            if readMark=="normal" :
                if data[i].lstrip().rstrip()=="Atomic states used for projection":
                    i+=3
                    readMark="state"
                else:
                    i+=1
            elif readMark=="state":
                #which means this line being read is of the kind: 
                #this is the "state #   1: atom   1 (Sn ), wfc  1 (l=2 m= 1)" orbits
                line=data[i].lstrip().rstrip()
                tmp=re.split('\(|\)',line)
                tmp_atomicNumber=tmp[0].split()[-1]
                tmp_stateName=tmp[1]+tmp_atomicNumber+' '+tmp[3]
                self.atomicOrbits.append(tmp_stateName)
                self.numOrbits+=1
                i+=1
                if data[i].lstrip().rstrip()=='':
                    #finished reading all the states, we proced to read projected wave functions.
                    readMark="projwfc"
                    i+=1
                    
            elif readMark=="projwfc":
                if data[i][0:2]==' k':
                    #which means this line is of form " k =   0.0000000000  0.0000000000  0.0000000000"
                    tmp=data[i].rstrip().split()
                    tmpkxyz=[float(tmp[2]),float(tmp[3]),float(tmp[4])]
                    self.allK.append(tmpkxyz)
                    tmp_Efork=[]
                    tmp_Coeff_fork=[]
                elif data[i][0]=='=':
                    #which means this line is like "==== e(   1) =   -13.33128 eV ==== "
                    tmp=data[i].split()
                    tmpE=float(tmp[4])
                    tmp_Efork.append(tmpE)
                    tmp_long=""
                else:
                    """
                    which means this line is one of the line:
                         psi = 0.209*[#   4]+0.209*[#  13]+0.209*[#  22]+0.209*[#  31]+0.037*[#   2]+
                               +0.037*[#  11]+0.037*[#  20]+0.037*[#  29]+0.003*[#   1]+0.003*[#  10]+
                               +0.003*[#  19]+0.003*[#  28]+0.001*[#  37]+0.001*[#  41]+0.001*[#  45]+
                               +0.001*[#  49]+
                        |psi|^2 = 1.000
                    """
                    if data[i].rstrip().lstrip()!='' and data[i].lstrip()[0]=='|':
                        #process the combined line
                        
                        #following reads the states that are included in the linear equations
                        tmplist=re.findall('(\[# +[0-9]+\])',tmp_long) 
                        #this finds all the [#  54] "#"+some" "+some"number" that is bracketed in [ ]
                        if tmplist==None:
                            #which means coefficient is all 0
                            tmp_coefficient=[0]*self.numOrbits
                        else:
                            tmp_states=[]
                            for word in tmplist:
                                tmp_states.append(int(re.findall("([0-9]+)",word)[0]))
                            #following part reads the coefficient for each
                            tmplist=re.findall('([0-9]+\.[0-9]+)',tmp_long)
                            tmp_coefficient=[0]*self.numOrbits
                            for sss in range(len(tmp_states)):
                                tmp_coefficient[tmp_states[sss]-1]=float(tmplist[sss])
                        #here we have the coefficient list of the linear combination
                        tmp_Coeff_fork.append(tmp_coefficient)
                    elif data[i].rstrip().lstrip()!='':
                        tmp_long+=data[i].rstrip().lstrip()
                    else:
                        #which means an empty line, which is fine except when this field of data end
                        self.E_forEachK.append(tmp_Efork)
                        self.allCoeff.append(tmp_Coeff_fork)
                        if data[i+1].rstrip().lstrip()=='Lowdin Charges:':
                            readMark='normal'
                i+=1

        self.allK=np.array(self.allK)
        self.allCoeff=np.array(self.allCoeff)
        self.E_forEachK=np.array(self.E_forEachK)

def get_datas(scfout, projout, feature_max:int = 3):
    # it returns a list of data that contain the orbital information
    scf = SCFout(scfout)
    result = projectedBand(projout)
    # we organize them into a torch_geometry object
    positions = scf.positions
    cell = scf.lattice
    list_of_data = []
    cartesian_positions = np.einsum("ij, zi -> zj", cell, positions)
    cartesian_positions = torch.from_numpy(cartesian_positions)
    irrep = ""
    for i in range(feature_max):
        irrep += f"1x{i}"
        irrep += {1:"e", -1:"o"}[(-1)**i]
        if i < feature_max-1:
            irrep += " + "

    len_irreps = Irreps(irrep).dim
    slice_irrep = Irreps(irrep).slices()

    gamma = result.kpoint_coefficients[0]
    for e, coeff in zip(gamma["energy"], gamma["coefficients"]):
        attributes = np.zeros((len(scf.positions), len_irreps))
        for iatom, indexs in result.state_index.items():
            for l, ms in indexs.items():
                comp = [ val for val in ms.values()]
                attributes[iatom][slice_irrep[l]] = coeff[comp].real

        attributes = torch.from_numpy(attributes)
        target = torch.tensor(e)
        list_of_data.append(Data(x = attributes, y = target, pos = cartesian_positions))

    return list_of_data

if __name__ == "__main__":
    datas = get_datas("scf/scf.out", "scf/proj_coeff.out", feature_max=2)
    print(datas[1].x)
    print(datas[2].x)
    print(datas[3].x)