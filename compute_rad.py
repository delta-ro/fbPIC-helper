import numpy as np
import h5py
from synchrad.calc import SynchRad

E_0 = 1.24e-06

if __name__ == "__main__":
    calc_input = {
        "grid": [ (1e3/E_0, 20e3/E_0),        #range of ENERGIES
                  (0, 1/20),                  #range of angles THETA
                  (0.0, 2 * np.pi),           #range of angles PHI
                  (256, 64, 32)],               #how many points to chose in each range (energy, theta, phi)
        "dtype": "double",
        "native": False,
        "ctx": 'mpi',
        #"ctx": [0, 0], #set your context to avoid being asked (how many GPUs to use)
    }

    calc = SynchRad(calc_input)
    calc.calculate_spectrum(file_tracks="tracks.h5",
                            it_range=[0,1750],                   #specify the iteration range (CAREFULL) 
                            nSnaps = 25,                         #add snaps along the calculation
                            file_spectrum="spectrum.h5",)