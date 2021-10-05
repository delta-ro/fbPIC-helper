from openpmd_viewer import ParticleTracker
from openpmd_viewer import OpenPMDTimeSeries

from synchrad.utils import tracksFromOPMD


if __name__ == "__main__":
    dNt = 1
    ts = OpenPMDTimeSeries("home/sims/02_shock_ins/test_03/shock_scan_diag_1/__analysis/shock_01/diags/hdf5", check_all_files=False)
    ref_iteration = 4718

    pt = ParticleTracker(ts, iteration=ref_iteration,
                         select={'uz':[2,None]}, 
                         preserve_particle_index=True)

    tracksFromOPMD(
        ts, pt, ref_iteration=ref_iteration,
        dNt=dNt,                   #add the step for iterations (equivalent to diag_frquency in the main code)
        fname='tracks_dNt_'+str(dNt)+'.h5'
    )