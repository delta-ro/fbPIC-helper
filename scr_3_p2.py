# -------
# Imports
# -------
import numpy as np
from scipy.constants import c, e, m_e

# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser, FewCycleLaser, add_laser_pulse
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
     set_periodic_checkpoint, restart_from_checkpoint

# ----------
# Parameters
# ----------

# Whether to use the GPU
use_cuda = True

# Order of the stencil for z derivatives in the Maxwell solver.
# Use -1 for infinite order, i.e. for exact dispersion relation in
# all direction (adviced for single-GPU/single-CPU simulation).
# Use a positive number (and multiple of 2) for a finite-order stencil
# (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
# to more overhead in MPI communications, but also to a more accurate
# dispersion relation for electromagnetic waves. (Typically,
# `n_order = 32` is a good trade-off.)
# See https://arxiv.org/abs/1611.05712 for more information.
n_order = 32

# The laser
a0 = 2.2             # Laser amplitude
w0 = 2.e-6          # Laser waist
ctau = c*5.e-15        # Laser duration (FWHM)
z0 = -3*ctau        # Laser centroid
zf = 37.5*1.e-6         # Laser focus
lambda0 = 800e-9    # Laser wavelength

L_R = np.pi * w0**2 / lambda0

R0 = w0 * (1+ (zf-z0)**2/L_R**2 )**0.5
rmax = 2.5 * R0

dz = lambda0/64
dr = 16 * dz
zmax = 0 
zmin = zmax - 15*ctau

Nz =  int(np.ceil((zmax- zmin)/dz / 32)) * 32
Nr =  int(np.ceil(rmax/dr / 8)) * 8
Nm = 3                    # Number of modes used

# The simulation timestep
dt = (zmax-zmin)/Nz/c     # Timestep (seconds)

# The particles
p_zmin = 0.e-6            # Position of the beginning of the plasma (meters)
p_zmax = 200.e-6          # Position of the end of the plasma (meters)
p_rmax = rmax + 5.e-6     # Maximal radial position of the plasma (meters)
n_e = 6.e19*1.e6          # Density (electrons.meters^-3)
p_nz = 1                  # Number of particles per cell along z
p_nr = 3                  # Number of particles per cell along r
p_nt = Nm*2               # Number of particles per cell along theta (put 6 if 3 modes)

# The moving window
v_window = c              # Speed of the window
   
ramp_start = 0.e-6
ramp_p1 = 20.e-6
ramp_p2 = ramp_p1+10.e-6
ramp_p3 = ramp_p2+10.e-6

#dens_func = lambda z, r: np.interp(z, [ramp_start, ramp_start+ramp_length], [0, 1], left=0, right=1)

def dens_func( z, r ):
    """Returns relative density at position z and r"""
    # Allocate relative density
    n = np.ones_like(z)
    n = np.where( z<ramp_p3, 4/3-1/3*(z-ramp_p2)/(ramp_p3-ramp_p2), 1)
    n = np.where( z<ramp_p2, 4/3, n )
    n = np.where( z<ramp_p1, 4/3*(z-ramp_start)/(ramp_p1-ramp_start), n )
    n = np.where( z<ramp_start, 0., n )
    return(n)

# The interaction length of the simulation (meters)
L_interact = 80.e-6 # increase to simulate longer distance!
# Interaction time (seconds) (to calculate number of PIC iterations)
T_ch = 52.e-6 / v_window
T_interact = ( L_interact + (zmax-zmin) ) / v_window
# (i.e. the time it takes for the moving window to slide across the plasma)

N_step_pre = int(T_interact/dt)
n_diag = 40                # Number of diagnostics to be saved
diag_period_pre = int(np.floor(N_step_pre/n_diag))
N_step_pre = diag_period_pre * n_diag + 1

# ---------------------------
use_restart = False        # Whether to restart from a previous checkpoint
save_checkpoints = True    # Whether to write checkpoint files
track_electrons = True     # Whether to track and write particle ids
# ---------------------------

# The diagnostics and the checkpoints/restarts
if use_restart is False:
    diag_period = diag_period_pre 
    N_step = N_step_pre
else:
    diag_period = 3       # Period of the diagnostics in number of timesteps
    
# ---------------------------
# Carrying out the simulation
# ---------------------------

# NB: The code below is only executed when running the script,
# (`python lwfa_script.py`), but not when importing it (`import lwfa_script`).
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        n_order=n_order, use_cuda=use_cuda,
        boundaries={'z':'open', 'r':'open'})
        # 'r': 'open' can also be used, but is more computationally expensive

    # Create the plasma electrons
    elec = sim.add_new_species( q=-e, m=m_e, n=n_e,
        dens_func=dens_func, p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
        p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    laser_profile = FewCycleLaser(a0=a0, waist=w0, tau_fwhm=ctau/c, z0=z0, zf=zf, lambda0=lambda0, propagation_direction=1)
    add_laser_pulse( sim, laser_profile )
    
    if use_restart is False:
        elec.track( sim.comm )
        sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm,  fieldtypes=['rho', 'E', ] ),
                      ParticleDiagnostic( diag_period, {"electrons" : elec},
                        select={"uz" : [1., None ]}, comm=sim.comm) ]
        N_step = int(T_interact/sim.dt)
        N_ch = int(T_ch/sim.dt)
        set_periodic_checkpoint( sim, N_ch, checkpoint_dir='./ch')
        sim.set_moving_window( v=v_window )
    else:
        # Load the fields and particles from the latest checkpoint file
        restart_from_checkpoint( sim, checkpoint_dir='./ch')
        sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm, write_dir = './_spectrum_comparison/diags_shock_per1' ),
                      ParticleDiagnostic( diag_period, {"electrons" : elec},
                        select={"uz" : [1., None ]}, comm=sim.comm, write_dir = './_spectrum_comparison/diags_shock_per1' ) ]
        N_step = int((zmax-zmin) / v_window /sim.dt/3)      #set number of steps to track the laser at the reflection position    
        sim.set_moving_window( v=0 )
    
        
    ### Run the simulation
    sim.step( N_step )
    print('')