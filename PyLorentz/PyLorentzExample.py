import sys 
sys.path.append("../PyTIE/")
sys.path.append("../SimLTEM/")
from TIE_helper import show_im, show_2D
from sim_helper import *
from TIE_reconstruct import TIE, SITIE
from comp_phase import mansPhi, linsupPhi
import numpy as np

%matplotlib widget

file = "/zota/Lorentz/AlecBender/for_Arthur/skyrm_size_variations/SkyrmTest_Region_Aex(1.100e-11)_Dind(1.000e-03)_MaxDind(1.393e-03)_MinDind(8.107e-04)_mAngle(15.76)_tvalue(2.750e-08)_Msat(1.450e+05)_Ku1(1.500e+06)_B(4.000e-01)_Time(193.74).ovf"
B0 = 1e4 # gauss
sample_V0 = 20 # V
sample_xip0 = 50 # nm
mem_thk = 50 # nm
mem_xip0 = 1000 # nm

from colorwheel import color_im
mag_x, mag_y, mag_z, del_px, zscale = load_ovf(file, 'norm', B0, v=0)
# The input ovf might be many layers thick, so we sum along the z-direction to 
# make a 2D image to display. 
show_im(color_im(np.sum(mag_x, axis=0), np.sum(mag_y,axis=0), hsvwheel=True), 
        title="Raw magnetization data from file", cbar=False, scale=del_px

dim = 128
del_px = 500/dim # nm/pixel
zscale = 10 # nm/pixel in the z-direction
b0 = 1e4 # Gauss 
V0 = 10 # V
Bloch_x, Bloch_y, Bloch_z = Bloch(dim, chirality = 'cw', pad = True, ir=0)
show_2D(Bloch_x, Bloch_y, a=20, l=0.15, w=0.75, title='in-plane magnetization', color=True)
show_3D(Bloch_x, Bloch_y, Bloch_z, show_all = True, l=2, a = 50)

phi0 = 2.07e7 # Gauss*nm^2 
pre_B_L = 2*np.pi*b0*zscale*del_px/phi0
thickness_nm = zscale * 1 # 1 layer only, so thickness is just 10nm
pre_E = Microscope().sigma*V0*thickness_nm

thk_map = make_thickness_map(mag_x, mag_y, mag_z)
show_im(thk_map.sum(axis=0), "Thickness map", cbar=False)

pscope = Microscope(E=200e3, Cs = 200.0e3, theta_c = 0.01e-3, def_spr = 80.0, verbose=True)
defval = 100_000 # nm
theta_x = 0 # degrees
theta_y = 0 # degrees
add_random = 0 # unitless scaling factor
flip=True # Bool

dim = 128
del_px = 500/dim # nm/pixel
zscale = 10 # nm/pixel in the z-direction
b0 = 1e4 # Gauss 
V0 = 10 # V
Bloch_x, Bloch_y, Bloch_z = Bloch(dim, chirality = 'cw', pad = True, ir=0)
show_2D(Bloch_x, Bloch_y, a=20, l=0.15, w=0.75, title='in-plane magnetization', color=True)
show_3D(Bloch_x, Bloch_y, Bloch_z, show_all = True, l=2, a = 50)

phi0 = 2.07e7 # Gauss*nm^2 
pre_B_L = 2*np.pi*b0*zscale*del_px/phi0
thickness_nm = zscale * 1 # 1 layer only, so thickness is just 10nm
pre_E = Microscope().sigma*V0*thickness_nm

ephi_L, mphi_L = linsupPhi(mx=Bloch_x.reshape(1,dim,dim),
                           my=Bloch_y.reshape(1,dim,dim),
                           mz=Bloch_z.reshape(1,dim,dim), 
                           pre_B=pre_B_L,
                           theta_x=0,
                           pre_E=pre_E)
show_im(mphi_L, "Magnetic phase shift from linear superposition method (rad)", scale=del_px, cbar_title="radians")
# show_im(ephi_L, "Electrostatic phase shift from linear superposition method (rad)")

# Apply mansuripur algorithm with some standard materials parameters. 
ephi_m, mphi_m = std_mansPhi(Bloch_x, Bloch_y, Bloch_z, 
                             isl_thk=zscale*1, # thickness of the magnetic structure in nm, 1 layer thick here
                             zscale=zscale, 
                             del_px=del_px,
                             b0=b0, isl_V0=V0)
show_im(mphi_m, title="magnetic phase shift from Mansuripur algorithm", cbar_title="radians")
# show_im(ephi_m, title="electrostatic phase shift from Mansuripur algorit

# using the linear superposition phase 
ALTEM = Microscope(E=200e3,Cs = 200.0e3, theta_c = 0.01e-3, def_spr = 80.0)
defval = 100_000 # nm 
amorphous_noise = 0.2 # unitless scaling parameter 
Tphi, im_un, im_in, im_ov = sim_images(mphi=mphi_L, ephi=ephi_L, 
                                       pscope=ALTEM,
                                       del_px=del_px, 
                                       def_val=defval,
                                       add_random=amorphous_noise)
show_sims(Tphi, im_un, im_in, im_ov)