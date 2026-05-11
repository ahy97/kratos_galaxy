"""
Proxy file containing various imports helpful for kratos data visualization and post processing in
jupyter notebooks.
"""
import os
import re
import sys
import yt
import pyxsim
sys.path.append( os.getenv("KRATOS_VISUAL_DIR") )
sys.path.append(os.path.join(os.path.dirname(__file__), "../base"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../init"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../tables"))
from hydro_data_gal import *
from yt_kratos import *
from init import gen_profiles_bin
from   hydro_data import get_dd, get_last_dd, enroll_mesh_tree
import glob
import numpy      as  np
from utils import *
from cooling_conversion import *
from SB99_conversion import *
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 30

import matplotlib as mpl
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm, SymLogNorm
mpl.rc( 'font', family = 'serif' );
mpl.rc( 'text', usetex = False    );
mpl.pyplot.rcParams[ 'image.cmap' ] = 'turbo'
mpl.rcParams[ 'font.size'        ] = 12;
mpl.rcParams[ 'axes.labelsize'   ] = 14;
mpl.rcParams[ 'legend.fontsize'  ] = 12;
mpl.rcParams[ 'legend.edgecolor' ] = 'k';
mpl.rcParams[ 'figure.facecolor' ] = 'w';
from plot_gal import *
############################################################
# Constants

h      = units.h           #6.62607e-27;   # CGS Planctk constant
kb     = units.kb          #1.38065e-16;   # CGS Boltzmann constant
eV     = units.eV          #1.60218e-12;   # CGS eV
c      = units.c           #2.99792458e10; # CGS speed of light
q_e    = units.q_e         #4.80321e-10;   # CGS electron charge
me     = units.me          #9.1094e-28;    # CGS electron mass;
mp     = units.mp          #1.67262e-24;   # CGS proton mass;
AU     = units.AU          #1.49598e13     # Astronomical Unit in cm
G      = units.G           #6.6742831e-8   # CGS graviational constant
sig_sb = units.sig_sb      #5.6704e-5      # Stefan-Boltzmann
yr     = units.yr          #365. * 86400.; # Year in seconds
pc     = units.pc          #3.0857e18;     # Parsec in cm

modot  = units.modot       #1.9891e33;     # Solar mass
rodot  = units.rodot       #6.96e10;       # Solar radius
lodot  = units.lodot       #3.828e33;      # Solar luminosity
mearth = units.mearth      #5.9742e27      # Earth mass
rearth = units.rearth      #6.37814e8;     # Earth Radius

########################################
# Basic dimension recoverage
t0     = units.t0     
l0     = units.l0     
rho0   = units.rho0   
m0     = units.m0     
v0     = units.v0     
G_code = units.G_code
############################################################
