############################################################
# Headers

import os
import re
import sys
import cProfile
import configparser
import copy
from background import *
from FDM import *
from init import *
sys.path.insert( 0, "../base" )
from data_field import *
from grid import *
from profile_base import *
from source import *
from unit import *
from utils import *

import numpy      as  np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams[ 'image.cmap' ] = 'turbo'
import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 30


if __name__ == "__main__":
    try:
        setup_file = sys.argv[1]
    except IndexError:
        print("No input file specified, defaulting to 'setup.par'")
        setup_file = "setup.par"
    config_setup( setup_file )
    
    prof = profile( config_flag='output_grid' )
    IC( prof )


    
