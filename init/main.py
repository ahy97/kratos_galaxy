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

import numpy      as  np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams[ 'image.cmap' ] = 'turbo'
import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 30


def config_setup( filename ):
    """
    Load in the configuration file. All other class parameters will be derived
    from the configuration file. User keyword arguments will override the configuration file
    parameters
    
    :param filename: Path to configuration file
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read( filename )
    units.read_config( config )
    data_field.unit_system = units( )

if __name__ == "__main__":
    try:
        setup_file = sys.argv[1]
    except IndexError:
        print("No input file specified, defaulting to 'setup.par'")
        setup_file = "setup.par"
    config_setup( setup_file )
    
    prof = profile( config_flag='output_grid' )
    IC( prof )


    
