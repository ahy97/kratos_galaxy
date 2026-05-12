import cooling_conversion as cool
import SB99_conversion as fdbk
import configparser
import sys
import numpy as np
sys.path.insert( 0, "../base" )
from utils import *
import pickle
import glob, os

if __name__ == "__main__":
    try:
        setup_file = sys.argv[1]
    except IndexError:
        print("No input file specified, defaulting to 'table_setup.par'")
        setup_file = "table_setup.par"
    config = config_setup( setup_file )
    cloudy_dir  = config.get( "cooling", "dirname" )
    cloudy_file = config.get( "cooling", "fname"   )
    cloudy_data_path = config.get( "cooling", "data_path", fallback="cloudy_data.pkl" )
    cloudy_int_path  = config.get( "cooling", "int_path" , fallback="int_data.pkl" )
    regen       = bool( int( config.get( "cooling", "regen", fallback=0 ) ) )

    if regen:
        print( "Purging existing obj files" )
        files = glob.glob( "obj/*" )
        for i in files:
            print( f"Purging {i}" )
            os.remove( i )

    try:
        with open( f"obj/{cloudy_data_path}", 'rb' ) as f:
            dat = pickle.load( f )
        print( f"Loaded existing CLOUDY data from directory: {cloudy_data_path}" )
    except FileNotFoundError:
        print( f"Existing CLOUDY data not found. Loading CLOUDY outputs from directory {cloudy_dir}/{cloudy_file} and generating data" )
        dat = cool.load_cloudy( cloudy_dir, cloudy_file, save_path=cloudy_data_path )

    try:
        with open( f"obj/{cloudy_int_path}", 'rb' ) as f:
            int_dat = pickle.load( f )
        print( f"Loaded existing interpolators from directory: {cloudy_int_path}" )
    except FileNotFoundError:
        print( f"Existing interpolators not found. Generating interpolators from CLOUDY data" )
        int_dat = cool.make_interpolators( dat, save_path=cloudy_int_path )

    #met_range_input = bool( int( config.get( "cooling", "met_range_input", fallback=0 ) ) )
    #if met_range_input:
    print( f"Existing metallicity grid: { np.unique( dat[ 'params_coord' ][ :,2 ] ) }")
    #    print( f"Input desired array indices" )
    #    met_idx = input( )
    #    print( met_idx )
    interp_range = []
    for prefix, param in zip( [ "eg", "rho", "met" ], dat[ 'params_coord' ].T ):
        ranges = config.get('cooling',f'{prefix}_interp_range').split( )
        new_range = []
        for j in ranges:
            if "min" in j:
                val_mod = j.replace( "min", "" )
                new_range.append( np.min( param ) + float( val_mod ) )
            elif "max" in j:
                val_mod = j.replace( "max", "" )
                new_range.append( np.max( param ) + float( val_mod ) )
            else:
                new_range.append( float( j ) )
        new_range[ -1 ] = int( new_range[ -1 ] )
        interp_range.append( np.linspace( *tuple( new_range ) ) )
    cloudy_out = config.get( "cooling", "outname" )
    print( f"Writing cooling table to directory: {cloudy_out}" )
    cool.write_table_cloudy( dat, int_dat, interp_range[ ::-1 ], cloudy_out, 
                        reduce_mu_interp =  bool( config.get( "cooling", "reduce_mu_interp" ) ), 
                        rho_rep          = float( config.get( "cooling", "rho_rep" ) ), 
                        met_rep          = float( config.get( "cooling", "met_rep" ) ) )
    
    
    fdbk_dir = config.get( "fdbk", "dirname" )
    print( f"Loading Starburst99 data from directory: {fdbk_dir}" )
    sbdat = fdbk.load_SB99( fdbk_dir )
    fdbk_out = config.get( "fdbk", "outname" )
    print( f"Writing feedback table to directory: {fdbk_out}" )
    fdbk.write_table_SB99( sbdat, fdbk_out )
    



