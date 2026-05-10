import numpy as np
import re
import sys
sys.path.append( "../base" )
from unit import units
from scipy.interpolate import LinearNDInterpolator
from elements import elements
from configparser import ConfigParser

elements_atomic_numbers = elements.elements_atomic_numbers
elements_atomic_weights = elements.elements_atomic_weights
elements_solar_abundance = elements.elements_solar_abundance

def load_SB99( dirname ):
    solarabundances = np.column_stack( ( np.array( list( elements_atomic_numbers .values( ) ) ), 
                                         np.array( list( elements_atomic_weights .values( ) ) ), 
                                         np.array( list( elements_solar_abundance.values( ) ) ) ) )
    SAtrunc = solarabundances[[0,1,5,6,7,11,13,15,25]]
    Z_solar = np.sum(solarabundances[2:][:,1]*solarabundances[2:][:,2])/np.sum(solarabundances[:,1]*solarabundances[:,2])
    Z_solar_trunc = np.sum(SAtrunc[2:][:,1]*SAtrunc[2:][:,2])/np.sum(SAtrunc[:,1]*SAtrunc[:,2])

    datayield  = np.loadtxt( f"{dirname}/CGKZ3.yield" )
    datasnr    = np.loadtxt( f"{dirname}/CGKZ3.snr"   )
    timeaxis = np.ones( len( datasnr[:,0] ) ) * datasnr[:,0][0] +\
               np.arange( len( datasnr[:,0] ) ) * np.diff( datasnr[:,0] )[0]
    datapower  = np.loadtxt( f"{dirname}/CGKZ3.power"  )
    dataquanta = np.loadtxt( f"{dirname}/CGKZ3.quanta" )
    
    data = { "t_par_yr" : [False, timeaxis],                   # Time axis in years
             "sne_mass" : [True,10**(datayield[:,11] - datasnr[:,1])], # Mass per SNe
             "sne_rate" : [False,(datasnr[:,1])],                   # SNe rate
             "met_rate" : [False,np.sum(10**datayield[:,3:10],         # Metallicity
                           axis=1)/np.sum(10**datayield[:,1:10],axis=1)/Z_solar_trunc],
             "wind_mass_rate" : [False,(datayield[:,10])],               # Wind mass
             "wind_pow_rate" : [False,(datapower[:,1])],                # Wind energy
             "wind_mom_rate" : [False,(datapower[:,7])],                # Wind momentum
             "photon_rate" : [False,(dataquanta[:,1])],               # Ionizing photon/s          
            }
    return data

def write_table_SB99( SB99_data, fname ):
    cfg = ConfigParser(  );
    cfg.optionxform = str;
    cfg[ 'fdbk_rate' ] = \
        { 't_par_yr'      : ' '.join( [ '%0.4f' % s for s in SB99_data[ "t_par_yr" ][ 1 ]   ] )   
        }
    
    for key, dat in SB99_data.items( ):
        dat_cp = dat[ 1 ].copy( )
        if dat[ 0 ]:
            dat_cp[ SB99_data[ "sne_rate" ][ 1 ] < -29 ] = 0
        cfg[ "fdbk_rate" ][ key ] = ' '.join( [ '%0.4f' % s for s in dat_cp   ] )
    with open( fname, 'w' ) as f:
        cfg.write( f );

#sbdat = load_SB99( "../../Starburst99/CGKZ3/Output" )
#write_table_SB99( sbdat, "SB99_conversion.dat" )