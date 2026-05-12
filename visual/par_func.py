import numpy as np
import sys
sys.path.append( "../base" )
from unit import units

def f_par_SFR( d, **kwargs ):
    x,   y,  z = d.data[ 'particle_x' ].T;
    sfr        = d.data[ 'particle_sfr' ].T;
    R = np.sqrt( x**2 + y**2 )
    R_out = kwargs.get( "R_out", 1e32 )
    Z_out = kwargs.get( "R_out", 1e32 )
    return np.sum( sfr[ ( R < R_out ) & ( np.abs( z ) < Z_out ) ] ) / units.t0 * units.m0 / units.modot * units.yr 

def f_par_Mstar( d, **kwargs ):
    x,   y,  z = d.data[ 'particle_x' ].T;
    R = np.sqrt( x**2 + y**2 )
    mstar      = d.data[ 'particle_mstar'].T
    R_out = kwargs.get( "R_out", 1e32 )
    Z_out = kwargs.get( "R_out", 1e32 )
    return np.sum( mstar[ ( R < R_out ) & ( np.abs( z ) < Z_out ) ] ) * ( units.m0 / units.modot )

def f_par_Nsne( d, **kwargs ):
    x,   y,  z = d.data[ 'particle_x' ].T;
    R = np.sqrt( x**2 + y**2 )
    Nsne       = d.data[ 'particle_Nsne' ].T
    R_out = kwargs.get( "R_out", 1e32 )
    Z_out = kwargs.get( "R_out", 1e32 )
    return np.sum( Nsne[ ( R < R_out ) & ( np.abs( z ) < Z_out ) ] ) ;

def f_par_Mgas( d, **kwargs ):
    x,   y,  z = d.data[ 'particle_x' ].T;
    R = np.sqrt( x**2 + y**2 )
    m          = d.data[ 'particle_m' ].T;
    mstar      = d.data[ 'particle_mstar'].T
    R_out = kwargs.get( "R_out", 1e32 )
    Z_out = kwargs.get( "R_out", 1e32 )
    return np.sum( ( m - mstar )[ ( R < R_out ) & ( np.abs( z ) < Z_out ) ] ) * ( units.m0 / units.modot )
