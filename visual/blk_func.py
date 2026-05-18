import numpy as np
import sys
sys.path.append( "../base" )
from unit import units

"""
bf prefix - function acting on individual meshblock hydro data
f_blk prefix - function acting on a list of all outputs from bf functions
"""

def bf_hyd_cmzmass( b, bd, **kwargs ):
    Xb, Yb, Zb = np.meshgrid( bd['x_c'][0], 
                              bd['x_c'][1], 
                              bd['x_c'][2], indexing='ij' )
    R = np.sqrt( Xb**2 + Yb**2 )
    R_out   = kwargs.get( "R_out", 1e32 )
    Z_out   = kwargs.get( "R_out", 1e32 )
    d_floor = kwargs.get( "d_floor", 1e-20 )
    masses = ( bd[ 'rho' ] * np.prod( bd[ 'dx' ], axis = 0 ) )[ ( R < R_out ) & 
                                                                ( np.abs( Zb ) < Z_out ) &
                                                                ( bd[ 'rho' ] >= d_floor ) ]
    R_out   = kwargs.get( "R_out", 1e32 )
    Z_out   = kwargs.get( "R_out", 1e32 )
    d_floor = kwargs.get( "d_floor", 1e-20 )
    if len( masses ) == 0:
        return 0
    else:
        return np.sum( masses ) * units.rho0 * units.l0**3 / units.modot

def f_blk_cmzmass( bfd, **kwargs ):
    return np.sum( np.array( bfd ) )

def bf_hyd_cmzvel( b, bd, **kwargs ):
    Xb, Yb, Zb = np.meshgrid( bd['x_c'][0], 
                              bd['x_c'][1], 
                              bd['x_c'][2], indexing='ij' )
    R = np.sqrt( Xb**2 + Yb**2 )
    R_out   = kwargs.get( "R_out", 1e32 )
    Z_out   = kwargs.get( "R_out", 1e32 )
    phi = np.arctan2( Yb, Xb )
    mom_phi = bd[ 'mom' ][ 0 ] * -np.sin( phi ) +\
              bd[ 'mom' ][ 1 ] *  np.cos( phi )
    mom_phi = ( mom_phi * np.prod( bd['dx'], axis = 0 ) )[ ( R < R_out ) & ( np.abs( Zb ) < Z_out ) ]
    dens    = ( bd[ 'rho' ] * np.prod( bd['dx'],axis = 0 ) )[ ( R < R_out ) & ( np.abs( Zb ) < Z_out ) ]
    if len( mom_phi ) == 0:
        return [ 0, 0 ]
    else:
        return [ np.sum( mom_phi ), 
                 np.sum( dens ) ]

def f_blk_cmzvel( bfd, **kwargs ):
    tot = np.sum( np.array( bfd ), axis = 0 )
    return tot[ 0 ] / tot[ 1 ] * units.l0 / units.t0 / 1e5

def bf_hyd_massflux( b, bd, **kwargs ):
    zflux = kwargs.get( "zflux", 140 )
    R_in  = kwargs.get( "R_in", 500 )
    R = np.sqrt( bd[ 'x_c' ][ 0 ]**2 + bd[ 'x_c' ][ 1 ]**2 )
    dz = bd[ 'dx0' ][ 2 ]
    cell_z_pos = (  zflux > bd[ 'x' ][ 2 ] - dz / 2 ) & (  zflux <= bd[ 'x' ][ 2 ] + dz / 2 )
    cell_z_neg = ( -zflux > bd[ 'x' ][ 2 ] - dz / 2 ) & ( -zflux <= bd[ 'x' ][ 2 ] + dz / 2 )
    cell_R_in  = R < R_in
    cell_area  = bd[ 'dx0' ][ 0 ] * bd[ 'dx0' ][ 1 ]
    flux_in  = np.sum( bd[ 'mom' ][ 2 ][ cell_z_pos & cell_R_in ] ) * cell_area - np.sum( bd[ 'mom' ][ 2 ][ cell_z_neg & cell_R_in ] ) * cell_area
    flux_out = np.sum( bd[ 'mom' ][ 2 ][ cell_z_pos & ~cell_R_in ] ) * cell_area - np.sum( bd[ 'mom' ][ 2 ][ cell_z_neg & ~cell_R_in ] ) * cell_area
    return [ flux_in, flux_out ]

def f_blk_massflux( bfd, **kwargs ):
    tot = np.sum( np.array( bfd ), axis=0 )
    tot *= units.m0 / units.modot
    tot /= ( units.t0 / units.yr )
    return [ tot[ 0 ], tot[ 1 ] ]