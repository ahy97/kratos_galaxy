import numpy as np
import sys
sys.path.append( "../base" )
from unit import units

"""
bf prefix - function acting on individual meshblock hydro data
f_blk prefix - function acting on a list of all outputs from bf functions
"""

def bf_hyd_cmzmass( b, bd, **kwargs ):
    """
    Per-block gas mass within a cylindrical region ``R_cyl < R_out, |z| < Z_out``.

    Returns
    -------
    float
        Gas mass in solar masses (Msun).
    """
    Xb, Yb, Zb = bd[ "x" ]
    R = np.sqrt( Xb**2 + Yb**2 )
    R_out   = kwargs.get( "R_out", 1e32 )
    Z_out   = kwargs.get( "Z_out", 1e32 )
    d_floor = kwargs.get( "d_floor", 1e-20 )
    masses = ( bd[ 'rho' ] * np.prod( bd[ 'dx' ], axis = 0 ) )[ ( R < R_out ) & 
                                                                ( np.abs( Zb ) < Z_out ) &
                                                                ( bd[ 'rho' ] >= d_floor ) ]

    if len( masses ) == 0:
        return 0
    else:
        return np.sum( masses ) * units.m0 / units.modot

def f_blk_cmzmass( bfd, **kwargs ):
    """
    Sum per-block gas masses.

    Returns
    -------
    float
        Total gas mass in Msun.
    """
    return np.sum( np.array( bfd ) )

def bf_hyd_cmzvel( b, bd, **kwargs ):
    """
    Per-block mass-weighted angular momentum and total mass within a cylinder.

    Returns
    -------
    list
        ``[sum(mom_phi * dV), sum(dens * dV)]`` in code units.
    """
    Xb, Yb, Zb = bd[ "x" ]
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
    """
    Compute mass-weighted mean azimuthal velocity from block sums.

    Returns
    -------
    float
        Mean azimuthal velocity in km/s.
    """
    tot = np.sum( np.array( bfd ), axis = 0 )
    return tot[ 0 ] / tot[ 1 ] * units.l0 / units.t0 / 1e5

def bf_hyd_massflux( b, bd, **kwargs ):
    """
    Per-block mass flux through two horizontal planes at ``z = ±Z_flux``,
    split into inner (R < R_in) and outer (R > R_in) contributions.

    Returns
    -------
    list
        ``[flux_inner, flux_outer]`` in code units.
    """
    zflux = kwargs.get( "Z_flux", 140 )
    R_in  = kwargs.get( "R_in", 500 )
    R = np.sqrt( bd[ 'x' ][ 0 ]**2 + bd[ 'x' ][ 1 ]**2 )
    dz = bd[ 'dx0' ][ 2 ]
    cell_z_pos = (  zflux > bd[ 'x' ][ 2 ] - dz / 2 ) & (  zflux <= bd[ 'x' ][ 2 ] + dz / 2 )
    cell_z_neg = ( -zflux > bd[ 'x' ][ 2 ] - dz / 2 ) & ( -zflux <= bd[ 'x' ][ 2 ] + dz / 2 )
    cell_R_in  = R < R_in
    cell_area  = bd[ 'dx0' ][ 0 ] * bd[ 'dx0' ][ 1 ]
    flux_in  = np.sum( bd[ 'mom' ][ 2 ][ cell_z_pos & cell_R_in ] ) * cell_area - np.sum( bd[ 'mom' ][ 2 ][ cell_z_neg & cell_R_in ] ) * cell_area
    flux_out = np.sum( bd[ 'mom' ][ 2 ][ cell_z_pos & ~cell_R_in ] ) * cell_area - np.sum( bd[ 'mom' ][ 2 ][ cell_z_neg & ~cell_R_in ] ) * cell_area
    return [ flux_in, flux_out ]

def f_blk_massflux( bfd, **kwargs ):
    """
    Aggregate per-block mass fluxes and convert to Msun/yr.

    Returns
    -------
    list
        ``[flux_in, flux_out]`` in Msun/yr.
    """
    tot = np.sum( np.array( bfd ), axis=0 )
    tot *= units.m0 / units.modot
    tot /= ( units.t0 / units.yr )
    return [ tot[ 0 ], tot[ 1 ] ]

def bf_hyd_cylflux( b, bd, **kwargs ):
    """
    Per-block mass flux through the lateral surface of a cylinder of radius
    *R_flux* and half-height *Z_cut*.  Uses exact circle--face intersection
    geometry on the Cartesian grid.

    Returns
    -------
    float  –  net mass flux in **code units** [m0 / t0].
             Positive = net outward radial flow.

    (If *separate_flow* is True, returns ``[flux_in, flux_out]`` instead.)
    """
    R      = kwargs.get( "R_flux", 100 )
    Z_cut  = kwargs.get( "Z_cut", 200 )
    sep    = kwargs.get( "separate_flow", False )

    x1d = bd[ "x_c" ][ 0 ]
    y1d = bd[ "x_c" ][ 1 ]
    z1d = bd[ "x_c" ][ 2 ]

    dx = bd[ "dx0" ][ 0 ]
    dy = bd[ "dx0" ][ 1 ]
    dz = bd[ "dx0" ][ 2 ]

    X, Y, Zgrid = np.meshgrid( x1d, y1d, z1d, indexing="ij" )

    xl = X - dx / 2
    xr = X + dx / 2
    yb = Y - dy / 2
    yt = Y + dy / 2

    r_cell = np.sqrt( X * X + Y * Y )
    R2 = R * R

    # ---- quick reject via corner radial bounds ----
    rmin = np.minimum(
        np.minimum( np.sqrt( xl**2 + yb**2 ), np.sqrt( xl**2 + yt**2 ) ),
        np.minimum( np.sqrt( xr**2 + yb**2 ), np.sqrt( xr**2 + yt**2 ) ),
    )
    rmax = np.maximum(
        np.maximum( np.sqrt( xl**2 + yb**2 ), np.sqrt( xl**2 + yt**2 ) ),
        np.maximum( np.sqrt( xr**2 + yb**2 ), np.sqrt( xr**2 + yt**2 ) ),
    )

    cyl_mask = ( R >= rmin ) & ( R <= rmax ) & ( r_cell > 0 )
    z_mask   = np.abs( Zgrid ) <= Z_cut
    mask     = cyl_mask & z_mask

    if not mask.any():
        return [ 0.0, 0.0 ] if sep else 0.0

    # ---- signs for root selection ----
    sign_x = np.where( X >= 0, 1, -1 )
    sign_y = np.where( Y >= 0, 1, -1 )

    # ---- intersection y / x coordinates (NaN where invalid) ----
    yi_left   = np.where( ( R2 >= xl**2 ) & ( sign_y != 0 ),
                          sign_y * np.sqrt( np.maximum( R2 - xl**2, 0 ) ), np.nan )
    yi_right  = np.where( ( R2 >= xr**2 ) & ( sign_y != 0 ),
                          sign_y * np.sqrt( np.maximum( R2 - xr**2, 0 ) ), np.nan )
    xi_bottom = np.where( ( R2 >= yb**2 ) & ( sign_x != 0 ),
                          sign_x * np.sqrt( np.maximum( R2 - yb**2, 0 ) ), np.nan )
    xi_top    = np.where( ( R2 >= yt**2 ) & ( sign_x != 0 ),
                          sign_x * np.sqrt( np.maximum( R2 - yt**2, 0 ) ), np.nan )

    theta_left   = np.where( ( yb <= yi_left   ) & ( yi_left   <= yt ),
                             np.arctan2( yi_left,   xl ), np.nan )
    theta_right  = np.where( ( yb <= yi_right  ) & ( yi_right  <= yt ),
                             np.arctan2( yi_right,  xr ), np.nan )
    theta_bottom = np.where( ( xl <= xi_bottom ) & ( xi_bottom <= xr ),
                             np.arctan2( yb, xi_bottom ), np.nan )
    theta_top    = np.where( ( xl <= xi_top    ) & ( xi_top    <= xr ),
                             np.arctan2( yt, xi_top    ), np.nan )

    theta_stack = np.stack( [ theta_left, theta_right,
                              theta_bottom, theta_top ], axis=0 )
    n_valid     = np.sum( ~np.isnan( theta_stack ), axis=0 )

    theta_min = np.nanmin( theta_stack, axis=0, initial=np.inf,
                           where=~np.all( np.isnan( theta_stack ), axis=0 ) )
    theta_max = np.nanmax( theta_stack, axis=0, initial=-np.inf,
                           where=~np.all( np.isnan( theta_stack ), axis=0 ) )

    dtheta = theta_max - theta_min
    dtheta = np.where( dtheta > np.pi, 2 * np.pi - dtheta, dtheta )

    valid  = mask & ( n_valid >= 2 ) & ~np.isnan( dtheta )
    dtheta = np.where( valid, dtheta, 0.0 )

    if not valid.any():
        return [ 0.0, 0.0 ] if sep else 0.0

    cos_phi = np.where( r_cell > 0, X / r_cell, 0.0 )
    sin_phi = np.where( r_cell > 0, Y / r_cell, 0.0 )
    mom_R   = bd[ "mom" ][ 0 ] * cos_phi + bd[ "mom" ][ 1 ] * sin_phi

    area       = R * dtheta * dz
    flux_cells = mom_R * area

    if sep:
        flux_out = np.sum( flux_cells[ flux_cells > 0 ] )
        flux_in  = np.sum( flux_cells[ flux_cells < 0 ] )
        return [ flux_in, flux_out ]
    return np.sum( flux_cells )


def f_blk_cylflux( bfd, **kwargs ):
    """
    Aggregate per-block cylinder-flux contributions and convert to Msun / yr.

    Parameters
    ----------
    bfd : list
        Outputs from ``bf_hyd_cylflux``, each a ``[flux_in, flux_out]`` pair
        or a scalar net flux.

    Returns
    -------
    list or float
        ``[tot_in, tot_out]`` or net total, in Msun / yr.
    """
    sep = kwargs.get( "separate_flow", False )
    arr = np.array( bfd )
    tot = np.sum( arr, axis=0 ) if arr.ndim == 2 else np.sum( arr )
    tot *= units.m0 / units.modot
    tot /= ( units.t0 / units.yr )
    if sep:
        return [ tot[ 0 ], tot[ 1 ] ]
    return tot