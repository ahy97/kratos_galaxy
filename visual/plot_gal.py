import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from   matplotlib.colors import LogNorm, SymLogNorm
mpl.rc( 'font', family = 'serif' );
mpl.rc( 'text', usetex = False    );
mpl.pyplot.rcParams[ 'image.cmap' ] = 'turbo'
mpl.rcParams[ 'font.size'        ] = 12;
mpl.rcParams[ 'axes.labelsize'   ] = 14;
mpl.rcParams[ 'legend.fontsize'  ] = 12;
mpl.rcParams[ 'legend.edgecolor' ] = 'k';
mpl.rcParams[ 'figure.facecolor' ] = 'w';

import sys
import os
sys.path.append( os.getenv("KRATOS_VISUAL_DIR") )
sys.path.append( "../base" )
from unit import units
from hydro_data_gal import galaxy_data, enroll_T
from hydro_data import get_dd, get_last_dd, enroll_mesh_tree
import numpy as np
from numpy import all
from slice_plot import slice, recognize_vec
from blk_func import *
from par_func import *

############################################################
# Handy functions
##############################
def figgen( figsize = None, count = None ):
    if count is None:
        count = figgen.count % 1000;
    figgen.count += 1;
    if figsize is None:
        figsize = ( 5, 4.5 );
    return figure\
           ( count, figsize = figsize, facecolor = 'w' );
#
figgen.count = 0;

def plot_field( fig, d, f, ** kwargs ):
    args = { 'field' : f, 'xlabel' : r'$x/l_0$', \
             'ylabel' : r'$y/l_0$'};
    for k, v in kwargs.items(  ):
        args[ k ] = v;
    ax = slice( d, args, fig );
    return ax;
#

def f_plot( d, f, fig=None, ** args ):
    if not 'loc' in args:
        args[ 'loc' ] = 0;
    if fig is None:
        fig = figgen( figsize = ( 4.5, 4 ) );
    return plot_field( fig,\
                       d, f, ** args );
#
def fstring( i ):
    return ( 5 - len( str( i ) ) ) * "0" + str( i )

def f_plot_grid( d, **kwargs ):#flds, names, zlims, axes, xlims, ylims ):
    defaults = { 'rho'  : { "name" : r'Density ($m_p \; \mathrm{cm}^{-3}$)', "zlim" : ( 1e-4, 1e3 ) },
                 'T_ent': { "name" :  'Temperature (K)'                    , "zlim" : ( 1e1 , 1e8 ) },
                 'met'  : { "name" : r'Metallicity ($Z/Z_\odot$)'          , "zlim" : ( 1e-1, 1e2 ) } }
    flds  = kwargs.get( 'flds',  [ 'rho', 'T_ent', 'met', 'rho' ] )
    if isinstance( flds, str ):
        flds = [ flds ]

    axes  = kwargs.get( 'axes', [ 2, 1 ] )
    if isinstance( axes, int ):
        axes = [ axes ]

    names = kwargs.get( 'names', [ defaults[ i ][ 'name' ] if i in defaults.keys( ) else i for i in flds ] )

    zlims = kwargs.get( 'zlims', [ defaults[ i ][ 'zlim' ] if i in defaults.keys( ) else ( 1e-3, 1e3 ) for i in flds ] )
    z0s   = kwargs.get( 'z0s', [ 1 for i in flds ] );
    xlims = kwargs.get( 'xlims', [ (-1500,1500) for i in axes ] )
    ylims = kwargs.get( 'ylims', [ (-1500,1500) if i == 2 else (-3000,3000) for i in axes ] )
    log   = kwargs.get( 'log', [ True for i in flds ]  )
    locs  = kwargs.get( 'locs', [ [ 1e-9 for j in flds ] for i in axes ] )
    if not isinstance( locs[ 0 ], list ):
        locs = [ locs ]
    part_plot_i = kwargs.get( 'part_plot_i', -1 )
    if isinstance( part_plot_i, int ):
        part_plot_i = [ part_plot_i ]

    if not all( len( axes ) == len( lim ) for lim in [ xlims, ylims ] ):
        raise ValueError( "axes, xlims and ylims should have the same length" )
    
    if not all( len( flds ) == len( names ) == len( zlims ) ):
        raise ValueError( "flds, names and zlims should have the same length" )

    figsize = kwargs.get( 'figsize', (12,13.5) )
    t0     = d.args( "unit",    "time" );
    l0     = d.args( "unit",  "length" );
    rho0   = d.args( "unit", "density" ); 
    T_conv   = ( l0 / t0 )**2 * 1.26 * units.mp / units.kb;
    if "T" in flds:
        d.enroll_field( 'T', lambda bd : \
            bd[ 'pre' ] / bd[ 'rho' ] * T_conv );
    if "T_ent" in flds:
        d.enroll_field( 'T_ent', lambda bd : \
            bd[ 'pre_ent' ] / bd[ 'rho' ] * T_conv );
    fig = plt.figure( figsize = figsize )
    fig.subplots_adjust( hspace = 0 )
    axs = [ [ None for _ in flds ] for _ in axes ]
    for i, ( axis, xlim, ylim ) in enumerate( zip( axes, xlims, ylims ) ):
        for k, ( fld, zlim, z0 ) in enumerate( zip( flds, zlims, z0s ) ):
            pos = i * len( flds ) + k + 1
            ax = fig.add_subplot( len( axes ), len( flds ), pos )

            ax, _  = f_plot( d,  fld, fig, z0 = z0,\
                        log = log[ k ], zlim = zlim, loc = locs[ i ][ k ],\
                        axis = axis, integrate = False, ax = ax )
            axs[i][k] = ax
            ax.set_xlabel( 'pc' )
            ax.set_ylabel( 'pc' )
            if pos <= len( flds ):
                ax.tick_params( labelbottom = False )
                ax.set_xlabel( '' )
                ax.set_title( names[ k ] )
            if pos != 1 and pos != 1 + len( flds ):
                ax.tick_params( labelleft = False ) 
                ax.set_ylabel( '' )

            ax.set_xlim( xlim )
            ax.set_ylim( ylim )
    
    axs = np.array( axs )

    #Particles
    d.load_particles(  );
    x_par = d.data[ 'particle_x' ].T;
    v_par = d.data[ 'particle_v' ].T;
    for i, j in enumerate( axes ):
        xi, yi = [ k for k in ( 0, 1, 2 ) if k != j ]
        x_par_plot = x_par[ xi ]
        y_par_plot = x_par[ yi ]
        vx_par_plot = v_par[ xi ]
        vy_par_plot = v_par[ yi ]
        for pi in part_plot_i:
            axs[ i ][ pi ].quiver( x_par_plot, y_par_plot, vx_par_plot, vy_par_plot )
    return fig

def time_evo( dirname, outputnames, **kwargs ): #max_out, data_funcs, data_names, output_data=None ):
    """
    Generates time evolution data of various fields for kratos outputs

    :param dirname ( str )                     : directory where outputs are found
    :param outputnames ( str or list of str )  : output names sans the numerical suffixes or file extension
    :param \**kwargs : See below
    
    :Keyword Arguments:
        **max_out    ( int )                  : last output number 
        **data_funcs ( list of func )         : list of functions that can parse through either kratos binary outputs or the outputs of the corresponding blk functions
        **data_names ( list of str )          : list of data field names
        **output_data ( dict )                : dictionary containing existing data to be appended if desired
        **blk_funcs  ( list of func or None ) : list of functions that operates on individual blocks whose outputs can be processed by their corresponding data_func
        **kwargs_all ( dict )                 : dictionary of keyword args to be universally applied across all data_funcs and blk_funcs
        **data_func_kwargs ( list of dict )   : specifies specific keyward arguments for each data_func, defaults to kwargs_all
        **blk_func_kwargs( list of dict )     : same as data_func_kwargs but for blk_funcs
        **unit_labels ( list of str )         : unit string for plot label
    """
    max_out          = kwargs.get( "max_out"         , 30 )
    data_funcs       = kwargs.get( "data_funcs"      , [ ] )
    kwargs_all       = kwargs.get( "kwargs_all"      , { } )
    data_func_kwargs = kwargs.get( "data_func_kwargs", [ kwargs_all for i in data_funcs ] )
    data_names       = kwargs.get( "data_names"      , [ ] )
    output_data      = kwargs.get( "output_data"     , { } )
    blk_funcs        = kwargs.get( "blk_funcs"       , [ None for i in data_funcs ] )
    blk_func_kwargs  = kwargs.get( "blk_func_kwargs" , [ kwargs_all for i in blk_funcs ] )
    unit_labels      = kwargs.get( "unit_labels"     , [ None for i in data_names ] )
    
    if isinstance( outputnames, str ):
        outputnames = [ outputnames ]

    for i in outputnames:
        if i not in output_data:
            output_data[ i ] = {}
        for dname, unit, f, bf, f_kw, bf_kw in zip( data_names,       unit_labels,
                                                    data_funcs,       blk_funcs,
                                                    data_func_kwargs, blk_func_kwargs ):
            if dname not in output_data[ i ]:
                output_data[ i ][ dname ] = { "__func__" : f   , "__blk_func__" : bf, 
                                              "f_kw" : f_kw, "bf_kw" : bf_kw, 
                                              "data" : [], "time" : [], "unit" : unit }

    for out, out_dat in output_data.items( ):
        min_len = np.min( [ len( dat[ 'data'] ) for dat in out_dat.values( ) ] )
        for l in range( min_len,  max_out + 1 ):
            fstring = (5 - len( str( l ) ) ) * "0" + str( l )
            try:
                d  = get_last_dd( f'{dirname}/{out}_{fstring}.bin', -1, enroll_prim=True, data_type = galaxy_data );
            except IndexError:
                print( f'{dirname}/{out}_{fstring}.bin not found' )
                continue
            d.load_particles(  )
            temp_blk_dat = {}
            for dat_name, dat_fld in out_dat.items( ):
                if len( dat_fld[ "data" ] ) < l:
                    continue
                elif dat_fld[ "__blk_func__" ] is None:
                    func = dat_fld[ "__func__" ]
                    f_kw = dat_fld[ "f_kw" ] 
                    output_data[ out ][ dat_name ][ "data" ].append( func( d, **f_kw ) )
                else:
                    temp_blk_dat[ dat_name ] = []
                output_data[ out ][ dat_name ][ "time" ].append( d.globals[ 'time' ] )

            for b, bd in d.data.items( ):
                if "block" not in b:
                    continue
                for dat_name in temp_blk_dat.keys( ):
                    blk_func = out_dat[ dat_name ][ "__blk_func__" ]
                    bf_kw    = out_dat[ dat_name ][ "bf_kw" ]
                    temp_blk_dat[ dat_name ].append( blk_func( b, bd, **bf_kw ) )
            del d
            for dat_name, temp_dat in temp_blk_dat.items( ):
                func = out_dat[ dat_name ][ "__func__" ]
                f_kw = out_dat[ dat_name ][ "f_kw"] 
                output_data[ out ][ dat_name ][ "data" ].append( func( temp_dat, **f_kw ) )
            
    return output_data





def plt_time_evo( output_data, **kwargs ):
    cols = len( output_data.keys( ) )
    rows = np.max( [ len( dat.keys( ) ) for dat in output_data.values( ) ] )
    figsize  = kwargs.get( 'figsize', ( cols*5, 10 ) )
    titles   = kwargs.get(  'titles', output_data.keys( ) )
    xlabel   = kwargs.get( 'xlabel', r'Time ( Myr )' )
    ylims    = kwargs.get( "ylims", [ None for i in range( rows ) ] )
    logs     = kwargs.get( "logs", [ True for i in range( rows ) ] )

    markers  = kwargs.get( "markers", [ "o" for i in range( rows ) ] )
    if isinstance( markers, str ):
        markers = [ markers for i in range( rows ) ]
    
    linestyles = kwargs.get( "linestyles", [ "-" for i in range( rows ) ] )
    if isinstance( linestyles, str ):
        linestyles = [ linestyles for i in range( linestyles ) ]

    colors   = kwargs.get( "colors", [ plt.get_cmap( 'turbo' )( float( i ) / float( rows ) ) 
                                       for i in range( rows ) ] )
    
    fplot = kwargs.get( "fplot", [ None for i in range( rows ) ] )
    fplot_kwargs = kwargs.get( "fplot_kwargs", [ {} for i in fplot ] )


    fig, ax = plt.subplots( rows, cols, figsize = figsize, sharex = 'col', sharey = 'row' )
    plt.subplots_adjust(hspace=0, wspace=0)
    if len( ax.shape ) == 1:
        ax = np.array( [ ax ] ).T
    
    for i, ( title, data ) in enumerate( zip( titles, output_data.values( ) ) ):
        ax[ 0, i ].set_title( title )
        ax[ -1, i ].set_xlabel( xlabel )
        for j, ( ( label, dat ), ylim, 
                   log, marker, linestyle,
                   color, fp, fpkw ) in enumerate( zip( data.items( ), ylims, 
                                                     logs, markers, linestyles, 
                                                     colors, fplot, fplot_kwargs ) ):
            ax[ j, i ].plot( dat[ 'time' ], dat[ 'data' ], linestyle=linestyle, 
                                                           color=color, 
                                                           label=label,
                                                           marker=marker )
            
            if fp is not None:
                fp( ax[ j, i ], dat, data, **fpkw )
            if ylim is not None:
                ax[ j, i ].set_ylim( ylim )
            if log:
                ax[ j, i ].set_yscale( 'log' )
            ax[ j, i ].set_ylabel( dat[ 'unit' ] )
            
            if i == 0:
                ax[ j, i ].legend( )
    return fig


def draw_bar_on( ax, dat, data, **kwargs ):
    t = kwargs.get( "bar_on_time", 146 )
    ax.axvline( t, color='k', linestyle='--', label='Bar Fully On' )
    return

def draw_obs_val( ax, dat, data, **kwargs ):
    low = kwargs.get( "low", 0.7 )
    high = kwargs.get( "high", 0.9 )
    ax.fill_between( dat[ 'time' ], low, high, facecolor='magenta',alpha=0.5, label='Obs Val')
    return draw_bar_on( ax, dat, data, **kwargs )


def plt_phase( d, ax=None, **kwargs ):
    enroll_mesh_tree( d )
    enroll_T( d )
    R_out = kwargs.get( "R_out", 1e32 )
    Z_out = kwargs.get( "Z_out", 1e32 )
    densities       = np.array( [] )
    temperatures     = np.array( [] )
    masses          =  np.array( [] )
    weights         = np.array( [] )
    weightfunc = kwargs.get( "weightfunc", None )
    filterfunc = kwargs.get( "filterfunc", lambda bd : True )
    for b, bd in d.data.items(  ):
        if 'particle' in b:
            continue;
        rcyl = np.sqrt( bd['x_c'][0]**2 + bd['x_c'][1]**2 )
        zcyl = np.abs( bd['x_c'][2] )
        cell_range = ( rcyl < R_out ) & ( zcyl < Z_out )
        bd_filter    = cell_range & filterfunc( bd ) 
        densities    = np.concatenate( ( densities   ,   bd[ 'rho'  ][ bd_filter ].flatten( ) ) )
        temperatures = np.concatenate( ( temperatures,   bd[ 'T_ent'][ bd_filter ].flatten( ) ) )
        masses       = np.concatenate( ( masses      , ( bd[ 'rho'  ][ bd_filter ] * np.prod( bd['dx0'] ) ).flatten( ) ) )
        if weightfunc is not None:
            blk_wgt  = weightfunc( bd )
            weights  = np.concatenate( ( weights, blk_wgt[ bd_filter ].flatten( ) ) )
        else:
            weights  = np.concatenate( ( weights, ( bd[ 'rho'  ][ bd_filter ] * np.prod( bd['dx0'] ) ).flatten( ) * units.m0 / units.modot ) )
    rho_bins = kwargs.get( "rho_bins", 10**np.linspace(-4,4,100) )
    T_bins   = kwargs.get( "T_bins", 10**np.linspace( 0, 8, 100 ) )
    figsize  = kwargs.get( "figsize", ( 5, 5 ) )

    counts, xbins, ybins = np.histogram2d(
        densities, temperatures,
        weights=weights,
        bins=[rho_bins, T_bins],
    )

    vmin     = kwargs.get( "vmin", np.min( counts[ counts > 0 ] ) )
    vmax     = kwargs.get( "vmax", np.max( counts ) )
    plot_contour = kwargs.get( "plot_contour", False )
    levels   = kwargs.get( "levels", 10 )
    if ax is None:
        fig, ax = plt.subplots( 1, 1, figsize=figsize )
    
    if plot_contour:

        xcenters = 0.5 * ( xbins[:-1] + xbins[1:] )
        ycenters = 0.5 * ( ybins[:-1] + ybins[1:] )
        s = ax.contourf( xcenters, ycenters, counts.T, 
                         levels=10**np.linspace( np.log10( vmin ), np.log10( vmax ), levels ), 
                         norm=LogNorm(vmin=vmin, vmax=vmax) )
        norm = mpl.colors.LogNorm(vmin=s.cvalues.min(), vmax=s.cvalues.max())
        sm = plt.cm.ScalarMappable( norm=norm, cmap = s.cmap )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax )#, ticks=s.levels)
    else:
        s = ax.hist2d( densities, temperatures, weights=weights, bins=[ rho_bins, T_bins ], norm=LogNorm(vmin=vmin, vmax=vmax) )
        cbar = plt.colorbar(s[3])
    cbar_label = kwargs.get( "cbar_label", r'Mass ($M_\odot$)' )
    cbar.set_label( cbar_label )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rho$ ($m_p$ cm$^{-3}$)')
    ax.set_ylabel(r'$T$ (K)')
    return ax
