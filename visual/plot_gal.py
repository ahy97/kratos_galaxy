import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.gridspec as gridspec
from   matplotlib.colors import LogNorm, Normalize, SymLogNorm
from scipy.ndimage import gaussian_filter
mpl.rc( 'font', family = 'serif' );
mpl.rc( 'text', usetex = False    );
mpl.pyplot.rcParams[ 'image.cmap' ] = 'turbo'
mpl.rcParams[ 'font.size'        ] = 12;
mpl.rcParams[ 'axes.labelsize'   ] = 14;
mpl.rcParams[ 'legend.fontsize'  ] = 12;
mpl.rcParams[ 'legend.edgecolor' ] = 'k';
mpl.rcParams[ 'figure.facecolor' ] = 'w';
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rcParams[ "text.usetex" ] = True

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
from yt_kratos import load_kratos_yt
import yt

############################################################
# Handy functions
##############################
def figgen( figsize = None, count = None ):
    """
    Generate a new matplotlib figure with an auto-incrementing number.

    Parameters
    ----------
    figsize : tuple or None
        Figure size (width, height) in inches.
    count : int or None
        Figure number (auto-incremented if None).

    Returns
    -------
    matplotlib.figure.Figure
    """
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
    """
    Slice-plot a single field on the given figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    d : hydro_data
    f : str
        Field name.
    **kwargs
        Passed to ``slice``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    args = { 'field' : f, 'xlabel' : r'$x/l_0$', \
             'ylabel' : r'$y/l_0$'};
    for k, v in kwargs.items(  ):
        args[ k ] = v;
    ax = slice( d, args, fig );
    return ax;
#

def f_plot( d, f, fig=None, ** args ):
    """
    Convenience wrapper for single-field slice plot.

    Parameters
    ----------
    d : hydro_data
    f : str
        Field name.
    fig : matplotlib.figure.Figure or None
    **args
        Passed to ``plot_field``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not 'loc' in args:
        args[ 'loc' ] = 0;
    if fig is None:
        fig = figgen( figsize = ( 4.5, 4 ) );
    return plot_field( fig,\
                       d, f, ** args );
#
def fstring( i ):
    return ( 5 - len( str( i ) ) ) * "0" + str( i )

def f_plot_grid( d, **kwargs ):
    """
    Multi-panel plot of several fields across multiple projection axes
    with optional particle velocity overlay.

    Parameters
    ----------
    d : galaxy_data
    flds : list of str
        Fields to plot (default ``['rho', 'T_ent', 'met', 'rho']``).
    axes : list of int
        Projection axes (default ``[2, 1]``).
    names : list of str
        Colourbar labels.
    zlims : list of (vmin, vmax)
        Per-field colour ranges.
    z0s : list of float
        Slice positions.
    xlims, ylims : list of (float, float)
        Axis limits per projection.
    log : list of bool
        Log-scaling flags per field.
    locs : list of list
        Slice locations per axis per field.
    part_plot_i : int or list of int
        Which field panels to overlay particle velocity quivers on.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
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
    """
    Plot time-evolution data produced by ``time_evo``.

    Parameters
    ----------
    output_data : dict
        Output from ``time_evo``.
    titles : iterable
        Column titles (defaults to output_data keys).
    xlabel : str
    ylims : list of (vmin, vmax)
    logs : list of bool
    markers, linestyles, colors : list
        Per-row style settings.
    fplot, fplot_kwargs : list
        Per-row annotation callbacks and their kwargs.
    savefile : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
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
            
            if any( isinstance( el, tuple ) for el in color ):#isinstance( color, tuple ):
                print( color )
                arrdat = np.array( dat[ 'data' ] )
                
                if isinstance( linestyle, tuple ):
                    lnst = linestyle
                else:
                    lnst = tuple( [ linestyle for k in color ] )

                if isinstance( label, tuple ):
                    lbl = label
                else:
                    lbl = tuple( [ label for k in color ] )
                
                if isinstance( marker, tuple ):
                    mkr = marker
                else:
                    mkr = tuple( [ marker for k in color ] )

                for k, ( ls, c, lb, mk ) in enumerate( zip( lnst, color, lbl, mkr ) ):
                    ax[ j, i ].plot( dat[ 'time' ], arrdat[ :,k ], linestyle=ls, 
                                                                   color=c, 
                                                                   label=lb,
                                                                   marker=mk )
            else:   
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
                ax[ j, i ].legend( loc='upper left' )
    savefile = kwargs.get( "savefile", None )
    if savefile is not None:
        plt.savefig( savefile, bbox_inches='tight' )
    return fig


def draw_bar_on( ax, dat, data, **kwargs ):
    """
    Draw a vertical dashed line at *bar_on_time* on a time-evolution axis.
    """
    t = kwargs.get( "bar_on_time", 146 )
    ax.axvline( t, color='k', linestyle='--', label='Bar Fully On' )
    return

def draw_obs_val( ax, dat, data, **kwargs ):
    """
    Fill between *low* and *high* values to indicate observed range,
    then call ``draw_bar_on``.
    """
    low = kwargs.get( "low", 0.7 )
    high = kwargs.get( "high", 0.9 )
    ax.fill_between( dat[ 'time' ], low, high, facecolor='magenta',alpha=0.5, label='Obs Val')
    return draw_bar_on( ax, dat, data, **kwargs )


def plt_phase( d, fig=None, ax=None, **kwargs ):
    """
    2D phase diagram (density vs. temperature) of gas cells weighted by mass.

    Parameters
    ----------
    d : galaxy_data
    fig : matplotlib.figure.Figure or None
    ax : matplotlib.axes.Axes or None
    R_out, Z_out : float
        Cylindrical selection radius / height.
    cmap : str
        Colour map (default ``'turbo'``).
    show_cbar : bool
        Show colour bar (default True).
    weightfunc : callable or None
        Custom per-cell weight function ``f(bd) -> array``.
    filterfunc : callable
        Per-block filter ``f(bd) -> bool array``.
    rho_bins, T_bins : ndarray
        Bin edges for density and temperature.
    vmin, vmax : float
        Colour scale limits.
    plot_contour : bool
        Use contourf instead of hist2d (default False).
    levels : int
        Number of contour levels.
    contour_fill : bool
        Fill contours (True) or line contours (False).

    Returns
    -------
    fig, ax
    """
    enroll_mesh_tree( d )
    enroll_T( d )
    R_out = kwargs.get( "R_out", 1e32 )
    Z_out = kwargs.get( "Z_out", 1e32 )
    densities       = np.array( [] )
    temperatures     = np.array( [] )
    masses          =  np.array( [] )
    weights         = np.array( [] )
    cmap       = kwargs.get( "cmap", plt.get_cmap( 'turbo' ) )
    show_cbar  = kwargs.get( "show_cbar", True )
    weightfunc = kwargs.get( "weightfunc", None )
    filterfunc = kwargs.get( "filterfunc", lambda bd : True )
    for b, bd in d.data.items(  ):
        if 'particle' in b:
            continue;
        rcyl = np.sqrt( bd['x'][0]**2 + bd['x'][1]**2 )
        zcyl = np.abs( bd['x'][2] )
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
    # --- Inherit norm from existing plot on ax, or build a new one ----------
    existing_norm = None
    if ax is not None and ax.collections:
        existing_norm = ax.collections[-1].norm   # reuse last artist's norm

    if existing_norm is not None:
        # Pull vmin/vmax out of the inherited norm; kwargs can still override.
        vmin = kwargs.get( "vmin", existing_norm.vmin )
        vmax = kwargs.get( "vmax", existing_norm.vmax )
    else:
        vmin = kwargs.get( "vmin", np.min( counts[ counts > 0 ] ) )
        vmax = kwargs.get( "vmax", np.max( counts ) )
    #vmin     = kwargs.get( "vmin", np.min( counts[ counts > 0 ] ) )
    #vmax     = kwargs.get( "vmax", np.max( counts ) )
    plot_contour = kwargs.get( "plot_contour", False )
    levels   = kwargs.get( "levels", 10 )
    contour_fill = kwargs.get( "contour_fill", True )
    if ax is None:
        fig, ax = plt.subplots( 1, 1, figsize=figsize )
    if contour_fill:
        contour_func = ax.contourf
    else:
        contour_func = ax.contour
    if plot_contour:

        xcenters = 0.5 * ( xbins[:-1] + xbins[1:] )
        ycenters = 0.5 * ( ybins[:-1] + ybins[1:] )
        s = contour_func( xcenters, ycenters, counts.T, 
                         levels=10**np.linspace( np.log10( vmin ), np.log10( vmax ), levels ), 
                         norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap )
        norm = mpl.colors.LogNorm(vmin=s.cvalues.min(), vmax=s.cvalues.max())
        sm = plt.cm.ScalarMappable( norm=norm, cmap = s.cmap )
        sm.set_array([])
        if show_cbar:
            cbar = fig.colorbar(sm, ax=ax )#, ticks=s.levels)
    else:
        s = ax.hist2d( densities, temperatures, weights=weights, bins=[ rho_bins, T_bins ], norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap )
        if show_cbar:
            cbar = plt.colorbar(s[3])
    if show_cbar:
        cbar_label = kwargs.get( "cbar_label", r'Mass ($M_\odot$)' )
        cbar.set_label( cbar_label )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rho$ ($m_p$ cm$^{-3}$)')
    ax.set_ylabel(r'$T$ (K)')
    return fig, ax

def plt_snaps( fnames, savefile=None, **kwargs ):
    """
    Multi-panel snapshot plot of a single field with a shared colourbar.

    Parameters
    ----------
    fnames : str or list of str
        Base filename(s) for Kratos outputs.
    savefile : str or None
        Output file path.
    rows, cols : int
        Grid layout (default 1, 1).
    figsize_base : float
    cbar_rat, cbar_space : float
        Colourbar width / spacing ratios.
    outs : ndarray
        Output numbers to plot.
    xlim, ylim : list
        Axis limits.
    fld : str
        Field to plot (default ``'rho'``).
    cbar_label : str
    xlabel, ylabel : str
    x_ticks, x_tick_labels, y_ticks, y_tick_labels

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = kwargs.get( "rows", 1 )
    cols = kwargs.get( "cols", 1 )
    figsize_base = kwargs.get( "figsize_base", 6 )
    cbar_rat   = kwargs.get( "cbar_rat", 0.1 )
    cbar_space = kwargs.get( "cbar_space", 0.1 )
    outs = kwargs.get( "outs", [ 0 ] )
    xlim = kwargs.get( "xlim", [ -1e3, 1e3 ] )
    ylim = kwargs.get( "ylim", [ -1e3, 1e3 ] )
    x_ticks       = kwargs.get( "x_ticks"      , [ -500, 0, 500 ] )
    x_tick_labels = kwargs.get( "x_tick_labels", [ "-0.5", "0", "0.5" ] )
    y_ticks       = kwargs.get( "y_ticks"      , [ -500, 0, 500 ] )
    y_tick_labels = kwargs.get( "y_tick_labels", [ "-0.5", "0", "0.5" ] )   
    xlabel_       = kwargs.get( "xlabel", r"$x$ (kpc)" )
    ylabel_       = kwargs.get( "ylabel", r"$y$ (kpc)" )
    fld  = kwargs.get( "fld", "rho" ); 
    cbar_label = kwargs.get( "cbar_label", fld )
    fontsize = 20
    ###########################################################
    xlen = xlim[ 1 ] - xlim[ 0 ]
    ylen = ylim[ 1 ] - ylim[ 0 ]
    aspect = ylen / xlen
    figsize = [ figsize_base, figsize_base ]
    if aspect > 1:
        figsize[ 1 ] *= aspect
    elif aspect < 1:
        figsize[ 0 ] *= aspect
    
    

    if savefile is not None and not isinstance( savefile, str ):
        raise TypeError( "savefile must be string or Nonetype" )
    outs = np.array( outs ).reshape( rows, cols )
    f = plt.figure(figsize=( figsize[ 0 ]*( cols + cbar_rat + cbar_space ),
                             figsize[ 1 ]*rows))
    width_ratios = [ 1 for i in range( cols ) ]
    width_ratios.append( cbar_space )
    width_ratios.append( cbar_rat )
    gs = gridspec.GridSpec(rows, cols + 2, figure=f, width_ratios=width_ratios, hspace=0, wspace=0)
    f.canvas.draw()
    for row, o in enumerate( outs ):
        for col, out in enumerate( o ):
            d  = get_last_dd( f'{fname}_{fstring( out )}.bin', -1, enroll_prim=True, data_type = galaxy_data )
            enroll_T( d )
            labelbottom = False
            if row == rows - 1:
                labelbottom = True
            labelleft = False
            if col == 0:
                labelleft = True

            if labelleft:
                ylabel = ylabel_
            else:
                ylabel=None

            if labelbottom:
                xlabel = xlabel_
            else:
                xlabel=None

            ax, _ = f_plot( d, fld, f, rect=gs[ row, col ], 
                            no_cbar=True, fontsize=fontsize, **kwargs )
            ax.set_xlim( xlim )
            ax.set_ylim( ylim )
            ax.set_xlabel( xlabel, fontsize=fontsize )
            ax.set_ylabel( ylabel, fontsize=fontsize )
            last_pcm = ax.collections[ -1 ]
            ax.tick_params( labelbottom=labelbottom, labelleft=labelleft )
            ax.set_xticks( x_ticks, labels=x_tick_labels, fontsize=fontsize )
            ax.set_yticks( y_ticks, labels=y_tick_labels, fontsize=fontsize )
            ax.text( xlim[ 0 ] * 0.9, ylim[ 1 ] * 0.8, f"{round( d.globals['time'], 2 )} Myr", color='black', fontsize=fontsize, 
                     bbox=dict(boxstyle="square", facecolor="white", edgecolor="black") )
    cbar_ax = f.add_subplot(gs[:, -1])
    cbar    = f.colorbar(last_pcm, cax=cbar_ax)
    cbar.set_label( cbar_label, fontsize=fontsize)
    cbar.ax.tick_params( labelsize=fontsize )
    if savefile is not None:
        plt.savefig( savefile, dpi=300, bbox_inches='tight' )
    return f

def plt_snaps_yt( fnames, savefile=None, **kwargs ):
    """
    Multi-panel snapshot plot using yt projections/slices with a shared
    colourbar. Supports stacking multiple output series.

    Parameters
    ----------
    fnames : str or list of str
        Base filename(s).
    savefile : str or None
    iso : bool
        Isothermal mode for ``load_kratos_yt``.
    rows, cols : int
        Grid layout.
    figsize_base, cbar_rat, cbar_space, fontsize, resolution_base
    outs : ndarray
        Output numbers.
    xlim, ylim : list
    x_ticks, x_tick_labels, y_ticks, y_tick_labels
    xlabel, ylabel : str
    fld : tuple or str
        yt field (default ``('gas', 'density')``).
    axis : str
        Projection axis (default ``'z'``).
    projection : bool
        Use projection (True) or slice (False).
    weight_field : tuple or None
        Weight field for projection.
    unit : str
        yt unit string.
    zlim : (vmin, vmax) or None
    NormClass : matplotlib.colors.Normalize
    derived_fields : dict
        ``{name: (func, units)}`` for custom yt fields.
    stack : str
        ``'row'`` or ``'col'`` to stack multiple fname series.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance( fnames, str ):
        fnames = [ fnames ]
    iso  = kwargs.get( "iso", False )
    rows = kwargs.get( "rows", 1 )
    cols = kwargs.get( "cols", 1 )
    figsize_base = kwargs.get( "figsize_base", 6 )
    cbar_rat   = kwargs.get( "cbar_rat", 0.1 )
    cbar_space = kwargs.get( "cbar_space", 0.1 )
    outs = np.array( kwargs.get( "outs", [ 0 ] ) )
    xlim = kwargs.get( "xlim", [ -1e3, 1e3 ] )
    ylim = kwargs.get( "ylim", [ -1e3, 1e3 ] )
    x_ticks       = kwargs.get( "x_ticks"      , [ -500, 0, 500 ] )
    x_tick_labels = kwargs.get( "x_tick_labels", [ "-0.5", "0", "0.5" ] )
    y_ticks       = kwargs.get( "y_ticks"      , [ -500, 0, 500 ] )
    y_tick_labels = kwargs.get( "y_tick_labels", [ "-0.5", "0", "0.5" ] )   
    xlabel_       = kwargs.get( "xlabel", r"$x$ (kpc)" )
    ylabel_       = kwargs.get( "ylabel", r"$y$ (kpc)" )
    stack         = kwargs.get( "stack", None )

    fld  = kwargs.get( "fld", ( "gas", "density" ) );
    if isinstance( fld, str ):
        fld = ( "gas", fld )
    axis = kwargs.get( "axis", "z" ) 
    resolution_base = kwargs.get( "resolution_base", 1024 )
    proj = kwargs.get( "projection", True )
    weight_field = kwargs.get( "weight_field", ("gas", "density") if proj else None )
    unit = kwargs.get( "unit", "mp/cm**2" )
    cbar_label = kwargs.get( "cbar_label", f"{fld} ({unit})" )
    zlim = kwargs.get( "zlim", None )
    fontsize = kwargs.get( "fontsize", 20 )
    cmap = kwargs.get( "cmap", "turbo" )
    NormClass = kwargs.get( "norm", None )
    if NormClass is None:
        log = kwargs.get( "log", True )
        NormClass = LogNorm if log else Normalize
    derived_fields = kwargs.get( "derived_fields", {} )
    for name, (func, d_units) in derived_fields.items():
        yt.add_field( ( "gas", name ), function=func,
                       units=d_units, sampling_type="cell",
                       force_override=True )
        
    ###########################################################
    xlen = xlim[ 1 ] - xlim[ 0 ]
    ylen = ylim[ 1 ] - ylim[ 0 ]
    aspect = ylen / xlen
    figsize = [ figsize_base, figsize_base ]
    resolution = [ resolution_base, resolution_base ]
    if aspect > 1:
        figsize[ 1 ] *= aspect
        resolution[ 1 ] = int( resolution[ 0 ] * aspect )
    elif aspect < 1:
        figsize[ 0 ] *= aspect
        resolution[ 0 ] = int( resolution[ 0 ] * aspect )
    
    if savefile is not None and not isinstance( savefile, str ):
        raise TypeError( "savefile must be string or Nonetype" )
    
    outs = outs.reshape( rows, cols )
    #outs = np.tile( outs, ( len( fnames ), 1 ) )
    #print( outs )
    #if stack == "col":
    #    outs = outs.T
    #print( outs )
    if len( fnames ) > 1:
        if stack == "row":
            rows *= len( fnames )
            outs = np.tile( outs, ( len( fnames ), 1 ) )
        elif stack == "col":
            cols *= len( fnames )
            outs = np.tile( outs, ( 1, len( fnames ) ) )
    print( outs )
    f = plt.figure(figsize=( figsize[ 0 ]*( outs.shape[ 1 ] + cbar_rat + cbar_space ),
                             figsize[ 1 ]*outs.shape[ 0 ]))
    width_ratios = [ 1 for i in range( cols ) ]
    width_ratios.append( cbar_space )
    width_ratios.append( cbar_rat )
    gs = gridspec.GridSpec(rows, cols + 2, figure=f, width_ratios=width_ratios, hspace=0, wspace=0)
    f.canvas.draw()
    for row, o in enumerate( outs ):
        for col, out in enumerate( o ):
            if stack == "row":
                fname = fnames[ row ]
            elif stack == 'col':
                fname = fnames[ col ]
            d  = get_last_dd( f'{fname}_{fstring( out )}.bin', -1, enroll_prim=True, data_type = galaxy_data )
            dyt = load_kratos_yt( d, iso=iso )
            if proj:
                prj = yt.ProjectionPlot( dyt, axis, fld,
                                        weight_field=weight_field,
                                        width =[ ( ( xlim[ 1 ] - xlim[ 0 ] )/1e3,"kpc"),
                                                 ( ( ylim[ 1 ] - ylim[ 0 ] )/1e3,"kpc") ] )
            else:
                prj = yt.SlicePlot( dyt, axis, fld, width =[ ( ( xlim[ 1 ] - xlim[ 0 ] )/1e3,"kpc"),
                                                                             ( ( ylim[ 1 ] - ylim[ 0 ] )/1e3,"kpc") ] )
            frb = prj.data_source.to_frb((( xlim[ 1 ] - xlim[ 0 ] )/1e3, "kpc"), resolution, height = ( ( ylim[ 1 ] - ylim[ 0 ] )/1e3,"kpc") )
            img = np.array( frb[ fld ].to( unit ) )
            labelbottom = False
            if row == rows - 1:
                labelbottom = True
            labelleft = False
            if col == 0:
                labelleft = True

            if labelleft:
                ylabel = ylabel_
            else:
                ylabel=None

            if labelbottom:
                xlabel = xlabel_
            else:
                xlabel=None

            ext = [ xlim[ 0 ], xlim[ 1 ], ylim[ 0 ], ylim[ 1 ] ]
            ax = f.add_subplot( gs[ row, col ] )
            if zlim is None:
                norm = NormClass()
            else:
                norm = NormClass( vmin=zlim[ 0 ], vmax=zlim[ 1 ] )
            im = ax.imshow(
                            img,
                            origin="lower",
                            extent=ext,
                            norm=norm,
                            cmap=cmap,
                            interpolation="nearest"
                            )
            ax.set_xlim( xlim )
            ax.set_ylim( ylim )
            ax.set_xlabel( xlabel, fontsize=fontsize )
            ax.set_ylabel( ylabel, fontsize=fontsize )
            #last_pcm = ax.collections[ -1 ]
            ax.tick_params( labelbottom=labelbottom, labelleft=labelleft )
            ax.set_xticks( x_ticks, labels=x_tick_labels, fontsize=fontsize )
            ax.set_yticks( y_ticks, labels=y_tick_labels, fontsize=fontsize )
            ax.text( xlim[ 0 ] * 0.9, ylim[ 1 ] * 0.8, f"{round( d.globals['time'], 2 )} Myr", color='black', fontsize=fontsize, 
                     bbox=dict(boxstyle="square", facecolor="white", edgecolor="black") )
    cbar_ax = f.add_subplot(gs[:, -1])
    cbar = f.colorbar( im, cax=cbar_ax )
    cbar.set_label( cbar_label, fontsize=fontsize)
    cbar.ax.tick_params( labelsize=fontsize )
    if savefile is not None:
        plt.savefig( savefile, dpi=300, bbox_inches='tight' )
    return f


from mpl_toolkits.axes_grid1 import make_axes_locatable


def plt_fields(d, flds, savefile=None, **kwargs):
    """
    Multi-panel plot showing *multiple fields* from a *single* Kratos snapshot
    (analogous to ``plt_snaps`` which plots a single field across multiple
    snapshots).  Each panel gets its own colour bar.

    Parameters
    ----------
    d : galaxy_data
        A single simulation data object (already loaded).
    flds : list of str
        Field names to plot, one per panel.
    savefile : str or None
        Output file path.
    rows : int, optional
        Number of rows in the grid.  If neither ``rows`` nor ``cols`` is
        given, defaults to a single row.
    cols : int, optional
        Number of columns.  If ``None``, computed from ``len(flds) // rows``.
    names : list of str, optional
        Display titles / colourbar labels for each field.  Defaults to
        ``flds``.
    zlims : list of (vmin, vmax) or dict, optional
        Per-field colour range.  Can be a list in the same order as *flds*
        or a dict keyed by field name.  Panels without an entry use automatic
        scaling.
    axis : int, optional
        Slice axis (0=x, 1=y, 2=z).  Default 2.
    loc : float, optional
        Slice location along *axis* in code units (default 0).
    log : bool, optional
        Use logarithmic colour scale (default True).
    unit_conv : dict, optional
        Field-specific conversion functions ``{fld: func(bd) -> array}`` that
        are enrolled before slicing.
    cmap : str or colormap, optional
        Matplotlib colourmap name (default ``'turbo'``).
    figsize_base : float, optional (default 6)
    fontsize : int, optional (default 20)
    cbar_fraction : float, optional
        Fraction of each panel width taken by its colour bar (default 0.05).
    cbar_pad : float, optional
        Padding between panel and its colour bar (default 0.05).
    xlim, ylim, x_ticks, x_tick_labels, y_ticks, y_tick_labels, xlabel,
        ylabel : same meaning as in ``plt_snaps``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = kwargs.get("rows", None)
    cols = kwargs.get("cols", None)
    nflds = len(flds)
    if rows is None and cols is None:
        rows, cols = 1, nflds
    elif rows is None:
        rows = int(np.ceil(nflds / cols))
    elif cols is None:
        cols = int(np.ceil(nflds / rows))

    figsize_base = kwargs.get("figsize_base", 6)
    cbar_fraction = kwargs.get("cbar_fraction", 0.05)
    cbar_pad = kwargs.get("cbar_pad", 0.05)

    xlim = kwargs.get("xlim", [-1e3, 1e3])
    ylim = kwargs.get("ylim", [-1e3, 1e3])
    x_ticks = kwargs.get("x_ticks", [-500, 0, 500])
    x_tick_labels = kwargs.get("x_tick_labels", ["-0.5", "0", "0.5"])
    y_ticks = kwargs.get("y_ticks", [-500, 0, 500])
    y_tick_labels = kwargs.get("y_tick_labels", ["-0.5", "0", "0.5"])
    xlabel_ = kwargs.get("xlabel", r"$x$ (kpc)")
    ylabel_ = kwargs.get("ylabel", r"$y$ (kpc)")
    fontsize = kwargs.get("fontsize", 20)
    cmap = kwargs.get("cmap", "turbo")
    log = kwargs.get("log", True)
    axis = kwargs.get("axis", 2)
    loc = kwargs.get("loc", 0)

    names = kwargs.get("names", list(flds))
    if isinstance(names, dict):
        names = [names.get(f, f) for f in flds]

    zlims = kwargs.get("zlims", {})
    if not isinstance(zlims, dict):
        zlims = {f: z for f, z in zip(flds, zlims)}

    unit_conv = kwargs.get("unit_conv", {})

    enroll_T(d)

    for fld_name, func in unit_conv.items():
        d.enroll_field(fld_name, func)

    xlen = xlim[1] - xlim[0]
    ylen = ylim[1] - ylim[0]
    aspect = ylen / xlen
    figsize = [figsize_base, figsize_base]
    if aspect > 1:
        figsize[1] *= aspect
    elif aspect < 1:
        figsize[0] *= aspect

    if savefile is not None and not isinstance(savefile, str):
        raise TypeError("savefile must be string or Nonetype")

    cbar_width_factor = 1 + cbar_fraction + cbar_pad
    f = plt.figure(
        figsize=(figsize[0] * cols * cbar_width_factor, figsize[1] * rows)
    )
    gs = gridspec.GridSpec(
        rows, cols, figure=f, hspace=0.02, wspace=0.08
    )
    f.canvas.draw()

    for idx, fld in enumerate(flds):
        row = idx // cols
        col = idx % cols

        labelbottom = row == rows - 1
        labelleft = col == 0

        if zlims.get(fld) is not None:
            z0, z1 = zlims[fld]
            panel_kw = {"zlim": [z0, z1]}
        else:
            panel_kw = {}

        ax, pcm = f_plot(
            d,
            fld,
            f,
            rect=gs[row, col],
            no_cbar=True,
            fontsize=fontsize,
            log=log,
            axis=axis,
            loc=loc,
            cmap=cmap,
            **panel_kw,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel_ if labelbottom else "", fontsize=fontsize)
        ax.set_ylabel(ylabel_ if labelleft else "", fontsize=fontsize)
        ax.tick_params(labelbottom=labelbottom, labelleft=labelleft)
        ax.set_xticks(x_ticks, labels=x_tick_labels, fontsize=fontsize)
        ax.set_yticks(y_ticks, labels=y_tick_labels, fontsize=fontsize)

        if pcm is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=f"{cbar_fraction * 100:.0f}%", pad=cbar_pad)
            cbar = f.colorbar(pcm, cax=cax)
            cbar.set_label(names[idx], fontsize=fontsize * 0.7)
            cbar.ax.tick_params(labelsize=fontsize * 0.6)

        ax.text(
            xlim[0] * 0.9,
            ylim[1] * 0.8,
            f"{round(d.globals['time'], 2)} Myr",
            color="black",
            fontsize=fontsize * 0.7,
            bbox=dict(boxstyle="square", facecolor="white", edgecolor="black"),
        )
        ax.set_title(names[idx], fontsize=fontsize)

    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
    return f


def plt_fields_yt(d, flds, savefile=None, **kwargs):
    """
    Multi-panel plot showing *multiple yt fields* from a *single* Kratos
    snapshot (analogous to ``plt_snaps_yt`` which plots a single field across
    multiple snapshots).  Each panel gets its own colour bar.

    Parameters
    ----------
    d : galaxy_data
        A single simulation data object (already loaded).
    flds : list of str
        yt field names (e.g. ``("gas", "density")``, ``("gas", "temperature")``).
    savefile : str or None
        Output file path.
    rows, cols : int, optional
        Grid layout (same semantics as ``plt_fields``).
    names : list of str, optional
        Panel titles (defaults to field name strings).
    units : list of str or dict, optional
        Per-field unit strings for yt conversion.  Default ``"mp/cm**2"``
        when ``projection=True``, else ``"g/cm**3"``.
    zlims : list of (vmin, vmax) or dict, optional
        Per-field colour range.
    axis : str, optional
        ``"x"`` / ``"y"`` / ``"z"`` (default ``"z"``).
    projection : bool, optional
        Integrated projection (True) or single-plane slice (False).  Default
        True.
    iso : bool, optional
        Passed through to ``load_kratos_yt``.
    resolution_base : int, optional (default 1024)
    cbar_fraction : float, optional
        Fraction of each panel width taken by its colour bar (default 0.05).
    cbar_pad : float, optional
        Padding between panel and its colour bar (default 0.05).
    figsize_base, fontsize, xlim, ylim, x_ticks, x_tick_labels, y_ticks,
        y_tick_labels, xlabel, ylabel : same as ``plt_fields``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = kwargs.get("rows", None)
    cols = kwargs.get("cols", None)
    nflds = len(flds)
    if rows is None and cols is None:
        rows, cols = 1, nflds
    elif rows is None:
        rows = int(np.ceil(nflds / cols))
    elif cols is None:
        cols = int(np.ceil(nflds / rows))

    figsize_base = kwargs.get("figsize_base", 6)
    cbar_fraction = kwargs.get("cbar_fraction", 0.05)
    cbar_pad = kwargs.get("cbar_pad", 0.05)

    xlim = kwargs.get("xlim", [-1e3, 1e3])
    ylim = kwargs.get("ylim", [-1e3, 1e3])
    x_ticks = kwargs.get("x_ticks", [-500, 0, 500])
    x_tick_labels = kwargs.get("x_tick_labels", ["-0.5", "0", "0.5"])
    y_ticks = kwargs.get("y_ticks", [-500, 0, 500])
    y_tick_labels = kwargs.get("y_tick_labels", ["-0.5", "0", "0.5"])
    xlabel_ = kwargs.get("xlabel", r"$x$ (kpc)")
    ylabel_ = kwargs.get("ylabel", r"$y$ (kpc)")
    fontsize = kwargs.get("fontsize", 20)

    axis = kwargs.get("axis", "z")
    projection = kwargs.get("projection", True)
    weight_field = kwargs.get("weight_field", ("gas", "density") if projection else None)
    iso = kwargs.get("iso", False)
    resolution_base = kwargs.get("resolution_base", 1024)
    NormClass = kwargs.get("norm", None)
    if NormClass is None:
        log_fld = kwargs.get("log", True)
        NormClass = LogNorm if log_fld else Normalize
    derived_fields = kwargs.get("derived_fields", {})

    names = kwargs.get("names", [str(f) for f in flds])
    if isinstance(names, dict):
        names = [names.get(f, str(f)) for f in flds]

    default_unit = "mp/cm**2" if projection else "g/cm**3"
    units_kv = kwargs.get("units", {})
    if not isinstance(units_kv, dict):
        units_kv = {f: u for f, u in zip(flds, units_kv)}

    zlims = kwargs.get("zlims", {})
    if not isinstance(zlims, dict):
        zlims = {f: z for f, z in zip(flds, zlims)}

    xlen = xlim[1] - xlim[0]
    ylen = ylim[1] - ylim[0]
    aspect = ylen / xlen
    figsize = [figsize_base, figsize_base]
    resolution = [resolution_base, resolution_base]
    if aspect > 1:
        figsize[1] *= aspect
        resolution[1] = int(resolution[0] * aspect)
    elif aspect < 1:
        figsize[0] *= aspect
        resolution[0] = int(resolution[0] * aspect)

    if savefile is not None and not isinstance(savefile, str):
        raise TypeError("savefile must be string or Nonetype")

    cbar_width_factor = 1 + cbar_fraction + cbar_pad
    f = plt.figure(
        figsize=(figsize[0] * cols * cbar_width_factor, figsize[1] * rows)
    )
    gs = gridspec.GridSpec(
        rows, cols, figure=f, hspace=0.02, wspace=0.08
    )
    f.canvas.draw()

    dyt = load_kratos_yt(d, iso=iso)

    for name, (func, d_units) in derived_fields.items():
        yt.add_field(("gas", name), function=func,
                      units=d_units, sampling_type="cell",
                      force_override=True)

    for idx, fld in enumerate(flds):
        row = idx // cols
        col = idx % cols

        unit = units_kv.get(fld, default_unit)
        zlim = zlims.get(fld, None)

        if projection:
            prj = yt.ProjectionPlot(
                dyt,
                axis,
                fld,
                weight_field=weight_field,
                width=[
                    ((xlim[1] - xlim[0]) / 1e3, "kpc"),
                    ((ylim[1] - ylim[0]) / 1e3, "kpc"),
                ],
            )
        else:
            prj = yt.SlicePlot(
                dyt,
                axis,
                fld,
                width=[
                    ((xlim[1] - xlim[0]) / 1e3, "kpc"),
                    ((ylim[1] - ylim[0]) / 1e3, "kpc"),
                ],
            )

        frb = prj.data_source.to_frb(
            ((xlim[1] - xlim[0]) / 1e3, "kpc"),
            resolution,
            height=((ylim[1] - ylim[0]) / 1e3, "kpc"),
        )
        img = np.array(frb[fld].to(unit))

        labelbottom = row == rows - 1
        labelleft = col == 0

        ext = [xlim[0], xlim[1], ylim[0], ylim[1]]
        ax = f.add_subplot(gs[row, col])
        if zlim is None:
            norm = NormClass()
        else:
            norm = NormClass(vmin=zlim[0], vmax=zlim[1])
        im = ax.imshow(
            img,
            origin="lower",
            extent=ext,
            norm=norm,
            cmap="turbo",
            interpolation="nearest",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel_ if labelbottom else "", fontsize=fontsize)
        ax.set_ylabel(ylabel_ if labelleft else "", fontsize=fontsize)
        ax.tick_params(labelbottom=labelbottom, labelleft=labelleft)
        ax.set_xticks(x_ticks, labels=x_tick_labels, fontsize=fontsize)
        ax.set_yticks(y_ticks, labels=y_tick_labels, fontsize=fontsize)

        if im is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=f"{cbar_fraction * 100:.0f}%", pad=cbar_pad)
            cbar = f.colorbar(im, cax=cax)
            cbar.set_label(names[idx], fontsize=fontsize * 0.7)
            cbar.ax.tick_params(labelsize=fontsize * 0.6)

        ax.text(
            xlim[0] * 0.9,
            ylim[1] * 0.8,
            f"{round(d.globals['time'], 2)} Myr",
            color="black",
            fontsize=fontsize * 0.7,
            bbox=dict(boxstyle="square", facecolor="white", edgecolor="black"),
        )
        ax.set_title(names[idx], fontsize=fontsize)

    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
    return f

def plot_lvd( d, **kwargs ):
    """
    Longitude-velocity diagram (l-b and l-v maps) from a Milky Way-like
    viewing geometry at the solar position.

    Computes line-of-sight velocities and column densities via exact
    ray-cell intersection lengths, then makes 2D histograms smoothed
    with Gaussian filters.

    Parameters
    ----------
    d : galaxy_data
    reflect : str
        Hemisphere reflection (``'xy'``, ``'xz'``, or ``'yz'``).
    tempcut : float or None
        Exclude cells hotter than this temperature (K).
    lbins, bbins, vbins : ndarray
        Bin edges for l, b, v_los.
    vmin, vmax : float
        Colour scale limits (column density).
    sigma_lb, sigma_lv : tuple
        Gaussian smoothing sigma in (l, b) and (l, v) panels.
    figsize_base : tuple
    fontsize : int
    savefile : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    reflect = kwargs.get( "reflect", "yz" )
    tempcut = kwargs.get( "tempcut", None )
    lbins   = kwargs.get( "lbins" )
    bbins   = kwargs.get( "bbins" )
    vbins   = kwargs.get( "vbins" )
    dyt = load_kratos_yt( d )
    lsr_u = 10.6 # km/s
    lsr_v = 10.7
    lsr_w = 7.6
    v_lsr = 236
    rsun  = 8150
    phisun = 28
    phisun += 180
    barpattern = 40 #km/s/kpc
    xsun = rsun*np.cos(phisun*np.pi/180.)
    ysun = rsun*np.sin(phisun*np.pi/180.)
    zsun = 25
    rsunvec = np.array( [ xsun, ysun, zsun ] )
    omega_sun = (v_lsr+lsr_v)/rsun - barpattern / 1000
    velxsun = -omega_sun*ysun - (lsr_u)*np.cos(phisun*np.pi/180.)
    velysun =  omega_sun*xsun - (lsr_u)*np.sin(phisun*np.pi/180.)
    velzsun = lsr_w

    ad = dyt.all_data()
    vx = np.array( ad['velocity_x'].in_units( "km/s" ) )
    vy = np.array( ad['velocity_y'].in_units( "km/s" ) )
    vz = np.array( ad['velocity_z'].in_units( "km/s" ) )
    x  = np.array( ad['x'].in_units( "pc" ) )
    y  = np.array( ad['y'].in_units( "pc" ) )
    z  = np.array( ad['z'].in_units( "pc" ) )
    
    if reflect == "xy":
        z = -z
        vz = -vz
    if reflect == "xz":
        y = -y
        vy = -vy
    if reflect == "yz":
        x = -x
        vx = -vx
    den = np.array(  ad['gas','density'].in_units( "mp/cm**3" ) )
    temp = np.array( ad["athena_pp","temperature"] )
    gc_dir   = -rsunvec[:2] / np.linalg.norm(rsunvec[:2])   # Sun → GC unit vector
    datavec  = np.c_[x, y] - rsunvec[:2]                    # Sun → cell
    solarvec = np.ones_like(datavec) * gc_dir

    cosang    = np.einsum("ij,ij->i", datavec, solarvec)
    sinang    = np.cross(datavec, solarvec)
    longitude = -np.arctan2(sinang, cosang) * 180. / np.pi
    dist_xy  = np.sqrt((x - xsun)**2 + (y - ysun)**2)   # pc, projected separation
    latitude = np.arctan2(z - zsun, dist_xy) * 180. / np.pi 
    e_LOS = np.array([x-xsun, y-ysun, z-zsun])/np.sqrt((x-xsun)**2+(y-ysun)**2+((z-zsun)**2))
    dx_cm = np.array(ad[('index', 'dx')].in_units('cm'))
    dy_cm = np.array(ad[('index', 'dy')].in_units('cm'))
    dz_cm = np.array(ad[('index', 'dz')].in_units('cm'))

    # Half-widths of each cell
    hx = dx_cm / 2.
    hy = dy_cm / 2.
    hz = dz_cm / 2.

    # e_LOS components (3, N) — unit vector from Sun through each cell centre
    ex = e_LOS[0]
    ey = e_LOS[1]
    ez = e_LOS[2]

    # Slab intersection: for each axis, find the t-values at which the ray
    # enters and exits the cell slab [ centre - h, centre + h ].
    # Ray: p(t) = cell_centre + t * e_LOS  (t=0 at cell centre)
    # For axis i: t = ±h_i / |e_i|  (avoid division by zero with np.where)

    tiny = 1e-30   # prevents divide-by-zero for LOS perpendicular to an axis

    tx = hx / np.where(np.abs(ex) > tiny, np.abs(ex), tiny)
    ty = hy / np.where(np.abs(ey) > tiny, np.abs(ey), tiny)
    tz = hz / np.where(np.abs(ez) > tiny, np.abs(ez), tiny)

    # Entry = max of the three lower slab bounds
    # Exit  = min of the three upper slab bounds
    # Path length = exit - entry, clamped to >= 0
    t_enter = np.maximum.reduce([-tx, -ty, -tz])   # most restrictive entry
    t_exit  = np.minimum.reduce([ tx,  ty,  tz])   # most restrictive exit

    dl = np.maximum(t_exit - t_enter, 0.)   # cm; 0 if ray misses (shouldn't happen)

    col_den = den * dl   # mp/cm^2
    vgrid_proj = np.sum(np.array([vx,vy,vz])*e_LOS,axis=0)
    vsunvecarray = np.ones_like(e_LOS)
    vsunvecarray[ 0 ] = velxsun
    vsunvecarray[ 1 ] = velysun
    vsunvecarray[ 2 ] = velzsun
    vsun_proj  = np.sum(vsunvecarray*e_LOS,axis=0)
    vel_los = vgrid_proj-vsun_proj
    lv_l = longitude.ravel()
    lv_b = latitude.ravel()
    lv_v = vel_los.ravel()
    lv_d = den.ravel()
    lv_cden = col_den.ravel()
    if tempcut is not None:
        lv_cden[ temp > tempcut ] = 0

    #Plotting
    #######################################
    figsize_base = kwargs.get( "figsize_base", ( 6, 6 ) )
    width_ratios = kwargs.get( "width_ratios", [ 1, 0.03 ])
    height_ratios = kwargs.get( "height_ratios", [ 0.25, 1 ] )
    hspace = kwargs.get( "hspace", 0.05 )
    wspace = kwargs.get ( "wspace", 0.05 )
    vmin   = kwargs.get( "vmin", 1e19 )
    vmax   = kwargs.get( "vmax", 1e23 )
    sigma_lb = kwargs.get( "sigma_lb", ( 6, 6 ) )
    sigma_lv = kwargs.get( "sigma_lv", ( 6, 6 ) )
    fontsize = kwargs.get( "fontsize", 20 )
    savefile = kwargs.get( "savefile", None )
    fig = plt.figure(figsize=(figsize_base[ 0 ] * ( width_ratios[ 0 ] + width_ratios[ 1 ] ),
                              figsize_base[ 1 ] * ( height_ratios[ 0 ] + height_ratios[ 1 ] )  ))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            width_ratios=width_ratios,
                            height_ratios=height_ratios,
                            hspace=hspace, wspace=wspace)
    plt.rcParams[ "font.size" ] = fontsize
    ax1     = fig.add_subplot(gs[0, 0])
    ax2     = fig.add_subplot(gs[1, 0], sharex=ax1)
    cbar_ax = fig.add_subplot(gs[:, 1])   # spans both rows

    norm = LogNorm(vmin=vmin, vmax=vmax)


    h_lb, _, _ = np.histogram2d(lv_l, lv_b, weights=lv_cden, bins=[lbins, bbins])
    h_lv, _, _ = np.histogram2d(lv_l, lv_v, weights=lv_cden, bins=[lbins, vbins])

    h_lb_smooth = gaussian_filter(h_lb, sigma=sigma_lb)
    h_lv_smooth = gaussian_filter(h_lv, sigma=sigma_lv)

    h_lb_smooth[h_lb_smooth == 0] = np.nan
    h_lv_smooth[h_lv_smooth == 0] = np.nan

    # ── Top: l-b map ──────────────────────────────────────────────────────────────
    im1 = ax1.pcolormesh(lbins, bbins, h_lb_smooth.T,
                         cmap='turbo', norm=norm, shading='flat')
    ax1.set_ylabel(r'$b \; (^\circ)$', fontsize=fontsize )
    ax1.set_ylim(np.min( bbins ), np.max( bbins ) )
    ax1.set_xlim(np.max( lbins ), np.min( lbins ))
    ax1.tick_params(labelbottom=False)   # hide x tick labels on top panel

    # ── Bottom: l-v diagram ───────────────────────────────────────────────────────
    im2 = ax2.pcolormesh(lbins, vbins, h_lv_smooth.T,
                         cmap='turbo', norm=norm, shading='flat')
    ax2.set_ylabel(r'$v_\mathrm{los}$ (km/s)', fontsize=fontsize )
    ax2.set_xlabel(r'$\ell \; (^\circ)$', fontsize=fontsize )
    ax2.set_ylim(np.min( vbins ), np.max( vbins ) )
    ax2.set_xlim(np.max( lbins ), np.min( lbins ))

    # ── Shared colourbar ──────────────────────────────────────────────────────────
    fig.colorbar(im2, cax=cbar_ax, label=r'$m_p$ cm$^{-2}$')

    if savefile is not None:
        plt.savefig( savefile, bbox_inches='tight')
    return fig
#plt.show()
