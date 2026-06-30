import sys
import os
from numpy import concatenate, array, zeros, cbrt, copy
import numpy as np
sys.path.append( os.getenv("KRATOS_VISUAL_DIR") )
sys.path.append( "../base" )
sys.path.append( "../init" )
from unit import units
import hydro_data
from hydro_data import enroll_mesh_tree
from slice_plot import slice, recognize_vec
import configparser
from scipy.interpolate import interp1d

class galaxy_data( hydro_data.hydro_data ):
    def tst_field( self, blk, fld, datfld, dtype = 'f' ):
        """
        Try to enroll *fld* into *datfld*; silently skip if the field
        does not exist in the binary output.
        """
        try:
            self.get_field( blk, fld, datfld,
                            dtype = dtype );
        except:
            #print( "Unable to enroll: ", fld );
            pass;
        #
    #
    def read_block_data( self, blk ):
        """
        Read hydro conserved variables for a mesh block and compute
        derived quantities  (entropy, thermal pressure, metallicity).
        """
        dat = self.data[ blk ];
        if  not 'n_cell' in dat:
            self.read_block_geometry( blk );
        #
        bd = self.get_field( blk, 'hydro_cons' );
        dat[ 'rho' ] = bd[ 0 ];
        dat[ 'ene' ] = bd[ 1 ];
        dat[ 'mom' ] = bd[ 2 : 5 ];
        try:
            dat[ 'met' ] = bd[ 6 ] / dat[ 'rho' ];
            dat[ 'met_fld' ] = bd[ 6 ]
        except:
            pass
        try:
            dat[ 'sne_fld' ] = bd[ 7 ]
        except:
            pass
        dat[ 'ent' ] = bd[ 5 ];
        gam1 = self.args( 'dynamics', 'gamma' ) - 1;      
        dat[ 'pre_ent' ] = bd[ 5 ] * dat[ 'rho' ]**gam1;
        self.gam1 = gam1

        # DEBUG - static field output currently not working
        #self.tst_field( blk, 'mg_s_',     'fld_s' );
        #self.tst_field( blk, 'mg_s_src', 'src_s' );

        
        self.tst_field( blk, 'mg',       'fld_d' );
        self.tst_field( blk, 'mg_src',   'src_d' );
        self.tst_field( blk, 'cic_acc',  'acc'   );
        self.tst_field( blk, 'cic_pidx', 'pidx',
                        dtype = 'i' );
        return;

    def load_particles( self ):
        """
        Load stellar / sink particle data from the binary output into
        ``self.data`` under ``particle_*`` keys.
        """
        self.bin_data.open(    );

        names = { "x"    : { "name":"x", "dim":3 } , 
                  "x0"    : { "name":"x0", "dim":3 } , 
                  "prop" : { "name":"m", "dim":1 } ,
                  "mstar": { "name":"mstar", "dim":1 },
                  "sfr"  : { "name":"sfr", "dim":1 },
                  "age"  : { "name":"age", "dim":1 },
                  "Nsne" : { "name":"Nsne","dim":1, "dtype":"i" },
                  "tcreate" : { "name":"tcreate", "dim":1 },
                  "tcreatemrg" : { "name":"tcreatemrg", "dim":1 },
                  "SFdelay" : { "name":"SFdelay", "dim":1 },
                  "Rs0" : { "name":"RS0", "dim":1 },
                  "logQ0" : { "name":"logQ0", "dim":1 },
                  "dir" : { "name":"v", "dim":3 } }

        for bin_name, dat_name in names.items( ):
            arr = [ ]
            skip = False
            for k in self.bin_data.hmap:
                if not ( 'par_rank_' in k and ( '_x' in k and not '_x0' in k ) ):
                    continue;
                try:
                    tst = self.bin_data.as_array( k.replace( 'x', bin_name ) )
                except:
                    print(f"{bin_name} not found")
                    skip = True
                    break; 
                if "dtype" in dat_name:
                    bin_data_arr = self.bin_data.as_array( k.replace( 'x', bin_name ), dtype=dat_name['dtype'] )
                else:
                    bin_data_arr = self.bin_data.as_array( k.replace( 'x', bin_name ) )

                if dat_name[ 'dim' ] != 1:
                    bin_data_arr = bin_data_arr.reshape( -1, dat_name[ 'dim' ] )
                arr.append( bin_data_arr)
                    
            if skip:
                continue
            self.data[ f'particle_{ dat_name[ "name" ] }' ] = concatenate( arr );

def stitch_fields( d, fields ):
    """
    Stitch mesh-block data onto a uniform Cartesian grid at the coarsest
    (root) level for the given *fields*.

    Parameters
    ----------
    d : hydro_data
    fields : list of str
        Field names to stitch.

    Returns
    -------
    res : dict of {str: ndarray}
        Stitched 3D arrays keyed by field name.
    l : tuple
        Shape of the stitched grid.
    """
    db0   = d.data[ 'block_0' ];
    dx0   = array( [ x[ -1 ] - x[ 0 ] for \
                     x in db0[ 'x_f' ] ] );
    l0    = array( db0[ 'rho' ].shape );
    l     = int( cbrt( len( d.data ) ) ) * l0;
    x_min = d.x_lim.T[ 0 ];
    res = { f : zeros( l ) for f in fields };
    for b, db in d.data.items(  ):
        if 'particle' in b:
            continue;
        x_min_s = [ x[ 0 ] for x in db[ 'x_f' ] ];
        dl = [ int( ( x_min_s[ a ] - x_min[ a ] \
                      + dx0[ a ] * 0.1 ) / dx0[ a ] ) \
               for a in range( 3 ) ];
        for f in fields:
            res[ f ]\
        [ dl[ 2 ] * l0[ 2 ] : l0[ 2 ] * ( dl[ 2 ] + 1 ),
          dl[ 1 ] * l0[ 1 ] : l0[ 1 ] * ( dl[ 1 ] + 1 ),
          dl[ 0 ] * l0[ 0 ] : l0[ 0 ] * ( dl[ 0 ] + 1 ) ] \
          += db[ f ];
    return res, l;

def refined_stitch(d, field=None ):
    """
    Stitch mesh-block data onto a uniform Cartesian grid at the finest
    refinement level.

    Parameters
    ----------
    d : hydro_data
    field : str or None
        Single field to stitch. If None, stitches the five primitive
        hydro fields (rho, mom[0-2], ene).

    Returns
    -------
    dict
        ``{field: ndarray}`` plus ``"x_c"`` coordinate arrays.
    """
    enroll_mesh_tree( d )
    logical_locations = array(list(d.tree.keys()))
    blocks = array(list(d.tree.values()))
    min_refine = np.min(logical_locations[:,-1])
    max_refine = np.max(logical_locations[:,-1])
    logic_max_refine = logical_locations[logical_locations[:,-1] == max_refine]
    cell_tot = 0

    for j in logic_max_refine:
        cell_tot += np.prod(d.data[d.tree[tuple(j)]]['rho'].shape)

    refine_shape = ( int(np.cbrt( cell_tot )),
                     int(np.cbrt( cell_tot )),
                     int(np.cbrt( cell_tot )) )

    root_dx = d.data[blocks[logical_locations[:,-1] == min_refine][0]]['dx0'][0]
    corner_logic = np.min(logic_max_refine, axis=0)[0:-1]
    stitched_data = {}
    if field is not None:
        fullcube = zeros( refine_shape )
        coord_cube = np.array( [ zeros( refine_shape ),
                             zeros( refine_shape ),
                             zeros( refine_shape )] )  
        for b, bd in d.data.items(  ):
            dx = bd['dx0'][0]
            if dx > root_dx / 2**(max_refine - min_refine):
                continue
            vflag = recognize_vec( field );
            if  vflag is not None:
                key, i = vflag;
                d_v = copy( bd[ key ][ i ] );
            elif field in bd:
                d_v = copy( bd[ field    ] );
            else:
                raise ValueError( "%s is undefined" % field );
            bd_logic = bd['i_logic'] - corner_logic
            corner = bd[ 'n_cell' ] * bd_logic
            fullcube[corner[0]:corner[0] + bd[ 'n_cell' ][0],\
                     corner[1]:corner[1] + bd[ 'n_cell' ][1],\
                     corner[2]:corner[2] + bd[ 'n_cell' ][2]]\
                   = d_v.transpose(2,1,0)
            coords = np.meshgrid(bd['x_c'][0],bd['x_c'][1],bd['x_c'][2], indexing='ij')
            for j in range(3):
                coord_cube[j][corner[0]:corner[0] + bd[ 'n_cell' ][0],\
                     corner[1]:corner[1] + bd[ 'n_cell' ][1],\
                     corner[2]:corner[2] + bd[ 'n_cell' ][2]]\
                   = coords[j]#d_v.transpose(2,1,0)
        #coord_cube = coord_cube.transpose(0,2,1,3)
        fullcube = fullcube.transpose(1,0,2)
        stitched_data[field] = fullcube
        stitched_data["x_c"] = [coord_cube[0,:,0,0],
                                coord_cube[1,0,:,0],
                                coord_cube[2,0,0,:]]
    else:
        for fi, field in enumerate(["rho", "mom[0]", "mom[1]", "mom[2]", "ene"]):
            fullcube = zeros( refine_shape )
            coord_cube = np.array( [ zeros( refine_shape ),
                                 zeros( refine_shape ),
                                 zeros( refine_shape )] )  
            for b, bd in d.data.items(  ):
                dx = bd['dx0'][0]
                if dx > root_dx / 2**(max_refine - min_refine):
                    continue
                vflag = recognize_vec( field );
                if  vflag is not None:
                    key, i = vflag;
                    d_v = copy( bd[ key ][ i ] );
                elif field in bd:
                    d_v = copy( bd[ field    ] );
                else:
                    raise ValueError( "%s is undefined" % field );
                bd_logic = bd['i_logic'] - corner_logic
                corner = bd[ 'n_cell' ] * bd_logic
                fullcube[corner[0]:corner[0] + bd[ 'n_cell' ][0],\
                         corner[1]:corner[1] + bd[ 'n_cell' ][1],\
                         corner[2]:corner[2] + bd[ 'n_cell' ][2]]\
                       = d_v.transpose(2,1,0)
                coords = np.meshgrid(bd['x_c'][0],bd['x_c'][1],bd['x_c'][2], indexing='ij')
                for j in range(3):
                    coord_cube[j][corner[0]:corner[0] + bd[ 'n_cell' ][0],\
                         corner[1]:corner[1] + bd[ 'n_cell' ][1],\
                         corner[2]:corner[2] + bd[ 'n_cell' ][2]]\
                       = coords[j]#d_v.transpose(2,1,0)
            #coord_cube = coord_cube.transpose(0,2,1,3)
            fullcube = fullcube.transpose(1,0,2)
            stitched_data[field] = fullcube
        stitched_data["x_c"] = [coord_cube[0,:,0,0],
                                coord_cube[1,0,:,0],
                                coord_cube[2,0,0,:]]
        stitched_data["mom"] = [stitched_data["mom[0]"],
                                stitched_data["mom[1]"],
                                stitched_data["mom[2]"]]
        for j in range(3):
            del stitched_data[f"mom[{j}]"]
    
    return stitched_data

def enroll_mu( d, cool_file="../outputs/therm_tables/cooling_table.dat" ):
    """
    Enroll a mean molecular weight field ``mu`` from a cooling table.

    Parameters
    ----------
    d : hydro_data
    cool_file : str
        Path to the cooling table config file.
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read( cool_file )
    lg_e      = np.array( config.get( "cooling", "lg_e" ).split( " " ), dtype=float )
    lg_mu_cgs = np.array( config.get( "cooling", "lg_mu_cgs" ).split( " " ), dtype=float )
    mu   = 10**lg_mu_cgs / units.mp
    f = interp1d( lg_e, mu, bounds_error=False, fill_value='extrapolate')
    d.enroll_field( "mu", lambda bd: \
                    f( np.log10( bd[ 'pre_ent' ] / d.gam1 * units.v0**2 ) ) )
    return

def enroll_sgn( d, fld, sgnfld='x' ):
    """
    Enroll signed components of a vector field: ``sgn{fld}[i] = fld[i] * sign(sgnfld[i])``.
    """
    d.enroll_field( f'sgn{fld}', lambda bd : \
                       np.array( [ ( bd[ fld ][ i ] if bd[ fld ].ndim == bd[ sgnfld ].ndim else bd[ fld ] ) *
                                     np.sign( bd[ sgnfld ][ i ] )
                                     for i in range( len( bd[ sgnfld ] ) ) ] ) )

def enroll_T( d, **kwargs ):
    """
    Enroll temperature fields ``T`` and ``T_ent``.

    Supports constant mean molecular weight or tabulated ``mu`` from
    a cooling table.

    Parameters
    ----------
    d : hydro_data
    mu_default : float
        Constant mu value when ``use_const_mu=True``.
    cool_file : str or None
        Path to cooling table; ``None`` uses default.
    use_const_mu : bool
        If True, skip mu enrolment and use *mu_default*.
    """
    mu_default = kwargs.get( "mu_default", 1.26 )
    cool_file  = kwargs.get( "cool_file", None )
    use_const_mu = kwargs.get( "use_const_mu", False )
    t0     = d.args( "unit",    "time" );
    l0     = d.args( "unit",  "length" );
    rho0   = d.args( "unit", "density" );
    T_conv   = ( l0 / t0 )**2 * units.mp / units.kb# * mu;
    if use_const_mu:
        T_conv *= mu_default
        d.enroll_field( 'T', lambda bd : \
            bd[ 'pre' ] / bd[ 'rho' ] * T_conv );
        d.enroll_field( 'T_ent', lambda bd : \
            bd[ 'pre_ent' ] / bd[ 'rho' ] * T_conv );
        return
    else:
        if cool_file is None:
            enroll_mu( d )
        else:
            enroll_mu( d, cool_file )
        d.enroll_field( 'T', lambda bd : \
            bd[ 'pre' ] / bd[ 'rho' ] * T_conv * bd[ 'mu' ] );
        d.enroll_field( 'T_ent', lambda bd : \
            bd[ 'pre_ent' ] / bd[ 'rho' ] * T_conv * bd[ 'mu' ] );        

def enroll_vel( d ):
    """
    Enroll the velocity field ``vel = mom / rho`` and its signed components.
    """
    t0     = d.args( "unit",    "time" );
    l0     = d.args( "unit",  "length" );
    rho0   = d.args( "unit", "density" ); 
    d.enroll_field( f'vel', lambda bd : \
                       np.array( [ bd[ 'mom' ][ i ] / bd[ 'rho' ] 
                                   for i in range( len( bd[ 'mom' ] ) ) ] ) )
    enroll_sgn( d, 'vel' )

def enroll_mach_ent( d ):
    """
    Enroll the entropic Mach number minus one: ``mach_m1_ent = |v| / c_s_ent - 1``.
    """
    enroll_vel( d )
    d.enroll_field( "mach_m1_ent", lambda bd: \
     bd[ "v_mag" ]/ \
     np.sqrt( ( bd[ 'gam1' ] + 1 ) * ( bd[ 'pre_ent' ] / bd[ 'rho' ] ) ) )
    return

def enroll_vphi( d ):
    """
    Enroll the azimuthal (cylindrical) velocity component.

    v_phi = (v_y x - v_x y) / r  =  (mom_y x - mom_x y) / (rho r)

    r = sqrt( x**2 + y**2 )  is computed per-cell using the cell-centre
    coordinate arrays ``x_c[0]`` and ``x_c[1]``.
    """
    def _vphi( bd ):
        Xb, Yb, Zb = np.meshgrid( bd['x_c'][0], 
                                  bd['x_c'][1], 
                                  bd['x_c'][2], indexing='ij' )
        R = np.sqrt( Xb**2 + Yb**2 )
        Phi = np.arctan2( Yb, Xb )
        vphi = bd[ 'vel' ][ 0 ] * -np.sin( Phi ) + bd[ 'vel' ][ 1 ] * np.cos( Phi )
        return vphi #/ bd[ 'rho' ]
    
    d.enroll_field( 'v_phi', _vphi )


def enroll_flux( d, fld, func=None ):
    """
    Enroll a mass-flux field for `fld` (typically density or a passive scalar).
    
    The flux in direction i is:
        F_i = vel_i * fld * A_i
    where A_i = product of dx in all directions j != i (the transverse face area),
    evaluated at the cell's actual refinement level (using dx, not dx0).
    
    All quantities are normalized by the code flux unit:
        flux_unit = rho0 * l0^2 * (l0 / t0)
    """
    t0   = d.args( "unit",   "time" )
    l0   = d.args( "unit", "length" )
    rho0 = d.args( "unit","density" )

    flux_unit = rho0 * l0**2 * ( l0 / t0 )   # [density * area * velocity]

    enroll_vel( d )

    if func is not None:
        d.enroll_field( fld, func )

    def flux_func( bd ):
        vel  = bd[ 'vel' ]                    # shape (ndim, ...) — cell-centered
        dx   = bd[ 'dx0'  ]                    # shape (ndim,)     — THIS level's dx
        phi  = bd[ fld   ]                    # scalar or vector field

        ndim = len( vel )

        fluxes = []
        for i in range( ndim ):
            # Face area in direction i = product of dx in all OTHER directions
            A_i   = np.prod( [ dx[j] for j in range( ndim ) if j != i ] )

            # Select the i-th component if phi is a vector, else use scalar
            phi_i = phi[i] if phi.ndim == vel.ndim else phi

            fluxes.append( vel[i] * phi_i * A_i )#/ flux_unit )

        return np.array( fluxes )   # shape (ndim, ...)

    d.enroll_field( f'flux_{fld}', flux_func )

    # Enroll signed flux along each axis separately so inflow/outflow
    # splitting respects the direction of each component independently
    enroll_sgn( d, f'flux_{fld}' )
def map_particle_blk( d, *pflds ): 
    """
    Deposit particle quantities onto mesh blocks using exact sub-cell
    overlap geometry (line-of-sight integration over particle prisms).

    Parameters
    ----------
    d : hydro_data
        Must have ``particle_x`` loaded.
    *pflds : str
        Particle field names to deposit (e.g. ``'particle_sfr'``).
    """
    enroll_mesh_tree( d )
    from itertools import product
    pos     = d.data[ 'particle_x' ]

    blk_map = {}
    for blk, dat in d.data.items(  ):
        if  not 'block' in blk:
            continue;
        
        blk_map[ str( np.append( dat['i_logic'], dat['level'] ) ) ] = blk
        for k in pflds:
            d.data[ blk ][ k ] = np.zeros( dat[ 'rho' ].shape )
    
    for blk, dat in d.data.items(  ):
        if not 'block' in blk:
            continue
        
        lower = np.array( [ b[  0 ]  for b in dat[ 'x_f' ] ] )
        upper = np.array( [ b[ -1 ]  for b in dat[ 'x_f' ] ] )
        inds  = np.all( ( pos >= lower ) & ( pos < upper ), axis=1 )
        if not inds.any( ):
            continue
        par_fields = [ d.data[ pfld ][ inds ] for pfld in pflds ]
        par_fields.insert( 0, pos[ inds ] )
        for par_dats in zip( *par_fields ):
            p_x = par_dats[ 0 ]
            p_ind = tuple( np.searchsorted( dat[ 'x_f' ][ i ], p_x[ i ], side='right' ) - 1 for i in range( 3 ) )
            quad = tuple( [ 1 if p_x[ i ] >= dat[ 'x_c' ][ i ][ p_ind[ i ] ] else -1 for i in range( 3 ) ] )
            powset = list( product( *[ ( 0, q ) for q in quad ]  ) )
            for c in powset:
                p_ind_n = tuple( np.array( p_ind ) + np.array( c ) )

                prism_range = [ ]
                out_of_box = [ ]
                for i, j in enumerate( c ):
                    out_of_box.append( p_x[ i ] - dat[ 'dx0' ][ i ] / 2 < d.x_lim[ i ][ 0 ] or \
                                       p_x[ i ] + dat[ 'dx0' ][ i ] / 2 >= d.x_lim[ i ][ 1 ]  )
                    if j == -1:
                        prism_range.append( [ p_x[ i ] - dat[ 'dx0' ][ i ] / 2, dat[ 'x_f' ][ i ][ p_ind_n[ i ] + 1 ]  ] )
                    elif j == 1:
                        prism_range.append( [ dat[ 'x_f' ][ i ][ p_ind_n[ i ] ], p_x[ i ] + dat[ 'dx0' ][ i ] / 2 ] )
                    else:
                        if quad[ i ] == 1:
                            prism_range.append( [ p_x[ i ] - dat[ 'dx0' ][ i ] / 2, dat[ 'x_f' ][ i ][ p_ind_n[ i ] + 1 ]  ] )
                        elif quad[ i ] == -1:
                            prism_range.append( [ dat[ 'x_f' ][ i ][ p_ind_n[ i ] ], p_x[ i ] + dat[ 'dx0' ][ i ] / 2 ] )

                if np.array( out_of_box ).any( ):
                    continue

                prism_range = np.array( prism_range )

                weight = np.prod( np.diff( prism_range, axis=1 ).flatten( ) ) / np.prod( dat[ 'dx0' ] )
                
                #DEBUG
                if weight < 0:
                    print( weight, prism_range, c )
                    raise ArithmeticError( "weight" )
                
                oob    = np.array( [ 1 if i >= s else -1 if i < 0 else 0 for i, s in zip( p_ind_n, dat[ 'n_cell' ] ) ] )
                if np.all( oob == 0 ):
                    for k, p_d in zip( pflds, par_dats[ 1: ] ):
                        d.data[ blk ][ k ][ tuple( p_ind_n[ ::-1 ] ) ]  += p_d * weight
                

                else:
                    oobpos = np.array( [ ( i[ 1 ] + i[ 0 ] ) / 2 for i in prism_range ] )
                    blk_n = blk_map[ str( d.find_blk( oobpos )[ 0 ] ) ]
                    dat_n = d.data[ blk_n ]
                    slices = []
                    for i, bounds in enumerate( prism_range ):
                        lo = np.searchsorted( dat_n[ 'x_f' ][ i ], bounds[ 0 ], side="right" ) - 1
                        hi = np.searchsorted( dat_n[ 'x_f' ][ i ], bounds[ 1 ], side="right" ) - 1
                        slices.append( np.linspace( lo, hi, 1, dtype='int' ) )
                    i_idx, j_idx, k_idx = np.meshgrid( *slices, indexing='ij' )
                    overinds = np.array( [ i_idx.flatten(), j_idx.flatten( ), k_idx.flatten() ] ).T
                    volumes = [ ]
                    for m, idx in enumerate( overinds ):
                        vol = 1 / np.prod( dat_n[ 'dx0' ] )
                        for dim in range( 3 ):
                            left = max( dat_n[ 'x_f' ][ dim ][ idx[ dim ] ], prism_range[ dim ][ 0 ] )
                            right = min( dat_n[ 'x_f' ][ dim ][ idx[ dim ] + 1 ], prism_range[ dim ][ 1 ] )
                            vol *= ( right - left )
                        if vol < 0:
                            print( vol, left, right, idx, prism_range, dat_n[ 'x_f' ], oobpos )
                            raise ArithmeticError( "vol" )
                        volumes.append( vol )
                    volumes = np.array( volumes )
                    volumes /= np.sum( volumes )
                    for idx_n, wgt_n in zip( overinds, volumes ):
                        for k, p_d in zip( pflds, par_dats[ 1: ] ):
                            d.data[ blk_n ][ k ][ tuple( idx_n[ ::-1 ] ) ] += p_d * wgt_n * weight
    
    for blk, dat in d.data.items(  ):
        if not 'block' in blk:
            continue
        for k in pflds:
            d.data[ blk ][ k ] /= np.prod( dat[ 'dx' ], axis=0 )
    return

def par_pos_cut( d, x1range, x2range, x3range=None, coord='cyl' ):
    """
    simple way to isolate particles in certain regions
    cart : x, y, z
    cyl  : R, z, phi
    sph  : R, theta, phi
    """
    x, y, z = d.data[ 'particle_x' ].T
    if not isinstance( x1range, ( list, tuple, np.ndarray ) ):
        x1range = [ 0, x1range ]
    if not isinstance( x2range, ( list, tuple, np.ndarray ) ):
        x2range = [ 0, x2range ]

    if coord == 'cart':
        x1 = x
        x2 = y
        x3 = z
    elif coord == 'cyl':
        x1 = np.sqrt( x**2 + y**2 )
        x2 = z
        x3 = np.arctan2( y, x )
    elif coord == 'sph':
        x1 = np.sqrt( x**2 + y**2 + z**2 )
        x2 = np.arccos( z / x1 )
        x3 = np.arctan2( y, x )
    filter = np.logical_and( np.logical_and( x1 >= x1range[ 0 ], x1 <= x1range[ 1 ] ),
                             np.logical_and( x2 >= x2range[ 0 ], x2 <= x1range[ 1 ] ) )
    if x3range is not None:
        if not isinstance( x3range, ( list, tuple, np.ndarray ) ):
            x3range = [ 0, x3range ]
        filter = np.logical_and( filter,
                                 np.logical_and( x3 >= x3range[ 0 ], x2 <= x3range[ 1 ] ) )
    return filter
