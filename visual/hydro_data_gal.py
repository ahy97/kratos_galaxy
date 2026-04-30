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


class galaxy_data( hydro_data.hydro_data ):
    def tst_field( self, blk, fld, datfld, dtype = 'f' ):
        try:
            self.get_field( blk, fld, datfld,
                            dtype = dtype );
        except:
            #print( "Unable to enroll: ", fld );
            pass;
        #
    #
    def read_block_data( self, blk ):    
        dat = self.data[ blk ];
        if  not 'n_cell' in dat:
            self.read_block_geometry( blk );
        #
        bd = self.get_field( blk, 'hydro_cons' );
        dat[ 'rho' ] = bd[ 0 ];
        dat[ 'ene' ] = bd[ 1 ];
        dat[ 'mom' ] = bd[ 2 : 5 ];
        dat[ 'met' ] = bd[ 6 ] / dat[ 'rho' ];
        dat[ 'met_fld' ] = bd[ 6 ]
        try:
            dat[ 'sne_fld' ] = bd[ 7 ]
        except:
            pass
        dat[ 'ent' ] = bd[ 5 ];
        gam1 = self.args( 'dynamics', 'gamma' ) - 1;      
        dat[ 'pre_ent' ] = bd[ 5 ] * dat[ 'rho' ]**gam1;


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
        self.bin_data.open(    );
        x_arr = [  ];
        v_arr = [  ];
        m_arr = [  ];
        mstar_arr = [ ];
        sfr_arr = [  ];
        age_arr = [  ];
        Nsne_arr = [  ];
        tcreate_arr = []
        SF_delay_arr = [ ]
        R_S0_arr = []
        log_Q0_arr = []
        for k in self.bin_data.hmap:
            if not ( 'par_rank_' in k and '_x' in k ):
                continue;
            x_arr.append( self.bin_data.as_array\
                          ( k ).reshape( -1, 3 ) );
            m_arr.append( self.bin_data.as_array\
             ( k.replace( 'x', 'prop' ) ) );
            mstar_arr.append( self.bin_data.as_array\
             ( k.replace( 'x', 'mstar' ) ) )
            sfr_arr.append( self.bin_data.as_array
                          ( k .replace( 'x', 'sfr' ) ) )
            age_arr.append( self.bin_data.as_array 
                          ( k .replace( 'x', 'age' ) ) )
            Nsne_arr.append( self.bin_data.as_array 
                          ( k .replace( 'x', 'Nsne' ), dtype='i' ) )
            tcreate_arr.append( self.bin_data.as_array\
             ( k.replace( 'x', 'tcreate' ) ) )
            SF_delay_arr.append( self.bin_data.as_array\
             ( k.replace( 'x', 'SFdelay' ) ) )
            R_S0_arr.append( self.bin_data.as_array\
             ( k.replace( 'x', 'Rs0' ) ) )
            log_Q0_arr.append( self.bin_data.as_array\
             ( k.replace( 'x', 'logQ0' ) ) )
            v_arr.append( self.bin_data.as_array\
             ( k.replace( 'x',  'dir' ) ).reshape( -1, 3 ) )
        #ss
        self.data[ 'particle_x' ] = concatenate( x_arr );
        self.data[ 'particle_v' ] = concatenate( v_arr );
        self.data[ 'particle_m' ] = concatenate( m_arr );
        self.data[ 'particle_mstar' ] = concatenate( mstar_arr );
        self.data[ 'particle_sfr' ] = concatenate( sfr_arr );
        self.data[ 'particle_age' ] = concatenate( age_arr );
        self.data[ 'particle_Nsne' ] = concatenate( Nsne_arr );
        self.data[ 'particle_tcreate' ] = concatenate( tcreate_arr );
        self.data[ 'particle_SFdelay' ] = concatenate( SF_delay_arr );
        self.data[ 'particle_RS0' ] = concatenate( R_S0_arr );
        self.data[ 'particle_logQ0' ] = concatenate( log_Q0_arr );
    #
def stitch_fields( d, fields ):
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

def enroll_sgn( d, fld, sgnfld='x' ):
    d.enroll_field( f'sgn{fld}', lambda bd : \
                       np.array( [ ( bd[ fld ][ i ] if bd[ fld ].ndim == bd[ sgnfld ].ndim else bd[ fld ] ) *
                                     np.sign( bd[ sgnfld ][ i ] )
                                     for i in range( len( bd[ sgnfld ] ) ) ] ) )

def enroll_T( d, mu=1.26 ):
    t0     = d.args( "unit",    "time" );
    l0     = d.args( "unit",  "length" );
    rho0   = d.args( "unit", "density" ); 
    T_conv   = ( l0 / t0 )**2 * mu * units.mp / units.kb;
    d.enroll_field( 'T', lambda bd : \
        bd[ 'pre' ] / bd[ 'rho' ] * T_conv );
    d.enroll_field( 'T_ent', lambda bd : \
        bd[ 'pre_ent' ] / bd[ 'rho' ] * T_conv );

def enroll_vel( d ):
    t0     = d.args( "unit",    "time" );
    l0     = d.args( "unit",  "length" );
    rho0   = d.args( "unit", "density" ); 
    d.enroll_field( f'vel', lambda bd : \
                       np.array( [ bd[ 'mom' ][ i ] / bd[ 'rho' ][ i ] 
                                   for i in range( len( bd[ 'mom' ] ) ) ] ) )
    enroll_sgn( d, 'vel' )
    
def enroll_flux( d, fld, func=None ):
    t0     = d.args( "unit",    "time" );
    l0     = d.args( "unit",  "length" );
    rho0   = d.args( "unit", "density" ); 
    enroll_vel( d )
    if func is not None:
        d.enroll_field( fld, func )
    d.enroll_field( f'flux_{fld}', lambda bd : \
                               np.array([ bd[ 'vel' ][ i ] / bd[ 'dx' ][ i ]**2 *
                                        ( bd[ fld ] if bd[ fld ].ndim < bd[ 'vel' ].ndim else bd[ fld ][ i ] )
                                          for i in range( len( bd[ 'vel' ] ) ) ] ) )
    enroll_sgn( d, f'flux_{fld}' )

def map_particle_blk( d, *pflds ): 
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