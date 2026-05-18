"""
Reads Kratos output data into yt via mimicking the athena++ athdf format. May be deprecated in the
future once yt can properly handle Kratos binary outputs

Current crude workaround involves the creation and deletion of an .athdf file. 
"""
import numpy as np
import h5py
import yt
import sys
import os
import io
import tempfile
sys.path.append( os.getenv("KRATOS_VISUAL_DIR") )
sys.path.append(os.path.join(os.path.dirname(__file__), "../base"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../init"))
from hydro_data import enroll_mesh_tree
from unit import units

def load_kratos_yt( d, gamma=5./3., fld_extra = [], **kwargs ):
    if "units_override" not in kwargs:
        kwargs["units_override"] = {
                                      "length_unit"    : ( units.l0, "cm" ),
                                      "time_unit"      : ( units.t0, "s"  ),
                                      "mass_unit"      : ( units.m0, "g"  ),
                                    }
        
    enroll_mesh_tree( d )
    buf = kratos_to_hdf5( d, fld_extra = fld_extra, gamma=gamma )
    with tempfile.NamedTemporaryFile( suffix=".athdf", delete=True ) as tmp:
        tmp.write( buf.getvalue() )
        tmp.flush()
        ds = yt.load( tmp.name, **kwargs )
    #ds = yt.load( "kratos_test.athdf", **kwargs )
    #if delete_file:
    #    os.remove(f"{filename}.athdf")
    return ds

def kratos_to_hdf5( d, fld_extra = [], gamma=5./3. ):
    buf = io.BytesIO()
    file = h5py.File( buf, "w" )
    #file = h5py.File( f"{filename}.athdf", "w" )
    levels = []
    logic  = []
    xf     = []
    xc     = []
    cons   = []
    names_base = np.array( [ b'dens', b'Etot', b'mom1', b'mom2', b'mom3', b'entropy', b'metallicity', b'pressure' ], dtype=np.bytes_ )
    var_names_extra = np.array( [ i for i in fld_extra ], dtype=np.bytes_ )
    names = np.concatenate( ( names_base, var_names_extra ) ) 
    for key, bd in d.data.items( ):
        if "block" not in key:
            continue
        levels.append( bd[ 'level' ] )
        logic.append( bd[ 'i_logic' ])
        xf.append( bd[ 'x_f' ] )
        xc.append( bd[ 'x_c' ] )
        con_arr = [ bd[ 'rho' ]     , bd[ 'ene' ]     , bd[ 'mom' ][ 0 ],
                    bd[ 'mom' ][ 1 ], bd[ 'mom' ][ 2 ], bd[ 'ent' ]     ,
                    bd[ 'met' ]     , bd[ 'pre_ent'] ]
        for name in fld_extra:
            con_arr.append( bd[ name ] )
            
        cons.append( np.array( con_arr ) )

    levels, logic, xf, xc, cons = np.array( levels ), np.array( logic ), np.array( xf ), np.array( xc ), np.array( cons )
    cons = cons.transpose( 1, 0, 2, 3, 4 )
    max_level = np.max( levels )
    file.attrs[ 'Coordinates' ]   = np.bytes_( b'cartesian' )
    file.attrs[ 'DatasetNames' ]  = np.array( [ b'cons' ], dtype=np.bytes_ )
    file.attrs[ 'MaxLevel' ]       = np.int32( max_level )
    file.attrs[ 'MeshBlockSize' ] = d.data[ 'block_0' ][ 'n_cell' ]
    file.attrs[ 'NumMeshBlocks' ] = len( levels )
    file.attrs[ 'NumVariables' ]  = np.array( [ len( cons ) ] )
    file.attrs[ 'RootGridSize' ]  = d.n_base
    for i in range( 3 ):
        file.attrs[ f'RootGridX{ i + 1 }' ] = np.concatenate( ( d.x_lim[ i ], [ 1 ] ) )
    file.attrs[ 'Time' ]          = d.globals[ 'time' ]
    file.attrs[ 'VariableNames' ] = names
    file.attrs[ 'Gamma' ] = gamma
    file.create_dataset( "Levels", data=levels )
    file.create_dataset( "LogicalLocations", data=logic )
    for i in range( 3 ):
        file.create_dataset( f"x{ i + 1 }v", data=xc[ :, i, : ] )
        file.create_dataset( f"x{ i + 1 }f", data=xf[ :, i, : ] )
    file.create_dataset( "cons", data=cons )
    file.close()
    buf.seek( 0 )
    return buf