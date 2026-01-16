from base.unit import units
from base.data_field import data_field
import numpy as np
from scipy.interpolate import LinearNDInterpolator

class grid( units ):
    """
    A basic grid object that wraps various array and mesh functionalities together. A grid can be initialized
    via a config file, from preset 1D arrays along each dimensions, or from keyword arguments. 

    Three representations of a grid are stored:
    
        grid  : A list of 1D arrays representing the gridpoints along each dimension. 
        mesh  : A list of ND arrays representing the coordinates of each gridpoint in ND space
        coords: A list of length N arrays representing the coordinates of every single point

    Additional functionalities include coordinate transformations between cartesian, spherical and cylindrical
    (which can be applied to data_field objects stored in child objects), and data interpolation between different
    grid objects with different gridpoints and/or coordinate systems.
    """
    def __init__( self, *args, **kwargs ):
        """
        Initializes an instance of a grid object. All three representations (grid, mesh, coords) are initialized
        
        :param args: N array-like arguments in the form arr1, arr2... forming the grid 
        :param \**kwargs: See below

        :Keyword Arguments:
            **coord       : Coordinate system of the grid - must be "cart" (cartesian), "cyl" (cylindrical) 
                            or "sph" (spherical). Note that for "cyl" and "sph", the radial coordinate is 
                            the first and the azimuthal the third
            **spacing     : Length N list/array representing the ratios for logarithmic spacing along each 
                            dimension. 1 means uniform spacing
            **x_min       : Length N list/array representing the lower limit along each dimension
            **x_min       : Length N list/array representing the upper limit along each dimension
            **n_cell      : Length N list/array representing the number of gridpoints along each dimension
            **n_gh        : Length N list/array representing the number of additional ghost cells tacked onto the ends
            **config_flag : string denoting the section in the config file where the above keyword arguments are
                            specified. 
        """
        if len( args ) > 0 :
            self.n_gh     = np.array( [ 0, 0 ] )
            self.n_cell   = np.array( [ len( i ) for i in args ] )
            self.coord    = kwargs.get( "coord", "sph" )
            self.cmcoord  = self.coord
            self.spacing  = np.array( [ ( i[ 2 ] - i[ 1 ] ) / ( i[ 1 ] - i[ 0 ] ) for i in args ] )
            self.x_range  = self.range_to_data_field( np.array( [ [ np.min( i ), np.max( i ) ] for i in args ] ) )
            self.dim_check( )
            self.grid     = list( args )
        elif "config_flag" in kwargs:
            self.config_init( kwargs[ "config_flag" ] )
        else:
            self.coord    = kwargs.get( "coord", "sph" )
            self.cmcoord  = self.coord
            self.spacing  = np.array( kwargs.get( "spacing", [ 1, 1 ] ) )
            xmin          = np.array( kwargs.get( "x_min"  , [ 0, 0 ] ) )
            xmax          = np.array( kwargs.get( "x_max"  , [ 1, 1 ] ) )
            x_range  = np.array( [ xmin, xmax ] ).transpose( )
            self.n_cell   = np.array( kwargs.get( "n_cell" , [ 100, 100 ] ) )
            self.n_gh     = np.array( kwargs.get( "n_gh"   , [ 0, 0 ] ) )
            self.x_range = self.range_to_data_field( x_range )
            self.dim_check( )

            self.grid     = self.grid_init( )
        self.mesh     = self.grid_to_mesh( )
        self.coords   = self.grid_to_coords( )
        return
    
    def grid_reinit( self ):
        self.grid = self.grid_init( )
        self.mesh = self.grid_to_mesh( )
        self.coords = self.grid_to_coords( )
        return

    def range_to_data_field( self, range ):
        df_range = [ ]
        for i, j in enumerate( range ):
            range_unit = [ 0, 1, 0, 0 ]
            if ( i > 0 and self.coord == 'sph' ) or ( i > 1 and self.coord == 'cyl' ):
                range_unit = [ 0, 0, 0, 0 ]
            df_range.append( data_field( j, range_unit ) )
        return df_range
    
    def config_init( self, flag ):
        #Initialize grid from config file
        self.coord = self.config[ flag ][ 'coord' ]
        self.cmcoord = self.coord
        if self.coord not in [ 'cart', 'cyl', 'sph' ]:
            raise TypeError( f"{ self.coord } is an invalid coordinate system ")

        self.spacing = np.array(   self.config[ flag ][ 'spacing'].split( ), dtype=np.float64 )
        x_range = np.array( [ self.config[ flag ][ 'x_min'  ].split( ), 
                                               self.config[ flag ][ 'x_max'  ].split( ) 
                                               ], dtype=np.float64 ).transpose( )
        for i, j in enumerate( x_range[ 1: ] ):
            if self.coord in [ 'cyl', 'sph' ]:
                x_range[ i + 1 ] *= np.pi

        self.x_range = self.range_to_data_field( x_range )
    
        self.n_cell = np.array( self.config[ flag ][ 'n_cell' ].split( ), dtype=np.int64 )   
        self.n_gh   = np.array( self.config[ flag ][ 'n_gh'   ].split( ), dtype=np.int64 )        
        self.dim_check( )
        self.grid     = self.grid_init( );
        return        

    def n_tot( self ):
        return list( self.n_cell + 2 * self.n_gh )
    
    def dim_check( self, *args ):
        dimcheck = [ self.spacing, self.x_range, self.n_cell, self.n_gh ]
        for i in args:
            dimcheck.append( i )
        if not all( len ( a ) == len( b ) 
                      for a, b in zip( dimcheck, dimcheck[ 1: ] ) ):
            raise IndexError( "Grid parameters must have the same dimension" )
        return 
        
    def grid_init( self ):
        xgrid = []
        for xrange, spacing, ncell, ngh in zip( self.x_range, self.spacing, self.n_cell, self.n_gh ):
            index = np.arange( ncell )
            lx    = xrange[ 1 ] - xrange[ 0 ]
            if spacing == 1: #constant spacing
                dx = lx / ncell
                range = dx * index + dx / 2
                if ngh > 0:
                    range = np.concatenate( ( range[ 0 ]  - ( np.arange( ngh ) + 1 ) * dx, 
                                              range, 
                                              range[ -1 ] + ( np.arange( ngh ) + 1 ) * dx ) )
                xgrid.append( range )
            else:            #logarithmic spacing
                eta = spacing
                range = grid.log_spacing( eta, xrange[ 0 ], xrange[ 1 ], ncell + 2 * ngh )
                xgrid.append( range )
        return xgrid#[ ::-1 ]
    
    def log_spacing( eta, xmin, xmax, ncell ):
        lx = xmax - xmin
        index = np.arange( ncell )
        spacing_base = lx * ( 1 - eta ) / ( 1 - eta**ncell )
        cellwidth    = spacing_base * eta**index
        cellspacing  = cellwidth[ :-1 ] / 2 + cellwidth[ 1: ] / 2
        range = data_field( np.cumsum( np.append( cellwidth [ 0 ] / 2 , cellspacing ) ), lx.unit )
        range -= xmin
        return range    
    
    # mesh: N dimensional grids after applying meshgrid
    # coords: list of all coordinate points on the grid
    # grid  : 1D arrays for each dimension

    def grid_to_mesh( self ):
        if len( self.grid ) == 1:
            return self.grid
        else:
            meshes = np.meshgrid( *self.grid, indexing='ij' )
            return meshes
        #return self.range_to_data_field( meshes )#np.array( np.meshgrid( *self.grid, indexing='ij' ) )
    
    def grid_to_coords( self ):
        if len( self.grid ) == 1:
            return self.grid
        else:
            mesh = self.grid_to_mesh( )
            coords = np.column_stack( [ a.flatten( ) for a in mesh ] )
            return data_field( coords, [ m.unit for m in mesh ] )
    
    def mesh_to_coords( self ):
        if len( self.grid ) == 1:
            return self.grid
        else:
            coords = np.column_stack( [ a.flatten( ) for a in self.mesh ] )
            return data_field( coords, [ m.unit for m in self.mesh ] )

    def coord_conv( self, tgt_coord ):
        """
        Coordinate conversion between cartesian, spherical and cylindrical coordinates. Transformations
        are only applied to "mesh" and "coords" objects
        
        :param tgt_coord: coordinate to transform to. Must be "sph", "cyl" or "cart"
        """
        if self.cmcoord == tgt_coord:
            print( "Coords/mesh already in target coordinates" )
            return
        if len( self.grid ) == 1:
            print( "Cannot perform coordinate conversion in one dimension" )
            return
        if tgt_coord == "cyl":
            if self.cmcoord == 'sph':
                conv = self.sph_to_cyl( *self.mesh )
            elif self.cmcoord == 'cart':        
                conv = self.cart_to_cyl( *self.mesh )
        elif tgt_coord == "sph":
            if self.cmcoord == 'cyl':
                conv = self.cyl_to_sph( *self.mesh )
            elif self.cmcoord == 'cart':
                conv = self.cyl_to_sph( self.cart_to_cyl( *self.mesh ) )
        elif tgt_coord == "cart":
            if self.cmcoord == "cyl":
                conv = self.cyl_to_cart( *self.mesh )
            elif self.cmcoord == "sph":
                conv = self.sph_to_cyl( self.cyl_to_cart( *self.mesh ) )
        else:
            raise TypeError( f"{tgt_coord} is an invalid coordinate system" )
        self.mesh = conv
        self.coords = self.mesh_to_coords( )
        self.cmcoord = tgt_coord
        return

    def cyl_to_sph( self, *args ):
        conv = [ data_field( np.zeros( i.data.shape ), i.unit ) for i in args ]
        conv[ 0 ] = np.sqrt( args[ 0 ]**2 + args[ 1 ]**2 )
        conv[ 1 ] = np.arccos( args[ 1 ] / conv[ 0 ] )
        if len( args ) > 2:
            conv[ 2 ] = args[ 2 ]
        return conv

    def sph_to_cyl( self, *args ):
        conv = [ data_field( np.zeros( i.data.shape ), i.unit ) for i in args ]
        conv[ 0 ] = args[ 0 ] * np.sin( args[ 1 ] )
        conv[ 1 ] = args[ 0 ] * np.cos( args[ 1 ] )
        if len( args ) > 2:
            conv[ 2 ] = args[ 2 ]
        return conv
    
    def cyl_to_cart( self, *args ):
        conv = [ data_field( np.zeros( i.data.shape ), i.unit ) for i in args ]
        conv[ 0 ] = args[ 0 ] * np.cos( args[ -1 ] )
        conv[ 1 ] = args[ 0 ] * np.sin( args[ -1 ] )
        if len( args ) > 2:
            conv[ 2 ] = args[ 1 ]
        return conv
            
    def cart_to_cyl( self, *args ):
        conv = [ data_field( np.zeros( i.data.shape ), i.unit ) for i in args ]
        conv[ 0 ] = np.sqrt( args[ 0 ]**2 + args[ 1 ]**2 ) 
        conv[ -1 ] = np.arctan2( args[ 1 ], args[ 0 ] )
        if len( args ) > 2:
            conv[ 1 ] = args[ 2 ]
        return conv
    
    def coord_revert( self ):
        """
        Reverts transformed mesh and coords coordinates back to grid coordinates
        """
        self.mesh     = self.grid_to_mesh( )
        self.coords   = self.grid_to_coords( )
        self.cmcoord  = self.coord
        return
    
    # Trivial function replacing current grid with a new grid. To be used by obj classes for interpolation purposes
    def copy_from( self, newgrid ):
        self.coord    = newgrid.coord
        self.cmcoord  = newgrid.cmcoord
        self.spacing  = newgrid.spacing
        self.x_range  = newgrid.x_range
        self.n_cell   = newgrid.n_cell
        self.n_gh     = newgrid.n_gh
        self.grid     = newgrid.grid
        self.mesh     = newgrid.mesh
        self.coords   = newgrid.coords
        return
    
    def coord_transform( self, newgrid, data=None, fill_val=0, print=False ):
        """
        Coordinate conversion to include data held in child classes. Individual "data_field" objects can be specified
        and transformations will only be applied to them, otherwise every "data_field" instance will be transformed.

        It is highly recommended that child classes do not contain too many "data_field" members or the data fields are
        specified

        :param newgrid  : new "grid" object that will be transformed to
        :param data     : string or list of strings specifying the names of the "data_field" objects to be transformed. Default
                          is a search through the class for all instances of "data_field" objects
        :param fill_val : parameter passed onto the interpolation function
        :param print    : debug parameter to print all data members to be transformed
        """
        #
        data_fields = [ ]
        if isinstance( data, str ):
            data_fields.append( data )
        else:
            for name, val in self.__dict__.items():
                if ( isinstance( data, list ) and name in data ) or data is None:
                    if isinstance( val, data_field ) and isinstance( getattr( val, "data", None ), np.ndarray ):
                        if val.shape == self.mesh[ 0 ].shape:
                            data_fields.append( name )
        if print:
            print( f"Fields to be transformed: { data_fields }" )
        newgrid.coord_conv( self.coord )
        for field in data_fields:
            data = getattr( self, field )
            #to_newgrid = LinearNDInterpolator(
            #    self.coords, data.flatten(), fill_value=fill_val )
            transformed_data = self.interpolate_data( newgrid, data, fill_val )
            setattr( self, field, transformed_data ) 
        newgrid.coord_revert( )
        self.copy_from( newgrid )
        return
    
    def interpolate_data( self, newgrid, data, fill_val=0 ):
        to_newgrid = LinearNDInterpolator( self.coords, data.flatten(), fill_value = fill_val )
        newgrid.coord_conv( self.coord )
        transformed_data = to_newgrid( newgrid.coords ).reshape( newgrid.mesh[ 0 ].shape )
        newgrid.coord_revert( )
        return data_field( transformed_data, data.unit )