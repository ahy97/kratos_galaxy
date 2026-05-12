from numpy      import array, byte, meshgrid, exp, ones_like, zeros_like, cosh, tanh
import numpy as np
import sys
from scipy.interpolate import LinearNDInterpolator
sys.path.insert( 0, "../base" )
from unit import units
from data_field import data_field as df
from grid       import grid
from profile_base import profile_base
from FDM import background_potential
from background import background, background_source
import sys
from binary_io  import binary_io
############################################################
# 

class gen_profiles_bin( binary_io ):

    def enroll( self, func, coords, prefix,
                field_names = None, coord_type  = 'sph' ):
        n_pts  = array( [ len( coords[ i ] ) for i in
                          range( len( coords ) ) ] );
        if  not isinstance( coord_type, byte ) :
            coord_type = bytes( coord_type,
                                encoding = 'ascii' )
        #
        self.cache( prefix + '_coord', coord_type );
        self.cache( prefix + '_n_pts',      n_pts );
        for a in range( len( n_pts ) ):
            self.cache( prefix + '_x_%d' % a, coords[ a ] );
        #
        if  field_names  is  None:
            field_names = [ 'data' ];
        if  not hasattr( func, '__len__' ):
            func = [ func ];
        if  len( func ) != len( field_names ):
            raise ValueError( "Inconsistent function "
                              "list length with names" );
        #
        coords_mg = meshgrid( * coords, indexing = 'ij' );
        for n, field in enumerate( field_names ):
            self.cache( prefix + '_' + field,
                        func[ n ]( * coords_mg ));
        #
        return;
    #
#


    
class profile( profile_base ):
    """
    Stores and computes the density and velocity profiles. Density profiles are computed in cylindrical coordinates
    before being transformed to whatever coordinate system this object is initialized with. 

    The construction of the profile assumes the following:
    
        1) Full separability of the density profile with respect to each orthogonal direction, such that the net profile
           is a product of profiles along each dimension
        2) The velocity profile is purely azimuthal
    
    """
    def __init__( self, **kwargs ):#flag ):
        """
        Initializes an instance of the "profile" object. Either a config file must be loaded as a class attribute of
        "units", or keyword arguments must be specified.

        :param \**kwargs: See below

        :Keyword Arguments:
            **profile_flags        : list of strings specifying the profile along each dimension. Allowed types are specified
                                     in base/profile_base.py and can be added to in the rho method defined in this class
            **rhobase              : Scaling factor of the net profile, with units of rho
            **mu                   : Mean molecular weight in units of m_p
            **Tdisc                : Initial temperature of the gas. An initial isothermal state is assumed
            **background_potential : Precomputed "background_potential" - default is recomputing it using configuration file parameters
            **background           : Existing "background" - default is re-initializing using configuration file parameters
            **cylgrid              : The cylindrical grid on which the profiles are computed before being transformed to the 
                                     base grid. Default is a cylindrical grid with r and z copying the gridpoints of the first dimension
                                     of the base grid
            **scale_lengths        : list of list of floats specifying the scale lengths for the predefined density profiles
            All other keyword arguments are directly passed to the "grid" class in base/grid.py
        """
        super().__init__( **kwargs )
        self.flags = kwargs.get( "profile_flags", self.config[ 'IC_profile' ][ 'profile_flags' ].split( ) )
        if len( self.flags ) != len( self.grid ):
            raise IndexError( "Grid dimension must be same as number of profile flags" )
        self.rhobase = kwargs.get( "rhobase", float( self.config.get( 'IC_hydro', "rhobase", fallback=1    ) ) )
        mu           = kwargs.get( "mu"     , float( self.config.get( 'IC_hydro', "mu"     , fallback=1.26 ) ) )
        Tdisc        = kwargs.get( "Tdisc"  , float( self.config.get( 'IC_hydro', "Tdisc"  , fallback=1e4  ) ) )
        
        self.cs2 = df( self.kb * Tdisc / mu / self.mp / ( self.v0**2 ), "v^2" )
        self.rhobase = df( self.rhobase, "rho" )

        if "background" in kwargs:
            background_obj = kwargs[ "background" ]
            self.background_potential = background_potential( background_obj )
        elif "background_potential" in kwargs:
            self.background_potential = kwargs[ "background_potential" ]
        else:
            self.background_potential = background_potential( background( ) )
            
        self.cylgrid = kwargs.get( 'cylgrid',
                               grid( coord    = 'cyl',
                                     spacing  = [ self.spacing[ 0 ], self.spacing[ 0 ] ],
                                     x_min    = [ df( self.x_range[ 0 ][ 0 ], "l" ), df( self.x_range[ 0 ][ 0 ], "l" ) ],
                                     x_max    = [ df( self.x_range[ 0 ][ 1 ], "l" ), df( self.x_range[ 0 ][ 1 ], "l" ) ],
                                     n_cell   = [ self.n_cell[ 0 ], self.n_cell[ 0 ] ],
                                     n_gh     = [ 0, 0 ] ) )    

        scale_lengths = kwargs.get( "scale_lengths", None )
        if scale_lengths is None:
            self.scale_lengths = []
        else:
            self.scale_lengths = [ df( np.array( i, dtype=np.float64 ), "l" ) for i in scale_lengths ]
        for i in self.flags:
            if i == 'zsolve':
                self.zsolve_init( **kwargs )
                if scale_lengths is None:
                    self.scale_lengths.append( [ None ] )
            else:
                if scale_lengths is None:
                    self.scale_lengths.append( df( np.array( self.config[ i ][ 'scale_lengths' ].split( ), dtype=np.float64 ), "l" ) ) 

        self.densprof = self.calc_densprof( )
        self.velprof  = self.calc_velprof( )

        self.metdisc      = kwargs.get( "metdisc"  , float( self.config.get( 'IC_hydro', "metdisc"  , fallback=3  ) ) )
        self.metcgm       = kwargs.get( "metcgm"   , float( self.config.get( 'IC_hydro', "metcgm"   , fallback=0.1  ) ) )
        self.metzcut      = kwargs.get( "metzcut"  , float( self.config.get( 'IC_hydro', "metzcut"  , fallback=600  ) ) )
        self.metzcut      = df( self.metzcut, "l" )

        self.metprof = self.calc_metprof( )

        return 
    
    def zsolve_init( self, **kwargs ):
        if self.flags[ 0 ] == 'zsolve':
            raise TypeError( "zsolve only applicable for z direction" )
        self.rcut = float( self.config.get( 'zsolve', 'rcut', fallback=2e4 ) )
        self.zcut = float( self.config.get( 'zsolve', 'zcut', fallback=2e4 ) )
        self.cylgrid.x_range[ 0 ] = df( np.array( [ 0, self.rcut ] ), "l" )
        self.cylgrid.x_range[ 1 ] = df( np.array( [ 0, self.zcut ] ), "l" )
        self.cylgrid.grid_reinit( )
        return
    
    def calc_densprof( self ):
        profs = []
        for i, j in enumerate( self.flags ):
            profs.append( self.rho( self.cylgrid.mesh[ i ], j, *self.scale_lengths[ i ] ) )
        res = self.rhobase * np.prod( np.array( profs ), axis=0 )
        return self.cylgrid.interpolate_data( self, res, fill_val=0 ) #self.rhobase * np.prod( np.array( profs ), axis=0 )
    
    def calc_velprof( self ):
        self.background_potential.coord_transform( self.cylgrid, fill_val=0 )
        potcyl = self.background_potential.phi_numeric
        Rcyl   = self.cylgrid.mesh[ 0 ]
        dphidr = ( potcyl[ 2: ] - potcyl[ :-2 ] ) / ( Rcyl[ 2:  ] - Rcyl[ :-2 ] )#np.zeros( Rcyl.shape )
        dphidr = np.concatenate( ( dphidr[ :1, : ], dphidr, dphidr[ -1:, : ] ) )
        dphidr += self.background_potential.background.dphidR_analytic( *self.cylgrid.mesh )
        velsq          =   Rcyl * dphidr + self.cs2 *\
                                         ( Rcyl / self.rho( Rcyl, self.flags[ 0 ], *self.scale_lengths[ 0 ] ) ) *\
                                           self.rho_prime( Rcyl, self.flags[ 0 ], *self.scale_lengths[ 0 ] )
        velsq[ velsq < 0 ] = 0
        velsq[ np.isnan( velsq.data ) ] = 0
        return self.cylgrid.interpolate_data( self, np.sqrt( velsq ), fill_val=0 )
        
    def calc_metprof( self ):
        metprof = self.metdisc * np.ones( self.densprof.shape )
        self.coord_conv( "cyl" )
        metprof[ self.mesh[ 1 ] > self.metzcut ] = self.metcgm
        self.coord_revert( )
        return metprof

    def rho( self, x, flag, *args ):
        #If you want another analytic profile
        #if flag == s:
        #    return profile
        #else:
        if flag == "zsolve":
            return self.zsolve( )
        return super().rho( x, flag, *args )

    def rho_prime( self, x, flag, *args ):
        #Don't forget to also define the derivative
        #if flag == s:
        #    return profile
        #else:
        return super().rho_prime( x, flag, *args )
    
    def zsolve( self ):
        """
        A relatively fast vertical profile solver using Wang+2010's scheme to approximate vertical hydrostatic equilibrium
        accounting for the background potential and gas self gravity. Applies RK4 from the midplane upwards in a single loop
        """
        rcyl, zcyl = self.cylgrid.grid
        self.background_potential.coord_transform( self.cylgrid )
        potcyl = self.background_potential.potential
        pot0 = np.reshape( potcyl[ :,0 ], ( len( potcyl[ :,0 ] ) , 1 ) )
        pot0cyl = np.tile( pot0, ( 1, len( pot0 ) ) )
        potz = potcyl - pot0cyl

        #third axis to represent integrating along Z and its corresponding dz
        Zdiff, Zcyl3, Rcyl3 = np.meshgrid(np.insert(np.diff(zcyl), 0, zcyl[0]), zcyl, rcyl, indexing='ij') 
        Zcyl3vert, _, _ = np.meshgrid(zcyl, zcyl, rcyl, indexing='ij')

        #only integrate from 0 to z - beyond z, pad with zeros
        Zdiff[Zcyl3vert > Zcyl3] = 0

        #potential variation along z plotted along third axis, same axis which we do integration
        pot3D = np.tile(potcyl, (len(Zdiff),1,1))
        pot3D = np.transpose( pot3D, (1,0,2) )
        pot3Dz = pot3D - pot3D[0]


        def d2pdz2array( index, phizgas ):
            return 4 * np.pi * self.G_code * self.rhobase * self.rho( Rcyl3[index], self.flags[ 0 ], *self.scale_lengths[ 0 ] ) * np.exp( - ( pot3Dz[index] + phizgas ) / self.cs2 )   

        
        phizgas     = df( np.zeros( self.cylgrid.mesh[ 0 ].shape ), [ 0, 2, -2, 0 ] )
        dphizgas_dz = df( np.zeros( self.cylgrid.mesh[ 0 ].shape ), [ 0, 1, -2, 0 ] )
        
        #RK integration
        for j, h in enumerate( Zdiff ):
            k1 = h * dphizgas_dz
            l1 = h * d2pdz2array( j, phizgas )
            arg1 = phizgas + 0.5 * k1

            k2 = h * (dphizgas_dz + 0.5 * l1)
            l2 = h * d2pdz2array( j, arg1 )
            arg2 = phizgas + 0.5 * k2

            k3 = h * (dphizgas_dz + 0.5 * l2)
            l3 = h * d2pdz2array( j, arg2 )
            arg3 = phizgas + k3

            k4 = h * (dphizgas_dz + l3)
            l4 = h * d2pdz2array( j, arg3 )

            phizgas = phizgas + (k1 + 2. * k2 + 2. * k3 + k4) / 6.
            dphizgas_dz = dphizgas_dz + (l1 + 2. * l2 + 2. * l3 + l4) / 6.
        
        #Interpolate to desired grid
        exp_term = potz + phizgas #self.cylgrid.interpolate_data( self, potz + phizgas, fill_val=-np.inf )
        return np.exp( - exp_term / self.cs2 )
            
    
        
class IC( gen_profiles_bin ):
    def __init__( self, profile, file_name=None, cache_used=True, **kwargs ):
        if file_name is None:
            file_name = profile.config.get( "IC_profile", "ICname", fallback=None )
        super().__init__( file_name, cache_used )

        grid_cgs = [ i.to_cgs( ) for i in profile.grid ]
        self.enroll( [ lambda *args : profile.densprof.to_cgs( ), 
                       lambda *args : profile.densprof.to_cgs( ) * profile.cs2.to_cgs( ),
                       lambda *args : profile.velprof.to_cgs( ), 
                       lambda *args : profile.metprof ],
                       [ i.data for i in grid_cgs ], 'hydro', 
                       field_names = [ 'rho', 'pre', 'vel', 'met' ] )
        
        if kwargs.get( "use_background", True ):
            bg2d = background_source( )
            bg3d = background_source( )
            for source_key, source in profile.background_potential.background.components.items():
                if source.dim == 2:
                    bg2d  = bg2d + source
                elif source.dim == 3:
                    bg3d  = bg3d + source
            
            if kwargs.get( "cmz", False ):
                bg2d = ( bg3d.mass + bg2d.mass ) / bg2d.mass * bg2d
            
            
            self.enroll( bg2d.src_cgs, grid_cgs, 'bg' )
            if len( profile.grid ) == 3:
                self.enroll( bg3d.src_cgs, grid_cgs, 'bar' )
            else:
                x3 = kwargs.get( "x3", grid_cgs[ -1 ] )
                self.enroll( bg3d.src_cgs, grid_cgs + [ x3 ], 'bar' )
        self.save( )
        self.close( )
        return
            
            

                    
    



