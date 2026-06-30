import scipy.sparse as sparse
import sys
sys.path.insert( 0, "../base" )
from unit import units
from data_field import data_field as df
from grid import grid
from background import background_source
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm



class background_potential( grid ):
    """
    Applies the background source functions to a grid and computes the potential. The analytic potential
    can be directly computed, while solvers can be incorporated to numerically solve non analytic potentials.
    """
    def __init__( self, background, **kwargs ):
        """
        Initializes an instance of a background_potential object. Either a config file must be loaded as a class attribute of
        "units", or keyword arguments must be specified. A background object containing the source functions being used must
        also be specified
        
        :param background: background object containing all the source functions in addition to the analytic potential
        :param \**kwargs: See below

        :Keyword Arguments:
            All keyword arguments are directly passed to the "grid" class in base/grid.py
        
        """
        super().__init__( config_flag = "FDM", **kwargs )
        self.background = background
        self.axsym_src = background_source( ) #empty dummy source
        for source_key, source in background.components.items():
            if source.dim == 2:
                self.axsym_src = self.axsym_src + source
                
        self.phi_analytic = self.background.phi_analytic( *self.mesh )
        if bool( self.config.get( "FDM", "FDM_on", fallback=False ) ) and not self.axsym_src.dummy_flag:
            self.phi_numeric = self.FDM_solve( )
        else:
            self.phi_numeric = df( np.zeros( self.phi_analytic.shape ), "v^2" )
        self.potential = self.phi_analytic + self.phi_numeric
        return

 
 
    def FDM_solve( self ):
        """
        A finite differences solver, using source functions from the background object and the grid as defined
        in the parent class. Currently only a 2D spherical-polar solver is implemented, with uniform or logarithmic
        spacing along r.
        """
        dens = 4 * np.pi * self.G_code * self.axsym_src.src( *self.mesh )
        bcs = np.array( [ self.config[ 'FDM' ][ 'bound_min' ].split( ),
                          self.config[ 'FDM' ][ 'bound_max' ].split( ) ], dtype='str' ).transpose()
        self.dim_check( bcs )
        #Only includes 2D spherical solver in r-theta now
        if self.coord == 'sph':
            if len( self.mesh ) == 2:
                Nr, Ntheta = self.n_tot( )
                R, Theta = self.mesh
                r_nonbound     = self.grid[ 0 ][ 1 : -1 ]
                theta_nonbound = self.grid[ 1 ][ 1 : -1 ]
                R_nonbound, Theta_nonbound = \
                np.meshgrid( r_nonbound, theta_nonbound, indexing='ij' )
                dr_p = np.diff( self.grid[ 0 ] )[ 1 :  ]
                dr_m = np.diff( self.grid[ 0 ] )[ : -1 ]
                dr_avg = dr_m / 2 + dr_p / 2
                dtheta = np.diff( self.grid[ 1 ] )[ 0 ] * np.ones( theta_nonbound.shape )#np.diff( self.grid.grid[ 1 ] )
                Dr_p, Dtheta   = np.meshgrid( dr_p, dtheta  , indexing='ij' )
                Dr_m, Dtheta   = np.meshgrid( dr_m, dtheta  , indexing='ij' )
                Dr_avg, Dtheta = np.meshgrid( dr_avg, dtheta, indexing='ij' )

                outer   = ( 2 / R_nonbound ) * ( 1 / ( Dr_m + Dr_p ) ) + ( 1 / ( Dr_p * Dr_avg ) ) 
                inner   = ( 1 / ( Dr_m * Dr_avg ) ) - ( 2 / R_nonbound )*( 1 / ( Dr_p + Dr_m ) )

                cw     = np.cos( Theta_nonbound ) / ( np.sin( Theta_nonbound ) * R_nonbound * R_nonbound * 2 * Dtheta )  + 1 / ( Dtheta * Dtheta * R_nonbound * R_nonbound )
                ccw    =  1 / ( Dtheta * Dtheta * R_nonbound * R_nonbound ) - ( np.cos( Theta_nonbound ) / ( np.sin( Theta_nonbound ) * R_nonbound * R_nonbound * 2 * Dtheta ) ) 
                center = - 1 / ( Dr_p * Dr_avg ) - 1 / ( Dr_m * Dr_avg ) - 2 / ( R_nonbound * R_nonbound * Dtheta * Dtheta )

                outer  = self.surround_with_zeros( outer )
                inner  = self.surround_with_zeros( inner )
                cw     = self.surround_with_zeros( cw )
                ccw    = self.surround_with_zeros( ccw )
                center = self.surround_with_zeros( center )
                
                sten_comp = {
                    ( 0, 1 ) : outer,
                    ( 0,-1 ) : inner,
                    ( 1, 0 ) : cw,
                    (-1, 0 ) : ccw,
                    ( 0, 0 ) : center }
                
                sten_comp, dens = self.boundaries( sten_comp, dens, bcs );
                outer, inner, cw, ccw, center = sten_comp
                
                centerinds = np.arange( np.prod( R.shape ) )
                outinds    = centerinds + Ntheta
                ininds     = centerinds - Ntheta
                cwinds     = centerinds + 1
                ccwinds    = centerinds - 1
                rowinds    = np.sort( np.tile( centerinds , 5 ) )

                columninds = np.array( [ ininds, ccwinds, centerinds, cwinds, outinds ], dtype=np.float64 ).T

                c1 = np.array( [ ininds, ccwinds, centerinds, cwinds, outinds ], dtype=np.float64 ).T
                NR, NT = np.meshgrid( np.arange( Nr ), 
                                      np.arange( Ntheta ), indexing='ij' )
                coordsnr = NR.flatten()
                coordsnt = NT.flatten()
                #remove 0 boundary matrix elements
                c1 [coordsnr == 0         ] += np.tile( np.array( [ np.nan,      0, 0,      0,      0 ] ), ( len( c1[ coordsnr == 0 ]         ), 1 ) )
                c1 [coordsnr == Nr - 1    ] += np.tile( np.array( [      0,      0, 0,      0, np.nan ] ), ( len( c1[ coordsnr == Nr - 1]     ), 1 ) )
                c1 [coordsnt == 0         ] += np.tile( np.array( [      0, np.nan, 0,      0,      0 ] ), ( len( c1[ coordsnt == 0 ]         ), 1 ) )
                c1 [coordsnt == Ntheta - 1] += np.tile( np.array( [      0,      0, 0, np.nan,      0 ] ), ( len( c1[ coordsnt == Ntheta - 1] ), 1 ) )


                columninds = c1.flatten( )
                posinds    = ~np.isnan( columninds )
                columninds = columninds[ posinds ]
                columninds = columninds.astype( 'int64' )
                rowinds    = rowinds[ posinds ]

                matrixelements = np.array( [ inner.flatten() , 
                                             ccw.flatten()   , 
                                             center.flatten(), 
                                             cw.flatten()    , 
                                             outer.flatten() ] ).T

                matrixelements = matrixelements.flatten()
                matrixelements = matrixelements[posinds]
                densvector = dens.flatten()
                
                
                matrix = sparse.csr_matrix( ( matrixelements, ( rowinds, columninds ) ), shape=( np.prod(R.shape), np.prod(R.shape)) )
                #Solve sparse system
                solve = sparse.linalg.factorized( matrix )
                potential = solve( densvector )
                potential2D = potential.reshape( R.shape )
                return df( potential2D, [ 0, 2, -2, 0 ] )
                
    def plot( self, **kwargs ):
        """
        A helper function plotting the solution to the FDM solver. 

        :param \**kwargs: See below. Note - these keyword arguments are not the same as those 
        passed in __init__. 

        :Keyword Arguments:
            **rhonorm : Density normalization
            **lnorm   : Length normalization. r only for spherical, r and z for cylindrical,
                        x, y, z for cartesian
            **phinorm : Potential normalization
            **tlim    : Theta axis limits (polar only)
            **rlim    : R axis limits (polar only)
            **vmin    : minimum density, both for colormesh plot and as axis limit
            **vmax    : maximum density, both for colormesh plot and as axis limit     
        """
        mesh = np.meshgrid( *self.grid, indexing='ij' )
        R, Theta = mesh
        fig = plt.figure( figsize = ( 12, 8 ) )
    
        rhonorm  = kwargs.get( 'rhonorm', 1 )
        lnorm    = kwargs.get( 'lnorm', 1 )
        phinorm  = kwargs.get( 'phinorm', 1 )    
        tlim     = kwargs.get( 'tlim', [ Theta.min(  ), Theta.max(  ) ] )    
        rlim     = kwargs.get( 'rlim', [ R.min(  ) * lnorm, R.max(  ) * lnorm ] )    
        vmin     = kwargs.get( 'vmin', 1e-2 )  
        vmax     = kwargs.get( 'vmax', 1e2 )
        filename = kwargs.get( 'filename', 'FDM_Solve' ) 
        
        Thetaplot = Theta[1:-1:,1:-1]
        Rplot     = R[1:-1:,1:-1] / lnorm
        potentialplot = self.potential[1:-1:,1:-1] / phinorm

        srcplot   = self.axsym_src.dens( Rplot * lnorm, Thetaplot ) / rhonorm
        ax11  = fig.add_subplot( 221, projection = 'polar' );
        ax12  = fig.add_subplot( 222, projection = 'polar' );
        ax21  = fig.add_subplot( 223 );
        ax22  = fig.add_subplot( 224 );
        
        pcm11 = ax11.pcolormesh( np.pi/2 - Thetaplot, Rplot, -potentialplot )
        ax11.set_xlim( tlim )
        ax11.set_ylim( rlim )
        cbar11 = fig.colorbar( pcm11 )
        
        pcm12 = ax12.pcolormesh( np.pi/2 - Thetaplot, Rplot, srcplot, norm=LogNorm( vmin = vmin, vmax = vmax ) )
        cbar12 = fig.colorbar( pcm12 )
        ax12.set_xlim( tlim )
        ax12.set_ylim( rlim )
        
        ax21.semilogx( Rplot[ :,-1 ], 
               potentialplot[ :,-1 ] )
        ax21.set_xlabel(r"R ($l_0$)")
        ax21.set_ylabel(r"$\Phi$")
        ax21.set_xlim( rlim )
        
        ax22.loglog( Rplot[ :,-1 ], 
                   srcplot[ :,-1 ] )
        ax22.set_xlabel(r"R ($l_0$)")
        ax22.set_ylabel(r"$\rho_*$")
        ax22.set_xlim( rlim )
        ax22.set_ylim( [ vmin , vmax ] )
        plt.savefig(f"{filename}.png",dpi=300)
        
    def surround_with_zeros( self, original_array ):
        """Pad a 2D array with a border of zeros, expanding shape by 2 in each dim.

        Args:
            original_array (ndarray): Input 2D array.

        Returns:
            ndarray with shape (N+2, M+2).
        """
        original_shape = original_array.shape
        N, M = original_shape
        surrounded_array = np.zeros( ( N + 2, M + 2 ) )
        surrounded_array[1:-1, 1:-1] = original_array
        return surrounded_array
    
    def boundaries( self, sten_comp, dens, bcs ):
        """Apply Neumann and Dirichlet boundary conditions to the FDM stencil.

        Args:
            sten_comp (dict): Dict of stencil coefficient arrays keyed by (di, dj).
            dens (ndarray): Source density array, modified in-place.
            bcs (ndarray): Boundary condition specifiers ('neu'/'dir' per edge).

        Returns:
            tuple of (sten_comp.values(), dens) with BCs applied.
        """
        idx_base = [ slice( None ), slice( None ) ]
        stencil_base = [ 0, 0 ]
        neu_inds  = [ 0, -1 ] #indices where neumann BCs are applied - outermost
        dir_inds  = [ 1, -2 ] #indices where dirichlet BCs are applied - 1 cell from outermost
        sten_inds = [ 1, -1 ] #inds of five point stencil logical locations
        for i, j in enumerate( bcs ):
            for k, l in enumerate( j ):
                idx = idx_base.copy()
                stencil = stencil_base.copy()
                stencil[ 1 - i ] = sten_inds[ k ]  #stencil logical location
                if "neu" in l:
                    idx[ i ] = neu_inds[ k ]       #Array slicing - specify index on array
                    sten_comp[ tuple( stencil ) ][ tuple( idx ) ] = 1
                    sten_comp[ ( 0, 0 ) ][ tuple( idx ) ] = -1
                    dens[ tuple( idx ) ] = 0
                if "dir" in l:
                    idx[ i ] = dir_inds[ k ]
                    sten_comp[ ( 0, 0 ) ][ tuple( idx ) ] = -1
                    dens[ tuple( idx ) ] = 0
        return sten_comp.values(), dens
    

        
        
    