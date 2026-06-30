from grid import grid
import numpy as np

class profile_base( grid ):
    """
    Contains typical disc or sphere profiles used in astrophysical simulations. This class is initialized as a
    child of the "grid" class to account for numerically computed profiles
    """
    def __init__( self, **kwargs ):
        """Initialize profile_base via parent grid initialization.

        Args:
            **kwargs: Passed through to grid.__init__.
        """
        super().__init__( **kwargs )
        return
    def rho( self, x, flag, *args ):
        """Evaluate a radial density profile.

        Args:
            x: Radial coordinate (scalar or array).
            flag (str): Profile type ('disc', 'exp', 'gauss', 'uniform', 'sech2').
            *args: Profile-specific scale parameters.

        Returns:
            Profile value(s) at x.

        Raises:
            ValueError if flag is unknown.
        """
        if flag == 'disc':
            return np.exp( -args[ 0 ] / x - x / args[ 1 ] )
        if flag == 'exp':
            return np.exp( -x / args[ 0 ] )
        if flag == 'gauss':
            return np.exp( - ( x / args[ 0 ] )**2 )
        if flag == 'uniform':
            return np.ones_like( x )
        if flag == 'sech2':
            return np.cosh( x / args[ 0 ] )**( -2 )
        raise ValueError( "Unknown radial profile flag: %s" %
                          flag )
    def rho_prime( self, x, flag, *args ):
        """Evaluate the radial derivative of a density profile.

        Args:
            x: Radial coordinate (scalar or array).
            flag (str): Profile type ('disc', 'exp', 'gauss', 'uniform', 'sech2').
            *args: Profile-specific scale parameters.

        Returns:
            Derivative value(s) at x.

        Raises:
            ValueError if flag is unknown.
        """
        if flag == 'disc':
            return self.rho( x, flag, *args ) * ( args[ 0 ] / x**2 - 1.0 / args[ 1 ] )
        if flag == 'exp':
            return -self.rho( x, flag, *args ) / args[ 0 ]
        if flag == 'gauss':
            return -2.0 * x * self.rho( x, *args ) / args[ 0 ]**2
        if flag == 'uniform':
            return np.zeros_like( x )
        if flag == 'sech2':
            return -2.0 * self.rho( x, flag, *args ) * np.tanh( x / args[ 0 ] ) / args[ 0 ]
        #else:
        #    print( "Invalid or nonexistent flag. Manually computing derivative" )
        #    if type( x ) == float or type( x ) == int:
        #        return ( self.rho( x + 1e-32, *args ) - self.rho( x - 1e-32, *args ) ) / ( 2e-32 )
        #    else:        
        raise ValueError( "Unknown radial profile flag: %s" %
                          flag )
    