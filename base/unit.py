from data_field import data_field
import configparser

class classproperty:
    """Descriptor that works as both a property (on instances) and a classmethod (on the class)."""
    def __init__(self, func):
        """Store the wrapped function.

        Args:
            func: The function to be used as a class-level property.
        """
        self.func = func

    def __get__(self, obj, cls):
        """Return the result of calling the stored function with the class.

        Args:
            obj: Instance (unused).
            cls: The class on which the descriptor is accessed.

        Returns:
            The return value of self.func(cls).
        """
        return self.func(cls)

class units:
    """
    The units class contains physical and astrophysical constants. It provides conversion factors
    between CGS and code units based on user-defined base units for density (rho0), length (l0).
    Code units are chosen as default
    """
    ############################################################
    # Constants

    h      = 6.62607e-27   #, [ 1,  2, -1,  0 ]) # CGS Planctk constant
    kb     = 1.38065e-16   #, [ 1,  2, -2, -1 ]) # CGS Boltzmann constant
    eV     = 1.60218e-12   #, [ 1,  2, -2,  0 ]) # CGS eV
    c      = 2.99792458e10 #, [ 0,  1, -1,  0 ]) # CGS speed of light
    q_e    = 4.80321e-10   #, [0.5, 1.5, -1,  0 ]) # CGS electron charge
    me     = 9.1094e-28    #, [ 1,  0,  0,  0 ]) # CGS electron mass;
    mp     = 1.67262e-24   #, [ 1,  0,  0,  0 ]) # CGS proton mass;
    AU     = 1.49598e13    #, [ 0,  1,  0,  0 ]) # Astronomical Unit in cm
    G      = 6.6742831e-8  #, [ -1, 3, -2,  0 ]) # CGS graviational constant
    sig_sb = 5.6704e-5     #, [ 1,  0, -3, -4 ]) # Stefan-Boltzmann
    yr     = 365.*86400.   #, [ 0,  0,  1,  0 ]) # Year in seconds
    pc     = 3.0857e18     #, [ 0,  1,  0,  0 ]) # Parsec in cm
    modot  = 1.9891e33     #, [ 1, 0, 0, 0 ] )     # Solar mass
    rodot  = 6.96e10       #, [ 0, 1, 0, 0 ] )     # Solar radius
    lodot  = 3.828e33      #, [ 0, 1, 0, 0 ] )     # Solar luminosity
    mearth = 5.9742e27     #, [ 1, 0, 0, 0 ] )     # Earth mass
    rearth = 6.37814e8     #, [ 0, 1, 0, 0 ] )     # Earth Radius
    
    rho1   = modot / pc**3  # Astrophysical density unit

    config = configparser.ConfigParser( )    
    rho0 = 1.67262e-24
    l0   = 3.0857e18
    t0   = 1e6 * 365. * 86400

    #@property
    #def m0( self ):
    #    return self.rho0 * self.l0**3
    #@property
    #def v0( self ):
    #    return self.l0 / self.t0
    #@property
    #def sigma0( self ):
    #    return self.rho0 * self.l0
    #
    #@classmethod
    #def m0( units ):
    #    return units.rho0 * units.l0**3
    #@classmethod
    #def v0( units ):
    #    return units.l0 / units.t0
    #@classmethod
    #def sigma0( units ):
    #    return units.rho0 * units.l0
    #
    #@classmethod
    #def G_code( units):
    #    return data_field( units.G / ( units.l0**3 / units.t0**2 / units.m0 ), [ -1, 3, -2,  0 ] )
    @classproperty
    def m0(cls):
        """Code unit mass: rho0 * l0^3."""
        return cls.rho0 * cls.l0**3

    @classproperty
    def v0(cls):
        """Code unit velocity: l0 / t0."""
        return cls.l0 / cls.t0

    @classproperty
    def sigma0(cls):
        """Code unit surface density: rho0 * l0."""
        return cls.rho0 * cls.l0

    @classproperty
    def G_code(cls):
        """Gravitational constant in code units [M^-1 L^3 T^-2]."""
        return data_field(cls.G / (cls.l0**3 / cls.t0**2 / cls.m0), [-1, 3, -2, 0])

    def read_config( config ):
        """Read unit base values from the given ConfigParser config.

        Args:
            config: configparser.ConfigParser object with a 'unit' section
                    containing keys 'rho0', 'l0', 't0'.
        """
        units.config = config
        units.rho0 = float( units.config[ 'unit' ][ 'rho0'] )
        units.l0   = float( units.config[ 'unit' ][ 'l0'  ] )
        units.t0   = float( units.config[ 'unit' ][ 't0'  ] )



    