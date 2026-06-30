import numpy as np
import sys
sys.path.insert( 0, "../base" )
from unit import units
from data_field import data_field as df
from source import source 
from inspect import signature
from scipy.integrate import tplquad, dblquad

class background_source( source ):
    """
    A wrapper for a background source function, incorporating the mass of the source and numerical 
    integration to compute the mass 

    """
    def __init__( self, **kwargs ):
        """
        Initializes an instance of a background_source object. Either a config file must be loaded as a class attribute of
        "units", or keyword arguments must be specified
        
        :param \**kwargs: See below

        :Keyword Arguments:
            **mass : Total mass of the density function if known. Defaults to a numerical integration over space
        
        Additional keyword arguments are passed into the "source" class in base/source.py
        """
        super().__init__( **kwargs )
        if self.dummy_flag:
            self.mass = df( 0, [ 1, 0, 0, 0 ] )
            return
        if "mass" in kwargs:
            mass = kwargs[ "mass" ]
        else:
            mass = self.getmass( )
        self.mass = df( mass, [ 1, 0, 0, 0 ] )
        return

    def getmass( self ):
        """Numerically integrate the source density to compute total mass.

        Returns:
            float: Total mass (code units).
        """
        rmax_intg = float( self.config.get( 'background_mass', 'rmax_intg', fallback=1e5 ) )
        if self.dim == 2:
            mass, err = dblquad(
            lambda r, theta: ( self.dens( r, theta ) ) * r**2 * np.sin(theta),
            0, np.pi,  # Limits of theta
            lambda theta: 0, lambda theta: rmax_intg # Limits of r
            )
            mass *= 2 * np.pi
        elif self.dim == 3:
            mass, err = tplquad(
            lambda r, theta, phi: self.dens(r, theta, phi ) * r**2 * np.sin(theta),
            0, 2*np.pi,        # Limits of phi
            lambda phi: 0, lambda phi: np.pi,    # Limits of theta
            lambda phi, theta: 0, lambda phi, theta: rmax_intg  # Limits of r
        )
        return mass
        
    def __add__( self, obj ):
        """Add two background sources, summing masses.

        Args:
            obj (background_source): Source to add.

        Returns:
            background_source with summed densities and masses.
        """
        source_sum = super().__add__( obj )
        if source_sum.dummy_flag:
            new_mass = 0
        else:
            new_mass = self.mass + obj.mass#getattr(obj, 'mass', 0)
        return background_source( func = source_sum.dens,
                                  dim  = source_sum.dim,
                                  mass = new_mass,
                                  unit = source_sum.unit )
    def __radd__( self, obj ):
        """Reverse add: delegate to __add__."""
        return self.__add__( obj )
    
    
    def __mul__( self, scalar ):
        """Scale background source density and mass by a scalar.

        Args:
            scalar (float): Multiplicative factor.

        Returns:
            background_source with scaled density and mass.
        """
        source_scaled = super().__mul__( scalar )
        new_mass = self.mass * scalar
        return background_source( func = source_scaled.dens,
                                  dim  = source_scaled.dim,
                                  mass = new_mass,
                                  unit = source_scaled.unit )
    def __rmul__( self, scalar ):
        """Reverse multiply: delegate to __mul__."""
        return self.__mul__( scalar )
        

        

class background( units ):
    """
    Compiles all background potential sources together. User can add additional source functions
    below.Source functions defined here should only take in float-like or array-like arguments in
    code units and should not be data_field objects.
    """
    def __init__( self, **kwargs ):
        """
        Initializes an instance of a background object. Either a config file must be loaded as a class attribute of
        "units", or dictionaries all_comp and comp_mass must be defined.

        :param \**kwargs: See below

        :Keyword Arguments:
            **all_components  : Dictionary of enabled source function names as keys. Values should all be 1
            **component_masses: Dictionary of enable source function masses if available. Keys should be 'M_{ func_name }'
                                values should be masses
        """

        all_components   = kwargs.get( "all_components", 
                                        dict( self.config.items( "background_src" ) ) )
        component_masses = kwargs.get( "component_masses",
                                        dict( self.config.items( "background_mass" ) ) )
        self.components = {}
        
        for name, enabled in all_components.items():
            if int( enabled ) != 1:
                continue
            # skip if function not implemented
            func = getattr( self, name, None )
            if func is None or not callable( func ):
                print( f"Source function {name} not implemented, skipping" )
                continue
            kwargs2 = dict( func=func,
                           unit=[ 1, -3, 0, 0 ] )
            mass_key = f"m_{ name }"
            if mass_key in component_masses:
                kwargs2[ "mass" ] = (
                    float( component_masses[ mass_key ] ) * self.modot / self.m0
                )
            src = background_source( **kwargs2 )
            setattr( self, f"{ name }_", src )
            self.components[ name ] = src

        # Parameters specifying an analytic background source

        self.rho_halo = df( float( self.config.get( 'background_analytic', 'rho_halo', fallback=0.3424 ) ), "rho" )
        self.r_halo   = df( float( self.config.get( 'background_analytic', 'r_halo'  , fallback=20200  ) ), "l"   )
        self.M_bh     = df( float( self.config.get( 'background_analytic', 'M_bh'    , fallback=4e6 * self.modot / self.m0 ) ), "m" )
        
        return
    
    
    def phi_analytic( self, *args ):
        """
        This function specifies the exact form of the analytic potential

        :param args: Position. Can be 1, 2 or 3 arguments
        """
        r = args[ 0 ]
        x = r / self.r_halo
        phi_halo = -4 * np.pi * self.G_code * self.rho_halo * self.r_halo**3 * np.log(1 + x) / r
        phi_point_src = - self.G_code * self.M_bh / r
        return phi_halo + phi_point_src
    
    def dphidR_analytic( self, R, z ):
        """
        This function specifies the cylindrical radial derivative of the analytic potential. Must be
        axisymmetric and must be in cylindrical coordinates
        """
        r = np.sqrt( R**2 + z**2 )
        fpgrhoR = 4 * np.pi * self.G_code * self.rho_halo * R
        logfactor = np.log( 1 + r / self.r_halo )
        dphidr_halo = fpgrhoR * self.r_halo**3 * logfactor / ( r**3 ) -\
                      fpgrhoR * self.r_halo**2 / ( r**2 * ( 1 + r / self.r_halo ) )
        dphidr_point_src = self.G_code * self.M_bh * R / ( r**3 )
        return dphidr_halo + dphidr_point_src

        


    def bar_portail( self, r, theta, phi ):
        """Bar+portail density profile (Sormani 2022 + Portail 2017).

        Args:
            r, theta, phi: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        def bar1( x, y, z ):
            rho_1 = 0.316 * 10**10 * self.modot * ( 1000 * self.pc )**-3 / self.rho0
            x_1   = 490 * self.pc / self.l0
            y_1   = 392 * self.pc / self.l0
            z_1   = 229 * self.pc / self.l0
            c_par = 1.991
            c_perp = 2.232
            m     = 0.873
            alpha = 0.626
            n     = 1.940
            c     = 1.342
            x_c   = 751  * self.pc / self.l0
            y_c   = 469  * self.pc / self.l0
            r_cut = 4370 * self.pc / self.l0

            a = ( ( ( np.abs( x ) / x_1 )**c_perp + 
                    ( np.abs( y ) / y_1 )**c_perp )**( c_par / c_perp ) +
                    ( np.abs( z ) / z_1 )**c_par )**( 1 / c_par )
            a_p = ( ( ( x + c*z ) / x_c )**2 +
                           ( y / y_c )**2 )**0.5
            a_m = ( ( ( x - c*z ) / x_c )**2 +
                           ( y / y_c )**2 )**0.5
            r = ( x**2 + y**2 + z**2 )**0.5

            comp1 = rho_1 / np.cosh( a**m )
            comp2 = 1 + alpha * ( np.exp( -a_p**n ) + np.exp( -a_m**n ) )
            comp3 = np.exp( -( r / r_cut )**2 )
            return comp1 * comp2 * comp3


        def barn( x, y, z, params ):
            rho_i, x_i, y_i, z_i, n_i,  c_perp_i, R_i_out, R_i_in, n_i_out, n_i_in = params
            a_i = ( ( np.abs( x ) / x_i )**c_perp_i +
                    ( np.abs( y ) / y_i )**c_perp_i )**( 1 / c_perp_i )
            R = ( x**2 + y**2 )**0.5
            comp1 = rho_i * np.exp( -a_i**n_i )
            comp2 = np.cosh( z / z_i )**-2
            comp3 = np.exp( -( R / R_i_out )**n_i_out )
            comp4 = np.exp( -( R_i_in / R )**n_i_in )
            return comp1 * comp2 * comp3 * comp4
        
        z = r * np.cos( theta )
        R = r * np.sin( theta ) 

        x = R * np.cos( phi )
        y = R * np.sin( phi )
        params2 = [ 0.050 * 10**10 * self.modot * ( 1000 * self.pc )**-3 / self.rho0,
                    5364           * self.pc / self.l0,
                    959            * self.pc / self.l0,
                    611            * self.pc / self.l0,
                    3.051,
                    0.970,
                    3190           * self.pc / self.l0,
                    558            * self.pc / self.l0,
                    16.731,
                    3.196 ]
        params3 = [ 1743.049 * 10**10 * self.modot * ( 1000 * self.pc )**-3 / self.rho0,
                    478               * self.pc / self.l0,
                    267               * self.pc / self.l0,
                    252               * self.pc / self.l0,
                    0.980,   
                    1.879,   
                    2204              * self.pc / self.l0,
                    7607              * self.pc / self.l0,
                   -27.291,
                    1.630 ]
        rhobar_1 = bar1( x, y, z )
        rhobar_2 = barn( x, y, z, params2 )
        rhobar_3 = barn( x, y, z, params3 )
        return rhobar_1 + rhobar_2 + rhobar_3

    def bulge_s19( self, r, theta ):
        """Bulge density profile (Sormani 2019).

        Args:
            r, theta: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        rhob0 = 9.5E4 * self.rho1 / self.rho0
        alpha = 1.8
        acut  = 500.  * self.pc   / self.l0
        qb    = 0.5
        a0    = 1.    * self.pc   / self.l0
        z = r * np.cos( theta )
        R = r * np.sin( theta ) 

        a = ( R**2 + z**2 / qb**2 )**0.5
        return ( rhob0 / ( 1 + ( a / a0 ) )**alpha ) * np.exp( -( a / acut )**2 )

    def disc_s19( self, r, theta ):
        """Disc density profile (Sormani 2019).

        Args:
            r, theta: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        sigmaunit = self.modot / self.pc**2
        sigma1    = 572.  * sigmaunit / self.sigma0
        Rd1       = 2900. * self.pc / self.l0
        z1        = 300.  * self.pc / self.l0
        sigma2    = 147.  * sigmaunit / self.sigma0
        Rd2       = 3310. * self.pc / self.l0
        z2        = 900.  * self.pc / self.l0

        z = r * np.cos( theta )
        R = r * np.sin( theta ) 
        return ( sigma1 / 2. / z1 ) * np.exp( -np.abs( z ) / z1 - R / Rd1 ) \
             + ( sigma2 / 2. / z2 ) * np.exp( -np.abs( z ) / z2 - R / Rd2 )

    def disc_m17( self, r, theta ):
        """Disc density profile (McMillan 2017).

        Args:
            r, theta: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        z = r * np.cos( theta )
        R = r * np.sin( theta )
        sigma = [ 2.2e9 * self.modot * ( 1000 * self.pc )**-2 / self.sigma0,
                  2.0e9 * self.modot * ( 1000 * self.pc )**-2 / self.sigma0 ]
        R_d   = [ 2600  * self.pc / self.l0,
                  2000  * self.pc / self.l0 ]
        R_cut = [ 3500  * self.pc / self.l0,
                  0 ]
        h     = [ 300   * self.pc / self.l0,
                  900   * self.pc / self.l0 ]
        rhodisk = 0
        for j in [0,1]:
            rhodisk += sigma[ j ]/ 2 / h[ j ] *\
                       np.exp( -R / R_d[ j ] -
                                R_cut[ j ] / R -
                                np.abs( z ) / h[ j ] )
        return rhodisk

    def disc_s22( self, r, theta ):
        """Disc density profile (Sormani 2022).

        Args:
            r, theta: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        z = r * np.cos( theta )
        R = r * np.sin( theta )
        sigma_0 = 0.103 * 10**10 * self.modot * ( 1000 * self.pc )**-2 / self.sigma0
        R_d = 4754   * self.pc / self.l0
        z_d = 151    * self.pc / self.l0
        R_cut = 4688 * self.pc / self.l0
        n_d = 1.536
        m_d = 0.716
        comp1 = sigma_0 / ( 4 * z_d )
        comp2 = np.exp( -( R / R_d )**n_d )
        comp3 = np.exp( -( R_cut / R ) )
        comp4 = np.cosh( np.abs( z ) / z_d )**-m_d
        return comp1 * comp2 * comp3 * comp4
    

    def nsd_s20( self, r, theta ):
        """Nuclear stellar disc density profile (Sormani 2020).

        Args:
            r, theta: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        rhounit = self.modot / ( 1000 * self.pc )**3

        q = 0.37
        n1 = 0.72
        n2 = 0.79
        R1 = 5.06 * self.pc / self.l0
        R2 = 24.6 * self.pc / self.l0
        rho_2 = 170. * 10**10 * rhounit / self.rho0
        rho_1 = 1.311 * rho_2

        z = r * np.cos( theta )
        R = r * np.sin( theta ) 

        a = ( R**2 + z**2/q**2 )**0.5

        return rho_1 * np.exp( -( a / R1 )**n1 ) + rho_2 * np.exp( -( a / R2 )**n2 )

    def nsc_s20( self, r, theta ):
        """Nuclear star cluster density profile (Sormani 2020).

        Args:
            r, theta: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        z = r * np.cos( theta )
        R = r * np.sin( theta ) 
        q = 0.73
        M_nsc = 6.1e7 * self.modot / self.m0
        a0    = 5.9   * self.pc / self.l0
        gamma = 0.71
        a = ( R**2 + z**2/q**2 )**0.5
        comp1 = ( 3 - gamma ) * M_nsc / ( 4 * np.pi * q )
        comp2 = a0 / ( a**gamma * ( a + a0 )**( 4 - gamma ) )
        return comp1 * comp2
    
    def bar_s19( self, r, theta, phi ):
        """Bar density profile (Sormani 2019).

        Args:
            r, theta, phi: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        rhob1 = 16. * self.rho1 / self.rho0
        rhob2 = 3.  * self.rho1 / self.rho0
        ab1 = 300.  * self.pc   / self.l0
        qb1 = 0.5
        ab2 = 1000. * self.pc   / self.l0
        qb2 = 0.5

        z = r * np.cos( theta )
        R = r * np.sin( theta ) 

        x = R * np.cos( phi )
        y = R * np.sin( phi )

        a1 = ( x**2 + ( y**2 + z**2 ) / qb1**2 )**0.5
        a2 = ( x**2 + ( y**2 + z**2 ) / qb2**2 )**0.5

        return rhob1 * np.exp( -a1 / ab1 ) + rhob2 * np.exp( -a2 / ab2 )


    def disc_h24( self, r, theta ):
        """Disc density profile (Hunter 2024).

        Args:
            r, theta: Spherical coordinates (code units).

        Returns:
            Density at the given point (code units).
        """
        z = r * np.cos( theta )
        R = r * np.sin( theta ) 

        sigma1 = 1.3719e3 * ( self.modot / self.m0 ) / ( self.pc / self.l0 )**2
        sigma2 = 9.2391e2 * ( self.modot / self.m0 ) / ( self.pc / self.l0 )**2

        Rd1 = 2e3 * self.pc / self.l0
        Rd2 = 2.8e3 * self.pc / self.l0

        z1  = 3e2 * self.pc / self.l0
        z2  = 9e2 * self.pc / self.l0

        Rcut = 2.4e3 * self.pc / self.l0

        return sigma1 / ( 2 * z1 ) * np.exp( - R / Rd1 - Rcut / R - np.abs( z ) / z1 ) +\
               sigma2 / ( 2 * z2 ) * np.exp( - R / Rd2 - Rcut / R - np.abs( z ) / z2 )



    def halo( self, r ):
        """NFW-like halo density profile.

        Args:
            r: Radial coordinate (code units).

        Returns:
            Density at the given point (code units).
        """
        x = r / self.r_halo
        return self.rho_halo / x / ( 1 + x )**2



