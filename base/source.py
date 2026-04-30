from unit import units
import numpy as np
from inspect import signature
from data_field import data_field

class source( units ):
    def __init__( self, **kwargs ):
        """
        Initializes an instance of a source object. Either a config file must be loaded as a class attribute of
        "units", or keyword arguments must be specified
        
        :param \**kwargs: See below

        :Keyword Arguments:
            **func : Density function of source, capable of handling float or array-like arguments. Defaults to
                     a dummy function
            **dim  : Number of required positional arguments in the function. Defaults to len( signature( func ).parameters )
            **unit : Unit of the function, defaulting to density rho
        """
        self.dummy_flag = False
        if 'func' not in kwargs:
            self.dummy_flag = True
        self.dens = kwargs.get( 'func', self.dummy )
        self.dim  = kwargs.get( 'dim', len(signature(self.dens).parameters))
        self.unit = np.array( kwargs.get( 'unit', [ 1, -3, 0, 0 ] ) )
        return
    
    def src( self, *args ):
        args_nounit = [ i.to_code( ).data if isinstance( i, data_field ) else i for i in args ]
        return data_field( self.dens( *args_nounit ), self.unit )
    
    def src_cgs( self, *args ):
        args_code = [ i.to_code( ) for i in args ]
        return self.src( *args_code ).to_cgs( )

    def dummy( self, *args ):
        return

    def __add__( self, obj ):
        if self.dummy_flag:
            return obj
        elif obj.dummy_flag:
            return self
        
        if self.dim != obj.dim:
            raise IndexError( "Cannot add source functions of differing dimensions" )
        
        if not ( self.unit == obj.unit ).all( ):
            raise ArithmeticError( "Cannot add source functions of differing units" )
        
        def src_sum( *args ):
            return self.dens( *args ) + obj.dens( *args )
        
        return source( func = src_sum,  
                       dim  = obj.dim,
                       unit = self.unit )

    def __mul__( self, scalar ):
        if self.dummy_flag:
            return self
        
        def src_scaled( *args ):
            return scalar * self.dens( *args )
        
        return source( func = src_scaled,
                       dim  = self.dim,
                       unit = self.unit )