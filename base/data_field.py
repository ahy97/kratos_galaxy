import numpy as np
import copy 

class data_field:
    __array_priority__ = 10000
    unit_system = None
    @property
    def l0( self ):
        return self.unit_system.l0

    @property
    def m0( self ):
        return self.unit_system.m0

    @property
    def t0( self ):
        return self.unit_system.t0
    
    def __init__( self, data, unit=[ 0, 0, 0, 0 ], flag="code" ):
        # unit - mass, length, time, temp
        if isinstance( data, data_field ):
            self.data = data.data
            self.unit = data.unit
            self.flag = data.flag
            return
        else:
            self.data = data
            if isinstance( unit, str ):
                basic_units = { "m"   : np.array( [  1,  0,  0,  0 ] ), 
                                "l"   : np.array( [  0,  1,  0,  0 ] ), 
                                "t"   : np.array( [  0,  0,  1,  0 ] ), 
                                "T"   : np.array( [  0,  0,  0,  1 ] ),
                                "rho" : np.array( [  1, -3,  0,  0 ] ),
                                "v"   : np.array( [  0,  1, -1,  0 ] ),
                                "ene" : np.array( [  1,  2, -2,  0 ] ) }
                unit_comps = unit.split( "*" )
                self.unit = np.zeros( 4 )
                for i in unit_comps:
                    unit_base = i
                    unit_pow  = 1
                    if "^" in i:
                        unit_pow  = float( i.split( "^" )[ -1 ] )
                        unit_base = i.split( "^" )[ 0 ]
                    self.unit += basic_units[ unit_base ] * unit_pow

                #self.unit = basic_units[ unit ]
            else:
                self.unit = np.array( unit )
        datashape = ( data.shape if isinstance( data, np.ndarray )
                      else ( len( data ), ) if isinstance( data, ( list, tuple ) )
                      else ( 1, ) )
        if len( self.unit ) not in datashape and self.unit.ndim > 1:
            raise IndexError( "Number of units does not match any axis of data" )
        self.flag = flag
        if flag != "code" and flag != "cgs":
            raise AttributeError( f"Invalid unit system flag {flag}" )
        return
    def to_cgs( self, convert=False ):
        # Returns data field in cgs. convert = True also converts the underlying data_field 
        if self.flag == "cgs":
            return self
        newdata = copy.deepcopy( self.data )
        if isinstance( self.data, np.ndarray ):
            newdata *= self.m0**self.unit[ 0 ]
            newdata *= self.l0**self.unit[ 1 ]
            newdata *= self.t0**self.unit[ 2 ]
        elif isinstance( self.data, ( list, tuple ) ):
            dtype = type( self.data )
            mod_data = [ i * self.m0**self.unit[ 0 ] *
                             self.l0**self.unit[ 1 ] *
                             self.t0**self.unit[ 2 ] for i in self.data ]
            newdata = dtype( mod_data )
        if convert:
            self.flag = 'cgs'
            self.data = newdata
            return self
        else:
            return data_field( newdata, self.unit, 'cgs' )
    
    def to_code( self, convert=False ):
        # Returns data field in code unit. convert = True also converts the underlying data_field 
        if self.flag == "code":
            return self
        newdata = copy.deepcopy( self.data )
        if isinstance( self.data, np.ndarray ):
            newdata /= self.m0**self.unit[ 0 ]
            newdata /= self.l0**self.unit[ 1 ]
            newdata /= self.t0**self.unit[ 2 ]
        elif isinstance( self.data, ( list, tuple ) ):
            dtype = type( self.data )
            mod_data = [ i / self.m0**self.unit[ 0 ] /
                             self.l0**self.unit[ 1 ] /
                             self.t0**self.unit[ 2 ] for i in self.data ]
            newdata = dtype( mod_data )
        if convert:
            self.flag = 'code'
            self.data = newdata
            return self
        else:
            return data_field( newdata, self.unit, 'code' )
        
    def zero_out( self ):
        if isinstance( self.data, np.ndarray ):
            self.data.fill( 0 )
        elif isinstance( self.data, list ):
            for i in range( len( self.data ) ):
                self.data[ i ] = 0
        else:
            self.data = 0
        return

    def __call__( self ):
        return self.data
    def __iter__( self ):
        if isinstance( self.data, ( np.ndarray, list, tuple ) ):
            return iter( [ data_field( a, self.unit, self.flag ) for a in self.data ] )
        else:
            return iter( ( self.data, ) )

    def __getattr__( self, name ):
        # delegate attribute calls to underlying data
        return getattr( self.data, name )

    def __getitem__( self, key ):
        return data_field( self.data[ key ], self.unit, self.flag )

    def __setitem__( self, key, value ):
        self.data[ key ] = value
        return
    def __len__( self ):
        if isinstance( self.data, ( np.ndarray, list, tuple ) ):
            return len( self.data )
        else:
            return 1
    # When the object is used in a numeric context, behave like the data
    def __float__( self ):
        return float( self.data )

    def __int__( self ):
        return int( self.data )
    
    def __repr__( self ):
        return f"{ self.data } [ { self.unit } ]"

    def __array__( self, dtype=None ):
        return np.asarray( self.data, dtype=dtype )

    def __neg__(self):
        return data_field( -self.data, self.unit, self.flag )
    
    def __array_ufunc__( self, ufunc, method, *inputs, **kwargs ):
        if method not in ("__call__",):
            return NotImplemented
        PRESERVE = {
        np.negative,
        np.absolute,
        np.fabs,
        np.floor,
        np.ceil,
        np.trunc }
        
        REQUIRE_SAME = {
        np.add,
        np.subtract,
        np.maximum,
        np.minimum }

        MULTIPLY = { np.multiply }

        DIVIDE   = { np.divide, np.true_divide }

        POWER = {
            np.power,
            np.sqrt,
            np.square,
            np.cbrt
        }

        DIMENSIONLESS_ONLY = {
            np.exp, np.expm1,
            np.log, np.log10, np.log2, np.log1p,
            np.sin, np.cos, np.tan,
            np.sinh, np.cosh, np.tanh,
            np.arcsin, np.arccos, np.arctan
        }

        COMPARISONS = {
            np.less, np.less_equal,
            np.greater, np.greater_equal,
            np.equal, np.not_equal
        }        
        # Convert MyObj -> internal .data for all inputs
        datas = []
        units = []
        for x in inputs:
            if isinstance( x, data_field ):
                datas.append( x.data )
                units.append( x.unit )
            else:
                datas.append( x )
                units.append( np.zeros( 4 ) )   
        # Apply the actual ufunc
        result = getattr( ufunc, method )( *datas, **kwargs )   
        if ufunc in PRESERVE:
            new_unit = units[ 0 ]

        elif ufunc in REQUIRE_SAME:
            if not np.array_equal( units[ 0 ], units[ 1 ] ):
                raise ArithmeticError("Unit mismatch")
            new_unit = units[ 0 ]

        elif ufunc in MULTIPLY:
            new_unit = units[ 0 ] + units[ 1 ]

        elif ufunc in DIVIDE:
            new_unit = units[ 0 ] - units[ 1 ]

        elif ufunc in POWER:
            if ufunc is np.sqrt:
                new_unit = units[ 0 ] // 2
            elif ufunc is np.square:
                new_unit = units[ 0 ] * 2
            elif ufunc is np.cbrt:
                new_unit = units[ 0 ] // 3
            elif ufunc is np.power:
                if units[ 1 ].any():
                    raise ArithmeticError("Exponent must be unitless")
                new_unit = units[ 0 ] * datas[ 1 ]

        elif ufunc in DIMENSIONLESS_ONLY:
            if units[ 0 ].any( ):
                raise ArithmeticError(f"{ ufunc.__name__ } requires dimensionless input")
            new_unit = np.zeros( 4, dtype=int )

        elif ufunc in COMPARISONS:
            new_unit = np.zeros( 4, dtype=int )

        else:
            return NotImplemented
        # Wrap result back into class
        return data_field( result, new_unit, self.flag )
    def __array_function__( self, func, types, args, kwargs ):
        unwrapped_args = [ ]
        units = [ ]
        flags = [ ]
        for a in args:
            if isinstance( a, data_field ):
                unwrapped_args.append( a.data )
                units.append( a.unit )
                flags.append( a.flag )
            elif isinstance( a, tuple ) or isinstance( a, list ):
                atype = type( a )
                a_ = [ b.data if isinstance( b, data_field ) else b for b in a ]
                u_ = [ b.unit if isinstance( b, data_field ) else np.zeros( 4 ) for b in a ]
                f_ = [ b.flag if isinstance( b, data_field ) else "code" for b in a ]
                unwrapped_args.append( atype( a_ ) )
                units.append( atype( u_ ) )
                flags.append( atype( f_ ) )
            else:
                unwrapped_args.append( a )
                units.append( np.zeros( 4 ) )
                flags.append( "code" )
        result = func( *unwrapped_args, **kwargs )
        if isinstance( result, np.ndarray ):
            return data_field( result, self.unit, self.flag )
        elif isinstance( result, ( list, tuple ) ) and len( result ) == len( args ):
            restype = type( result )
            return_result = []
            for i, j, k in zip( result, units, flags ):
                return_result.append( data_field( i, j, k ) )
            return restype( return_result )
        return result#data_field( result, self.unit )
    # Basic arithmetic operations
    def _binary_op( self, obj, op, same ):
        if not isinstance( obj, data_field ):
            obj = data_field( obj, [ 0, 0, 0, 0 ] )
        if self.flag != obj.flag:
            raise ArithmeticError( "Invalid operation between code and cgs units" )
        new_data = op( self.data, obj.data )
        if same:
            if ( self.unit == obj.unit ).all( ):
                new_unit = self.unit
            else:
                print( self, obj )
                raise ArithmeticError( "Invalid operation for given units ")
        else:
            new_unit = np.log10( op( 10.**self.unit, 10.**obj.unit ) )
            new_unit = new_unit.astype( int )
        # optionally: check unit consistency
        return data_field( new_data, new_unit, self.flag )


    def __add__( self, obj )      : return self._binary_op( obj, lambda a, b : a + b , True )
    def __sub__( self, obj )      : return self._binary_op( obj, lambda a, b : a - b , True )
    def __mul__( self, obj )      : return self._binary_op( obj, lambda a, b : a * b , False )
    def __truediv__( self, obj )  : return self._binary_op( obj, lambda a, b : a / b , False )
    def __radd__( self, obj )     : return self.__add__( obj )#data_field ( obj + self.data , self.unit )
    def __rsub__( self, obj )     : return self.__sub__( obj )#data_field ( obj - self.data , self.unit )
    def __rmul__( self, obj )     : return self.__mul__( obj )#data_field ( obj * self.data , self.unit )
    def __rtruediv__( self, obj ) : return self._binary_op( obj, lambda a, b : b / a, False )#data_field ( obj / self.data , self.unit )

    #Special pow operator
    def __pow__( self, obj ):
        if isinstance( obj, data_field ): 
            if obj.unit.any( ):
                raise ArithmeticError( "Exponent must be unitless" )
            else:
                return data_field( self.data**obj.data, self.unit*obj.data, self.flag )
        else:
            return data_field( self.data**obj, self.unit*obj, self.flag )
    def __rpow__( self, obj ):
        if isinstance( obj, data_field ):
            return obj.__pow__( self )
        elif self.unit.any( ):
            raise ArithmeticError( "Exponent must be unitless" )
        else:
            return obj**self.data
    
    # Booleans
    def _comparison_op( self, obj, op ):
        comp1 = self.data if isinstance( self, data_field ) else self
        comp2 = obj .data if isinstance( obj , data_field ) else obj
        unit1 = self.unit if isinstance( self, data_field ) else np.zeros( 4 ) 
        unit2 = obj .unit if isinstance( obj , data_field ) else np.zeros( 4 )
        if not ( unit1 == unit2 ).all( ) and unit1.any( ) and unit2.any( ):
            raise TypeError( "Cannot perform boolean comparison between data fields with different non-zero units" )
        else:
            return op( comp1, comp2 )
    
    def __lt__( self, obj ): return self._comparison_op( obj, lambda a, b: a <  b )
    def __gt__( self, obj ): return self._comparison_op( obj, lambda a, b: a >  b )
    def __le__( self, obj ): return self._comparison_op( obj, lambda a, b: a <= b ) 
    def __ge__( self, obj ): return self._comparison_op( obj, lambda a, b: a >= b )
    def __eq__( self, obj ): return self._comparison_op( obj, lambda a, b: a == b )
    def __ne__( self, obj ): return self._comparison_op( obj, lambda a, b: a != b )


