from numpy import array, frombuffer

############################################################
# Basic binary output reader

class binary_io:
    ########################################################
    # Initialization and finalization
    def __init__( self,    file_name, cache_used = True ):
        """Initialize binary I/O handler.

        Args:
            file_name (str): Path to the binary file.
            cache_used (bool): Whether to cache read data in memory.
        """
        self. file_name =  file_name;
        self.    __dmap =  dict(   );
        self.      hmap =  dict(   );
        self.cache_used = cache_used;
        self.    endian =   'little';
        self.    offset =          0;
        self.    stream =       None;
    #
    def set_stream( self, stream ):
        """Set the binary stream directly (used for merging).

        Args:
            stream: A file-like binary stream.
        """
        self.stream     = stream  ;
    #
    def get_size_t( self, bin_data = None ):
        """Read a size_t-width integer from the stream.

        Args:
            bin_data (bytes, optional): Pre-read bytes, otherwise reads from stream.

        Returns:
            int.
        """
        if  bin_data is None:
            bin_data =  self.stream.read( self.s_size_t );
        return int.from_bytes ( bin_data, self.  endian );
    #
    def get_char_t( self, bin_data = None ):
        """Read a single byte from the stream as an integer.

        Args:
            bin_data (bytes, optional): Pre-read byte, otherwise reads from stream.

        Returns:
            int (0-255).
        """
        if  bin_data is None:
            bin_data =  self.stream.read            ( 1 );
        return int.from_bytes ( bin_data, self.  endian );
    #
    def open( self ):
        """Open the binary file for reading and parse the header map."""
        if self.stream is not None:
            return;
        #
        self.stream   = open( self.file_name, 'rb' );
        self.stream   . seek( 0, 0 );
        # First byte stores: ( is_le | sizeof( size_t ) )
        # Note: one byte does not care about endian
        sst           = int.from_bytes\
                      ( self.stream.read( 1 ), 'little' );
        self.endian   = "little" if sst & 1  else "big";
        self.s_size_t = sst & ( ~ 1 );
        self.s_hdr    = self.get_size_t(  );
        self.s_hmap   = self.get_size_t(  );
        for i_hmap in range( self.s_hmap  ):
            s_kstr = self.get_size_t(  );
            key    = self.stream.read( s_kstr );
            size   = self.get_size_t(  );
            u_size = self.get_char_t(  );
            offset = self.get_size_t(  ) + self.s_hdr;
            self.hmap[ key.decode( 'ascii' ) ]\
                   = ( size, u_size,  offset );
        #
        return;
    #
    def close( self ):
        """Close the binary file stream if open."""
        if  self.stream is not None:
            self.stream.close(  );
            self.stream =    None;
        return;
    #
    def __enter__( self ):
        """Context manager entry: open the file.

        Returns:
            self.
        """
        self.open(  );
        return self;
    #
    
    def __exit__ ( self, etype, evalue, traceback ):
        """Context manager exit: close the file and print any exception info.

        Args:
            etype: Exception type (or None).
            evalue: Exception value (or None).
            traceback: Traceback object (or None).
        """
        self.close(  );
        if etype is not None:
            print( etype, evalue, traceback );
        #
    #
    ########################################################
    # Data access
    def __getitem__  ( self,    key ):
        """Retrieve raw binary data for a given key.

        Args:
            key (str): Data key from the header map.

        Returns:
            bytes of the stored binary data.
        """
        if  key in self.__dmap:
            return self.__dmap[ key ];
        #
        size, u_size, offset = self.hmap[ key ];
        self.stream.seek( offset, 0 );
        bin_data     = self.stream.read( size );
        if  self.cache_used:
            self.__dmap[ key ] = bin_data;
        #
        return bin_data;
    #
    def read( self ):
        """Read all data from the file into the memory cache."""
        for key in self.hmap:
            size, u_size, offset = self.hmap     [  key ];
            self.stream.seek   ( offset, 0 );
            self.__dmap[ key ] = self.stream.read( size );
        #
    #
    def as_array( self, key, dtype  = 'f' ):
        """Interpret cached binary data as a numpy array.

        Args:
            key (str): Data key.
            dtype (str): numpy dtype character (default 'f' for float32).

        Returns:
            ndarray.
        """
        u_size  = self.hmap[ key ][ 1 ];
        return  frombuffer( self[ key ], dtype = \
                            '<%s%d' %  ( dtype, u_size ) );
    #    
    ########################################################
    # Write data to file
    def cache( self, key, data, dtype = None ):
        """Store data in the memory cache for later writing.

        Args:
            key (str): Data key.
            data: Array-like data to store.
            dtype: numpy dtype for serialization.
        """
        dat   = array( data, dtype = dtype ).flatten(  );
        usize = len( dat[ 0 ].tobytes(  ) );
        size  = dat.size * usize;
        self.  hmap[ key ] = [ size, usize, self.offset ];
        self.__dmap[ key ] = dat.tobytes(  );
        self.     offset  += size;
        return;
    #
    def merge( self, src, skip_keys = [  ] ):
        """Merge data from another binary_io instance into this one.

        Args:
            src (binary_io or str): Source instance or filename to merge from.
            skip_keys (list): Keys to skip when merging.
        """
        if  isinstance( src, str ):
            src = binary_io( src );
        if  src.stream is None:
            src.open(  );
            src.read(  );
        #
        offset = self.offset;
        for key, d in src .hmap.items(  ):
            if key in self.hmap or key in skip_keys:
                print( "Skipping key " + key );
                continue;
            #
            self.  hmap[ key ] = ( d[ 0 ] , d[ 1 ], offset )
            self.__dmap[ key ] = src.__dmap[ key ];
            offset += d[ 0 ] ;
        #
        self.offset = offset;
        return;
    #
    def save( self ):
        """Write all cached data to the binary file, building the header map."""
        self.stream = open( self.file_name, 'wb' );
        self.stream . seek( 0, 0 );

        self.endian   = 'little';
        self.s_size_t = 8;
        def i2b( i, length = self.s_size_t ):
            return int.to_bytes( i, byteorder = self.endian,
                                 length = length );
        #        
        hdr = i2b( self.s_size_t + \
                   ( self.endian == 'little' ), 1 );
        # "+ 1" for the little endian

        skip_map = i2b( len( self.__dmap ) );
        for key in self.hmap:
            size, usize, offset = self.hmap[ key ];
            k_bin     = key.encode( 'ascii' );
            skip_map += i2b( len( k_bin ) ) + k_bin \
            + i2b( size ) + i2b( usize, 1 ) + i2b( offset );
        #
        hdr += i2b( len( skip_map ) + 1 + self.s_size_t );
        hdr += skip_map;
        self.stream.write( hdr );
        for key in self.hmap:
            self.stream.write( self.__dmap[ key ] );
        self.stream.close(  );
        return;
    #
#
