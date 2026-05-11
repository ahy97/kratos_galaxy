"""
Additional utility functions
"""
import configparser
from unit import units
from data_field import data_field

def config_setup( filename ):
    """
    Load in the configuration file. All other class parameters will be derived
    from the configuration file. User keyword arguments will override the configuration file
    parameters
    
    :param filename: Path to configuration file
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read( filename )
    units.read_config( config )
    data_field.unit_system = units( )
    return config