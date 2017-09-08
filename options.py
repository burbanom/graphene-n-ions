import yaml
import re

def read_options( filename ):
    """Reads in the options file in YAML format

    Args:
        filename (str): filename of the options file

    Returns:
        config  (dict): potential fitting options, as a dictionary

    Notes:
        fixes the YAML resolver for scientific notation: http://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number    
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    config = yaml.load( open( filename ), Loader = loader )
    return config    

