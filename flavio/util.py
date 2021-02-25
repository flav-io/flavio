import pkgutil
import os
import sys
from itertools import chain
import re
import ast


def get_datapath(package, resource):
    """Rewrite of pkgutil.get_data() that just returns the file path.

    Taken from https://stackoverflow.com/a/13773912"""
    loader = pkgutil.get_loader(package)
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    mod = sys.modules.get(package) or loader.load_module(package)
    if mod is None or not hasattr(mod, '__file__'):
        return None
    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.join(*parts)
    return resource_name

def extract_citations():
    string_in_parantheses_matcher = (
        r'\(\s*[rfuRFU]{0,2}".*?(?<!\\)"\s*\)' # string between (" ")
        '|'
        r"\(\s*[rfuRFU]{0,2}'.*?(?<!\\)'\s*\)" # string between (' ')
    )
    regexp = re.compile(fr'\.register({string_in_parantheses_matcher})')
    flavio_dir = get_datapath('flavio', '')
    generator_py_files = chain.from_iterable((
        (   os.path.join(root, name) for name in files
            if os.path.splitext(name)[1] == '.py')
        for root, dirs, files in os.walk(flavio_dir)
    ))
    citations = set()
    for filename in generator_py_files:
        with open(filename, 'r') as f:
            citations |= set(chain.from_iterable((
                {ast.literal_eval(v) for v in regexp.findall(line)} for line in f
            )))
    return citations
