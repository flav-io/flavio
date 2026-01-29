import importlib.resources
import os
from itertools import chain
import re
import ast


def get_datapath(package, resource):
    """Return the file path for a resource within a package."""
    return str(importlib.resources.files(package).joinpath(resource))

def extract_citations():
    string_in_parantheses_matcher = (
        r'\(\s*[rfuRFU]{0,2}".*?(?<!\\)"\s*\)' # string between (" ")
        '|'
        r"\(\s*[rfuRFU]{0,2}'.*?(?<!\\)'\s*\)" # string between (' ')
    )
    regexp = re.compile(fr'\.register({string_in_parantheses_matcher})')
    flavio_dir = get_datapath('flavio', '')
    generator_py_files = chain.from_iterable((
        (   os.path.join(root, name) for name in filenames
            if os.path.splitext(name)[1] == '.py')
        for root, dirs, filenames in os.walk(flavio_dir)
    ))
    citations = set()
    for filename in generator_py_files:
        with open(filename, 'r') as f:
            citations |= set(chain.from_iterable((
                {ast.literal_eval(v) for v in regexp.findall(line)} for line in f
            )))
    return citations
