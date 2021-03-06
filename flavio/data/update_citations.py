#! /usr/bin/env python3

"""Standalone helper script to update the list of citations referenced in the
flavio source code."""

import argparse
import logging
import sys
import yaml
import pkgutil
logging.basicConfig(level=logging.INFO)

def main(argv):
    parser = argparse.ArgumentParser(description='Update the list of citations referenced in the flavio source code.')
    args = parser.parse_args()

    from flavio.util import get_datapath, extract_citations

    filename = get_datapath('flavio', 'data/citations.yml')
    with open(filename, 'w') as f:
        f.write(yaml.dump(sorted(extract_citations())))
        logging.info(f"Saved updated citations to {filename}")

if __name__ == '__main__':
    main(sys.argv)
