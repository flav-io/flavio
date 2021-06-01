#!/bin/bash

# Deploy the current source tree to PyPI.
# Install requirements:
# python3 -m pip install --user twine

cd "${0%/*}"  # make sure we are in the correct working dir
rm dist/*
python3 setup.py sdist
twine upload dist/*