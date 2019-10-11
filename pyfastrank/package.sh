#!/bin/bash
set -eu

rm -rf fastrank.egg-info
rm -rf dist

pip install -r requirements.txt
pip install --upgrade setuptools wheel twine
python setup.py sdist bdist_wheel
