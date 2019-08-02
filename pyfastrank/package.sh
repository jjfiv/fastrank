#!/bin/bash
set -eu
pip install -r requirements.txt
pip install --upgrade setuptools wheel twine
python setup.py sdist bdist_wheel
