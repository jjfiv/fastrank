#!/bin/bash

set -eu

pip3 install --user --upgrade pip
pip3 install --user pyo3-pack

cd cfastrank && pyo3-pack publish -b cffi -u @token -p PYPI_CFASTRANK_TOKEN && cd -

