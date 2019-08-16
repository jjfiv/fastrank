#!/bin/bash

#pip3 install --user --upgrade pip
#pip3 install --user pyo3-pack

set -eu

cd cfastrank && pyo3-pack publish -b cffi -u __token__ -p PYPI_CFASTRANK_TOKEN && cd -
