#!/bin/bash

#pip3 install --user --upgrade pip
#pip3 install --user maturin

set -eu

cd cfastrank && maturin publish -b cffi -u __token__ -p $PYPI_CFASTRANK_TOKEN && cd -

