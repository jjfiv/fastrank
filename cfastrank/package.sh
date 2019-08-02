#!/bin/bash

set -eu

pip install pyo3-pack
pyo3-pack -b cffi -u @token -p PYPI_CFASTRANK_TOKEN
