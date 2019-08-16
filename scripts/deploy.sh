#!/bin/bash

# now on travis...
source venv/bin/activate

set -eu

cd cfastrank && pyo3-pack publish -b cffi -u __token__ -p PYPI_CFASTRANK_TOKEN && cd -
