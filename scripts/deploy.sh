#!/bin/bash

# now on travis...
source venv/bin/activate

set -eu

cd cfastrank && maturin publish -b cffi -u __token__ -p PYPI_CFASTRANK_TOKEN && cd -
