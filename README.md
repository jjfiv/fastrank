# FastRank [![Build Status](https://travis-ci.com/jjfiv/fastrank.svg?token=wqGZxUYsDSPaq1jz2zn6&branch=master)](https://travis-ci.com/jjfiv/fastrank)

My most frequently used learning-to-rank algorithms ported to rust for efficiency.

## Python Usage

```shell
pip install fastrank
```

## Code Structure

### fastrank

The core algorithms and data structures are implemented in Rust.

### cfastrank [![PyPI version](https://badge.fury.io/py/cfastrank.svg)](https://badge.fury.io/py/cfastrank)

A very thin layer of rust code provides a C-compatible API. A manylinux version is published to pypi.

### pyfastrank

A pure-python libary accesses the core algorithms using cffi via cfastrank. A version is published to pypi.
