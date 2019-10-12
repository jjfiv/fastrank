# FastRank [![Build Status](https://travis-ci.com/jjfiv/fastrank.svg?token=wqGZxUYsDSPaq1jz2zn6&branch=master)](https://travis-ci.com/jjfiv/fastrank) [![PyPI version](https://badge.fury.io/py/fastrank.svg)](https://badge.fury.io/py/fastrank)


My most frequently used learning-to-rank algorithms ported to rust for efficiency.

Read my [blog-post](https://jjfoley.me/2019/10/11/fastrank-alpha.html) announcing the first public version: 0.4. It's alpha because I think the API needs work, not because there's any sort of known correctness or compatiblity issues.

## Python Usage 

```shell
pip install fastrank
```

See this [Colab notebook](https://colab.research.google.com/drive/1IjF7yTin1XaNO_6mBNxAoQYTmF0nckk1) for more, or see a static version [here on Github](https://github.com/jjfiv/fastrank/blob/master/examples/FastRankDemo.ipynb).

## Code Structure

There are three subprojects here.

### fastrank 

The core algorithms and data structures are implemented in Rust. Formatted with rustfmt.

### cfastrank [![PyPI version](https://badge.fury.io/py/cfastrank.svg)](https://badge.fury.io/py/cfastrank)

A very thin layer of rust code provides a C-compatible API. A manylinux version is published to pypi. Don't install this manually -- install the ``fastrank`` package and let it be pulled in as a dependency.

### pyfastrank [![PyPI version](https://badge.fury.io/py/fastrank.svg)](https://badge.fury.io/py/fastrank)

A pure-python libary accesses the core algorithms using cffi via cfastrank. A version is published to pypi.
