# FastRank ![CI Status Badge](https://github.com/jjfiv/fastrank/workflows/CI/badge.svg) [![PyPI version](https://badge.fury.io/py/fastrank.svg)](https://badge.fury.io/py/fastrank)

My most frequently used learning-to-rank algorithms ported to rust for efficiency.

Read my [blog-post](https://jjfoley.me/2019/10/11/fastrank-alpha.html) announcing the first public version: 0.4. It's alpha because I think the API needs work, not because there's any sort of known correctness or compatiblity issues.

## Python Requirement

 - 0.5 and earlier require only Python 3.5, but no windows builds were pushed.
 - 0.6 requires Python 3.6 due to EOL for Python 3.5 becoming prevalent in the latest pip.
 - 0.7 and forward will require Python 3.7 so we can use the standard @dataclass annotation and drop the attrs dependency.

## Python Usage 

```shell
pip install fastrank
```

See this [Colab notebook](https://colab.research.google.com/drive/1IjF7yTin1XaNO_6mBNxAoQYTmF0nckk1) for more, or see a static version [here on Github](https://github.com/jjfiv/fastrank/blob/master/examples/FastRankDemo.ipynb).

