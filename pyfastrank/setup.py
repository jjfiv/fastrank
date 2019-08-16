from setuptools import setup, find_packages

pyfastrank_version = '0.3.1'
cfastrank_version = '0.3.0'

setup(
    name='fastrank',
    version=pyfastrank_version,
    author="John Foley",
    author_email="jjfoley@smith.edu",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="A set of learning-to-rank algorithms.",
    url="https://github.com/jjfiv/fastrank",
    packages=find_packages(),
    install_requires=['cfastrank=={0}'.format(cfastrank_version), "attrs", "cffi", "numpy", "scikit-learn", "ujson"],
    platforms='any',
)
