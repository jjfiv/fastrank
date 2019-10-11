from setuptools import setup, find_packages
import os

pyfastrank_version = "0.4.1"
cfastrank_version = "0.4.0"

dependencies = ["cfastrank=={0}".format(cfastrank_version)]
dependencies.extend(open("requirements.txt").readlines())

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, '..', 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="fastrank",
    version=pyfastrank_version,
    author="John Foley",
    author_email="jjfoley@smith.edu",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    description="A set of learning-to-rank algorithms.",
    url="https://github.com/jjfiv/fastrank",
    packages=find_packages(),
    install_requires=dependencies,
    platforms="any",
)
