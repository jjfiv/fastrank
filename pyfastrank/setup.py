from setuptools import setup, find_packages

pyfastrank_version = "0.4.0"
cfastrank_version = "0.4.0"

dependencies = ["cfastrank=={0}".format(cfastrank_version)]
dependencies.extend(open("requirements.txt").readlines())

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
    description="A set of learning-to-rank algorithms.",
    url="https://github.com/jjfiv/fastrank",
    packages=find_packages(),
    install_requires=dependencies,
    platforms="any",
)
