from setuptools import setup

version = '0.3.2'

setup(
    name='fastrank',
    version=version,
    author="John Foley",
    author_email="jjfoley@smith.edu",
    classifiers=[
        "Programming Language :: Python :: 3.5"
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="A set of learning-to-rank algorithms.",
    url="https://github.com/jjfiv/fastrank",
    packages=setuptools.find_packages(),
    install_requires=['cfastrank=={0}'.format(version)],
    platforms='any',
)
