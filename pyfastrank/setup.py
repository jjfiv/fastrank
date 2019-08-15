from setuptools import setup

version = '0.2.0'

setup(
    name='fastrank',
    version=version,
    author="John Foley",
    author_email="jjfoley@smith.edu",
    classifiers=[
        "Programming Language :: Python :: 3.5"
    ],
    description="A set of learning-to-rank algorithms.",
    url="https://github.com/jjfiv/fastrank",
    packages=['fastrank'],
    install_requires=['cfastrank=={0}'.format(version)],
    platforms='any',
)
