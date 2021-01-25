#!/usr/bin/env python

from setuptools import setup

setup(name='pipedream-solver',
      version='0.1.2',
      description='Interactive hydrodynamic solver for sewer/stormwater networks',
      author='Matt Bartos',
      author_email='mdbartos@umich.edu',
      url='https://mdbartos.github.io/pipedream',
      packages=["pipedream_solver"],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'numba',
          'matplotlib'
      ]
     )
