#!/usr/bin/env python

from setuptools import setup

setup(name='pipedream-solver',
      version='0.1',
      description='Interactive hydrodynamic solver for sewer/stormwater networks',
      author='Matt Bartos',
      author_email='mdbartos@umich.edu',
      url='https://mattbartos.com',
      packages=["pipedream_solver"],
      install_requires=[
          'numpy',
          'pandas',
          'scipy'
      ]
     )
