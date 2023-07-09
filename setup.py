#!/usr/bin/env python

from setuptools import setup

setup(name='pipedream-solver',
      version='0.2.2',
      description='🚰 Interactive hydrodynamic solver for pipe and channel networks',
      long_description="🚰 Interactive hydrodynamic solver for pipe and channel networks",
      long_description_content_type="text/x-rst",
      author='Matt Bartos',
      author_email='mdbartos@utexas.edu',
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
