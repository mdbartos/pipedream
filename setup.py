#!/usr/bin/env python

from setuptools import setup

setup(name='superlink',
      version='0.1',
      description='Implementation of superlink hydraulic solver',
      author='Matt Bartos',
      author_email='mdbartos@umich.edu',
      url='http://open-storm.org',
      packages=["superlink"],
      install_requires=[
          'numpy',
          'pandas',
          'scipy'
      ]
     )
