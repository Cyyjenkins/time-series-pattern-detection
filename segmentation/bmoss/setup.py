#!/usr/bin/python

from setuptools import setup

setup(
    name='change_detec',
    version='1.0',
    author='Gabriel Agamennoni',
    author_email='g.agamennoni@gmail.com',
    description=('Bayesian change detection for input-output sequence data.'),
    py_modules=['change_detec'],
    install_requires=['numpy']
)
