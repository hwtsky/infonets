# setup.py
"""
Created on Tue Jun 26 17:07:11 2018

@author: Wentao Huang
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'numpy',
    'matplotlib',
    'torch',
    'torchvision'
    ]

setup(
    name="infonets",
    version="0.0.1",
    author='Wentao Huang',
    author_email='wnthuang@gmail.com',
    url='https://github.com/hwtsky/infonets',
    long_description=long_description,
    license='BSD',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
    )