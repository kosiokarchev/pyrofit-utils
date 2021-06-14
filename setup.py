from setuptools import setup

setup(
    name='pyrofit-utils',
    description='General utilities for torch and pyro.',
    version='0.1',
    packages=['pyrofit.utils'],
    install_requires=[
        'packaging',
        'torch>=1.8',
        'pykeops',
        'pyro-ppl'
    ],
)
