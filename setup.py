from setuptools import setup

setup(
    name='pyrofit-utils',
    description='General utilities for torch and pyro.',
    version='0.1',
    packages=['pyrofit.utils'],
    install_requires=[
        'torch',
        'pykeops',
        'pyro-ppl'
    ],
)
