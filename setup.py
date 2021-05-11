from distutils.core import setup

setup(
    name='curlyBrace',
    version='1.0',
    description='matplotlib curly brace support',
    author='Dr. Gao, Siyu',
    license='MIT',
    url='https://github.com/jhultman/matplotlib-curly-brace',
    packages=['curlyBrace'],
    python_requires='>=3.5',
    requires=['numpy', 'matplotlib'],
)
