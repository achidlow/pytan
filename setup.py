from pathlib import Path
from setuptools import setup, find_packages

_here = Path(__file__).parent

with open(_here / 'README.md', 'r') as f:
    long_description = f.read()

setup(
    name='pytan',
    version='0.1.0',
    description='scikit-learn compatible implementation of Tree Augmented Naïve Bayes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Adam Chidlow',
    author_email='adam@chidlow.net',
    license='Apache License 2.0',
    url='https://github.com/achidlow/pytan',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ]
)
