from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='heft',
    version='1.0.0',
    description='A Python 3.6+ Implementation of HEFT (Heterogeneous Earliest Finish Time) DAG Scheduling Algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mackncheesiest/heft',
    author='Joshua Mack',
    author_email='jmack2545@email.arizona.edu',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    packages=['heft'],
    install_requires=[
        'matplotlib',
        'numpy',
        'networkx'
    ],
    extras_require={
        'test': ['pytest']
    },
    project_urls={
        'Bug Reports': 'https://github.com/mackncheesiest/heft/issues',
        'Source': 'https://github.com/mackncheesiest/heft'
    }
)