from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rechelper',
    version='0.0.1',
    description='Utilities for building recommenders',
    long_description=readme,
    author='Aki Saarinen',
    author_email='aki.saarinen@gmail.com',
    url='https://github.com/akisaarinen/rechelper',
    license=license,
    packages=find_packages(exclude=('data', 'tests'))
)