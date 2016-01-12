from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

setup(name='flavio',
      author='David M. Straub, Ece Gürler',
      author_email='david.straub@tum.de',
      packages=find_packages(),
      package_data={
      'flavio':['data/*'],
      'flavio.physics':['physics/data/arXiv-1503-05534v1/*','physics/data/arXiv-1501-00367v2/*']
      },
     )
