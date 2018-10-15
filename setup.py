from setuptools import setup, find_packages

with open('flavio/_version.py', encoding='utf-8') as f:
    exec(f.read())

with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='flavio',
      version=__version__,
      author='David M. Straub',
      author_email='david.straub@tum.de',
      url='https://flav-io.github.io',
      description='A Python package for flavour physics phenomenology in the Standard Model and beyond',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      license='MIT',
      packages=find_packages(),
      package_data={
      'flavio':['data/*.yml',
                'data/test/*',
                'physics/data/arXiv-0810-4077v3/*',
                'physics/data/arXiv-1503-05534v1/*',
                'physics/data/arXiv-1503-05534v2/*',
                'physics/data/arXiv-1501-00367v2/*',
                'physics/data/arXiv-1602-01399v1/*',
                'physics/data/pdg/*',
                'physics/data/qcdf_interpolate/*',
                'physics/data/wcsm/*',
                ]
      },
      install_requires=['numpy', 'scipy>=0.18', 'setuptools>=3.3', 'pyyaml',
                        'wcxf>=1.4.4', 'ckmutil', 'wilson>=1.3.1', ],
      extras_require={
            'testing': ['nose'],
            'plotting': ['matplotlib>=1.4'],
            'sampling': ['pypmc>=1.1', 'emcee', 'iminuit',],
            },
    )
