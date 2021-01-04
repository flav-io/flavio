from setuptools import setup, find_packages

with open('flavio/_version.py', encoding='utf-8') as f:
    exec(f.read())

with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='flavio',
      version=__version__,
      author='David M. Straub',
      author_email='straub@protonmail.com',
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
                'physics/data/arXiv-1602-01399v1/*',
                'physics/data/arXiv-1811-00983v1/*',
                'physics/data/qcdf_interpolate/*',
                'physics/data/wcsm/*',
                ]
      },
      install_requires=['numpy>=1.16.5', 'scipy', 'setuptools>=3.3', 'pyyaml',
                        'ckmutil', 'wilson>=2.0', 'particle', ],
      extras_require={
            'testing': ['nose2'],
            'plotting': ['matplotlib>=2.0'],
            'sampling': ['iminuit>=2.0'],
            },
    )
