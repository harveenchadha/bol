from setuptools import setup
from pkg_resources import parse_requirements,run_script
import pathlib
import os
import setuptools


setup(name='bol-library',
      version='0.1.2',
      description='Speech to Text Library plus addons',
      url='https://github.com/harveenchadha/bol/',
      author='Harveen Singh Chadha',
      author_email='harveensinghchadha@gmail.com',
      license='MIT',
      # packages=['bol',
      #            'bol.inference',
      #            'bol.models',
      #            'bol.utils',
      #            'bol.metrics',
      #            'bol.data'],
#       scripts=['bin/install.sh'],
      platforms=["linux", "unix"],
      python_requires=">3.5.2",
      include_package_data=True,
      install_requires=[
        str(requirement) for requirement
            in parse_requirements(pathlib.Path('requirements.txt').open())
      ],
      packages=setuptools.find_packages(),
      
#       install_requires=[run_script(['fairseq'], 'bin/install.sh')],
      zip_safe=False)
