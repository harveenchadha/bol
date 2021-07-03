from setuptools import setup
from pkg_resources import parse_requirements,run_script
import pathlib
import os


setup(name='bol',
      version='0.1',
      description='Speech to Text Library',
      url='https://github.com/harveenchadha/bol/',
      author='Harveen Singh Chadha',
      author_email='harveen54@gmail.com',
      license='MIT',
      packages=['bol'],
#       scripts=['bin/install.sh'],
      platforms=["linux", "unix"],
      python_requires=">3.5.2",
      include_package_data=True,
#       install_requires=[
#         str(requirement) for requirement
#             in parse_requirements(pathlib.Path('requirements.txt').open())
#       ],
      
      install_requires=[run_script(['fairseq'], 'bin/install.sh')],
      zip_safe=False)
