from setuptools import setup

setup(name='bol',
      version='0.1',
      description='Speech to Text Library',
      url='https://github.com/harveenchadha/bol/',
      author='Harveen Singh Chadha',
      author_email='harveen54@gmail.com',
      license='MIT',
      packages=['bol'],
      dependency_links=['https://github.com/Open-Speech-EkStep/fairseq.git#egg=fairseq']
      install_requires=['pypi-kenlm'],
      zip_safe=False)
