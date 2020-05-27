from setuptools import setup
from setuptools import find_packages
# from setuptools.command.install import install

def get_install_required():
    with open("./requirements.txt", "r") as reqs:
        requirements = reqs.readlines()
    return [r.rstrip() for r in requirements]

def get_version():
    with open("./btrack/VERSION.txt" ,"r") as ver:
        version = ver.readline()
    return version.rstrip()

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='btrack',
      version=get_version(),
      description='BayesianTracker is a simple Python/C++ based framework for multi-object tracking',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Alan R. Lowe',
      author_email='a.lowe@ucl.ac.uk',
      url='https://github.com/quantumjot/BayesianTracker',
      packages=find_packages(),
      package_data={'btrack': ['libs/libtracker*', 'VERSION.txt']},
      install_requires=get_install_required(),
      python_requires='>=3.6',
      license='LICENSE.md',
      classifiers=['Topic :: Scientific/Engineering'])
