# !/usr/bin/env python

import os
import sys
import subprocess

from setuptools import setup
from setuptools.command.install import install
# from setuptools.command.build import build


cwd = os.getcwd()

EIGEN_GIT = "https://github.com/eigenteam/eigen-git-mirror.git"

def get_install_required():
    with open("./requirements.txt", "r") as reqs:
        requirements = reqs.readlines()
    return [r.rstrip() for r in requirements]

def check_eigen_installed():
    sig_file = "signature_of_eigen3_matrix_library"
    eigen_dst = os.path.join(cwd, "btrack", "include", "eigen")
    eigen_signature = os.path.join(eigen_dst, sig_file)
    # if not os.path.exists(eigen_signature):
    #     # try to clone it
    #     clone_eigen = ["git", "clone", EIGEN_GIT, eigen_dst]
    #     if subprocess.call(clone_eigen) != 0:
    #         sys.exit(-1)

class BuildTracker(install):
    def run(self):
        # check for eigen
        check_eigen_installed()

        # make the local libs and obj folder
        if not os.path.exists('./btrack/libs'):
            os.makedirs('./btrack/libs')
        if not os.path.exists('./btrack/obj'):
            os.makedirs('./btrack/obj')

        # run the build scripts
        make_BT = ["make"]
        if subprocess.call(make_BT) != 0:
            sys.exit(-1)

        # run the build
        install.run(self)


setup(name='BayesianTracker',
      version='0.3.0',
      description='BayesianTracker is a simple Python/C++ based framework for multi-object tracking',
      author='Alan R. Lowe',
      author_email='a.lowe@ucl.ac.uk',
      url='https://github.com/quantumjot/BayesianTracker',
      packages=['btrack', 'btrack.optimise', 'btrack.libs'],
      package_data={'btrack': ['libs/libtracker.so', 'libs/libtracker.dylib']},
      install_requires=get_install_required(),
      license='LICENSE.md',
      cmdclass={'install': BuildTracker})
