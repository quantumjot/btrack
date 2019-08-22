# !/usr/bin/env python

import os
import sys
import subprocess

from setuptools.command.install import install
from distutils.command.build import build
from distutils.core import setup

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
    if not os.path.exists(eigen_signature):
        # try to clone it
        clone_eigen = ["git", "clone", EIGEN_GIT, eigen_dst]
        if subprocess.call(clone_eigen) != 0:
            sys.exit(-1)

class BuildTracker(build):
    def run(self):
        # check for eigen
        check_eigen_installed()

        # run the build
        build.run(self)

        # run the build scripts
        build_sh = os.path.join(cwd, "btrack", "bin", "build_tracker.sh")
        os.system("chmod +x {}".format(build_sh))
        os.system("sh {}".format(build_sh))


# TODO(arl): finish this!
class InstallTracker(install):
    def run(self):
        install.run(self)
        # install btrack executables
        # self.copy_tree(self.build_lib, self.install_lib)
        # print self.build_lib, self.install_lib



setup(name='BayesianTracker',
      version='0.2.11',
      description='BayesianTracker is a simple Python/C++ based framework for multi-object tracking',
      author='Alan R. Lowe',
      author_email='a.lowe@ucl.ac.uk',
      url='https://github.com/quantumjot/BayesianTracker',
      packages=['btrack', 'btrack.optimise', 'btrack.libs'],
      package_data={'btrack': ['libs/libtracker.so', 'libs/libtracker.dylib']},
      install_requires=get_install_required(),
      license='LICENSE.md',
      cmdclass={'build': BuildTracker,
                'install': InstallTracker})
