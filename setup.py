# !/usr/bin/env python

# import os
# import sys
# import subprocess

from setuptools import setup
# from setuptools.command.install import install

# cwd = os.getcwd()
#
# EIGEN_GIT = "https://github.com/eigenteam/eigen-git-mirror.git"
#
def get_install_required():
    with open("./requirements.txt", "r") as reqs:
        requirements = reqs.readlines()
    return [r.rstrip() for r in requirements]
#
# def check_eigen_installed():
#     sig_file = "signature_of_eigen3_matrix_library"
#     eigen_dst = os.path.join(cwd, "btrack", "include", "eigen")
#     eigen_signature = os.path.join(eigen_dst, sig_file)
#     if not os.path.exists(eigen_signature):
#         # try to clone it
#         clone_eigen = ["git", "clone", EIGEN_GIT, eigen_dst]
#         if subprocess.call(clone_eigen) != 0:
#             sys.exit(-1)
#
# # to be run before installation?
# def build_btrack():
#     check_eigen_installed()
#
#     if not os.path.exists('./btrack/libs'):
#         os.makedirs('./btrack/libs')
#     if not os.path.exists('./btrack/obj'):
#         os.makedirs('./btrack/obj')
#
#     # run the build scripts
#     make_BT = ["make"]
#     if subprocess.call(make_BT) != 0:
#         sys.exit(-1)
#
# # NOTE(arl): I'm not sure I like this!
# build_btrack()


setup(name='btrack',
      version='0.3.0',
      description='BayesianTracker is a simple Python/C++ based framework for multi-object tracking',
      author='Alan R. Lowe',
      author_email='a.lowe@ucl.ac.uk',
      url='https://github.com/quantumjot/BayesianTracker',
      packages=setuptools.find_packages(),
      package_data={'btrack': ['libs/libtracker*']},
      include_package_data=True,
      install_requires=get_install_required(),
      license='LICENSE.md')
