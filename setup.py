#!/usr/bin/env python

# import os
# from setuptools.command.install import install
# from distutils.core import setup
#
# cwd = os.getcwd()
#
# def get_install_required():
#     with open("./requirements.txt", "r") as reqs:
#         requirements = reqs.readlines()
#     return [r.rstrip() for r in requirements]
#
# def check_eigen_installed():
#     sig_file = "signature_of_eigen3_matrix_library"
#     eigen_signature = os.path.join(cwd, "btrack", "include", "eigen", sig_file)
#     if not os.path.exists(eigen_signature):
#         raise ModuleNotFoundError("Eigen does not seem to be installed.")
#
# class TrackerInstaller(install):
#     def run(self):
#         install.run(self)
#         # check for eigen
#         check_eigen_installed()
#         # run the build scripts
#         build_sh = os.path.join(cwd, "btrack", "bin", "build_tracker.sh")
#         os.system("chmod +x {}".format(build_sh))
#         os.system("sh {}".format(build_sh))

# TODO(arl): finish this!

# setup(name='BayesianTracker',
#       version='0.2.10',
#       description='BayesianTracker is a simple Python/C++ based framework for \
#       multi-object tracking',
#       author='Alan R. Lowe',
#       author_email='a.lowe@ucl.ac.uk',
#       url='https://github.com/quantumjot/BayesianTracker',
#       packages=['btrack'],
#       install_requires=get_install_required(),
#       cmdclass={'install': TrackerInstaller})
