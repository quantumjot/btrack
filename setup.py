"""BayesianTracker (`btrack`) is a multi object tracking algorithm,
specifically used to reconstruct trajectories in crowded fields.  New
observations are assigned to tracks by evaluating the posterior probability of
each potential linkage from a Bayesian belief matrix for all possible
linkages. """

from setuptools import find_packages, setup


def get_install_required():
    with open("./requirements.txt", "r") as reqs:
        requirements = reqs.readlines()
    return [r.rstrip() for r in requirements]


def get_version():
    with open("./btrack/VERSION.txt", "r") as ver:
        version = ver.readline()
    return version.rstrip()


DESCRIPTION = 'A framework for Bayesian multi-object tracking'
LONG_DESCRIPTION = __doc__


setup(
    name='btrack',
    version=get_version(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Alan R. Lowe',
    author_email='a.lowe@ucl.ac.uk',
    url='https://github.com/quantumjot/BayesianTracker',
    packages=find_packages(),
    package_data={'btrack': ['libs/libtracker*', 'VERSION.txt']},
    install_requires=get_install_required(),
    python_requires='>=3.6',
    license='LICENSE.md',
    classifiers=[
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)
