"""BayesianTracker (`btrack`) is a multi object tracking algorithm,
specifically used to reconstruct trajectories in crowded fields.  New
observations are assigned to tracks by evaluating the posterior probability of
each potential linkage from a Bayesian belief matrix for all possible
linkages. """

from setuptools import find_packages, setup


def get_install_required(extra=None):
    extra = f"-{extra}" if extra is not None else ""
    with open(f"./requirements{extra}.txt", "r") as reqs:
        requirements = reqs.readlines()
    return [r.rstrip() for r in requirements]


def get_version():
    with open("./btrack/VERSION.txt", "r") as ver:
        version = ver.readline()
    return version.rstrip()


DESCRIPTION = "A framework for Bayesian multi-object tracking"
LONG_DESCRIPTION = __doc__

extras = ["docs", "napari"]
extras_require = {extra: get_install_required(extra) for extra in extras}

setup(
    version=get_version(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={"btrack": ["libs/libtracker*", "VERSION.txt", "napari.yaml"]},
    install_requires=get_install_required(),
    extras_require=extras_require,
    python_requires=">=3.6",
    license="LICENSE.md",
)
