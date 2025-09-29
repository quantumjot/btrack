# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys

import pytz

sys.path.insert(0, os.path.abspath(".."))

import btrack

# -- Project information -----------------------------------------------------

project = "Bayesian Tracker (btrack) 🔬💻"
author = "Alan R Lowe"
year = datetime.datetime.now(tz=pytz.timezone("GMT")).year
copyright = f"2014-{year}, {author}"  # noqa: A001


# The full version, including alpha/beta/rc tags
version = f"v{btrack.__version__}"
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_panels",
    "sphinx_automodapi.automodapi",
    "numpydoc",
    "sphinx_rtd_theme",
]

numpydoc_show_class_members = False
automodapi_inheritance_diagram = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/btrack_logo.png"
html_theme_options = {
    "logo_only": True,
}
