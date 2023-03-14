import os

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "libtracker",
            extra_compile_args=["-std=c++11", "-v"],
            include_dirs=["btrack/include"],
            language="c++",
            libraries=["c++"],
            library_dirs=[os.path.abspath("./btrack/lib")],
            sources=["btrack/src/tracker.cc"],
        )
    ],
)
