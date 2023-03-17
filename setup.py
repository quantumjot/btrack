from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "libtracker",
            ["btrack/src/tracker.cc"],
            include_dirs=["btrack/include"],
            extra_compile_args=["-c", "-std=c++17", "-m64", "-fPIC"],
        ),
    ],
)
