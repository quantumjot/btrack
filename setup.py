from distutils.command.build_ext import build_ext as _build_ext

from setuptools import Extension, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class CTypesExtension(Extension):
    pass


class BuildExt(_build_ext):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


class BdistWheelAbiNone(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        return "py3", "none", plat


setup(
    ext_modules=[
        CTypesExtension(
            "btrack.src.tracker",
            ["btrack/src/tracker.cc"],
            include_dirs=["btrack/include"],
            language="c++",
            extra_compile_args=["-std=c++11", "-v"],
        ),
    ],
    cmdclass={"build_ext": BuildExt, "bdist_wheel": BdistWheelAbiNone},
)
