import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExtension(build_ext):
    def build_extension(self, extension):
        source = Path(__file__).resolve().parent
        build = Path(self.build_temp).resolve()
        output = Path(self.get_ext_fullpath(extension.name)).resolve().parent
        subprocess.run([
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(output),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=" + str(output),
            "-DPython_EXECUTABLE=" + sys.executable,
            "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
        ], check=True)
        subprocess.run([
            "cmake", "--build", str(build), "--config", "Release",
            "--parallel", str(self.parallel or 2),
        ], check=True)


setup(
    ext_modules=[Extension("quad_converter.quad_converter", sources=[])],
    cmdclass={"build_ext": BuildExtension},
)
