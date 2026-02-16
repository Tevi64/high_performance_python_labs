from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "gauss_seidel_cpp",
        ["gauss_seidel_cpp.cpp"],
        language='c++',
        cxx_std=17
    ),
]

setup(
    name="gauss_seidel_cpp",
    version=__version__,
    description="Модуль для решения уравнений с помощью pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)