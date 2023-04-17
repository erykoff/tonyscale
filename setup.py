from setuptools import setup, Extension
import numpy


ext = Extension(
    "tonyscale._tonyscale",
    [
        "tonyscale/tonyscale.c",
    ],
)

setup(
    ext_modules=[ext],
    include_dirs=numpy.get_include(),
)
