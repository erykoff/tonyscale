try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tonyscale")
except PackageNotFoundError:
    # package is not installed
    pass

from . import tonyscale

from .tonyscale import *
