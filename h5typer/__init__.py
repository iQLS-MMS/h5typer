"""
h5typer: Automatic Type Mapping for Python Objects to HDF5

A lightweight, robust package for saving and loading nested Python dictionaries
to/from HDF5 files. Automatically handles type conversion for numpy arrays,
pandas DataFrames/Series, and standard Python types.

Features:
- Save and load arbitrarily nested dictionaries
- Automatic type conversion (numpy, pandas, strings, None, etc.)
- Support for complex nested structures
- Robust handling of various data types

Basic Usage
-----------
>>> import h5typer
>>>
>>> # Save data
>>> data = {'experiment': {'results': np.array([1, 2, 3])}}
>>> h5typer.save_data('output.h5', data)
>>>
>>> # Load data
>>> loaded_data = h5typer.load_data('output.h5')

Object-Oriented Usage
--------------------
>>> from h5typer import H5Typer
>>>
>>> io_handler = H5Typer()
>>> io_handler.save_data('output.h5', data)
>>> loaded = io_handler.load_data('output.h5')
"""

from importlib.metadata import version, PackageNotFoundError

try:
    # Set from the git tag by setuptools-scm at install time
    __version__ = version("h5typer")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "HiC-SCA Project"
__license__ = "MIT"

from .io import (
    H5Typer,
    save_data,
    load_data
)

__all__ = [
    'H5Typer',
    'save_data',
    'load_data',
    '__version__'
]
