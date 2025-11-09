# h5typer

**Automatic type mapping for Python objects to HDF5**

h5typer is a lightweight Python package that provides seamless type conversion when saving and loading nested Python dictionaries to/from HDF5 files. It automatically handles the complexity of mapping Python types (numpy arrays, pandas DataFrames/Series, None, etc.) to HDF5-compatible formats.

Originally developed as part of the [HiC-SCA](https://github.com/iQLS-MMS/hic-sca) package for Hi-C analysis, h5typer has been extracted as a standalone, reusable package that can be used in any Python project requiring HDF5 I/O with automatic type handling.

## Features

- **Automatic Type Conversion**: Transparently handles numpy arrays, pandas DataFrames/Series, strings, None, and standard Python types
- **Nested Dictionary Support**: Save and load arbitrarily nested dictionary structures
- **Robust**: Handles edge cases like empty arrays, None values, and object dtypes
- **Simple API**: Just two main functions - `save_data()` and `load_data()`
- **Efficient**: Uses HDF5's hierarchical storage for fast I/O

## Why h5typer?

Standard HDF5 libraries like h5py require manual handling of type conversions:

```python
# Without h5typer - manual type handling
import h5py

with h5py.File('data.h5', 'w') as f:
    f.create_dataset('array', data=np.array([1, 2, 3]))
    f.create_dataset('string', data='hello'.encode('utf-8'))  # Must encode
    # Complex types like pandas DataFrames require custom serialization

# With h5typer - automatic!
import h5typer

data = {
    'array': np.array([1, 2, 3]),
    'string': 'hello',
    'dataframe': pd.DataFrame({'A': [1, 2]})
}
h5typer.save_data('data.h5', data)
```

h5typer handles all the boilerplate, letting you focus on your data.

## Installation

```bash
# Install from source
git clone https://github.com/iQLS-MMS/h5typer.git
cd h5typer
pip install .
```

### Requirements

- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.0.0
- h5py >= 3.0.0

## Quick Start

```python
import h5typer
import numpy as np
import pandas as pd

# Create some data
data = {
    'experiment': {
        'results': np.array([1, 2, 3, 4, 5]),
        'metadata': {
            'name': 'Test Experiment',
            'date': '2025-10-27',
            'valid': True
        },
        'dataframe': pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        }),
        'empty_value': None
    }
}

# Save to HDF5
h5typer.save_data('output.h5', data)

# Load from HDF5
loaded_data = h5typer.load_data('output.h5')

# Access your data - it's exactly as you saved it!
print(loaded_data['experiment']['results'])  # array([1, 2, 3, 4, 5])
print(loaded_data['experiment']['metadata']['name'])  # 'Test Experiment'
print(loaded_data['experiment']['dataframe'])  # Original DataFrame
```

## API Reference

### Functions

#### `save_data(filename, data_dict, update=False)`

Save a nested dictionary to an HDF5 file.

**Parameters:**
- `filename` (str): Path to the HDF5 file
- `data_dict` (dict): Nested dictionary to save
- `update` (bool, optional): If True, update existing file; if False, overwrite. Default: False

**Example:**
```python
import h5typer

data = {'key': 'value', 'array': np.array([1, 2, 3])}
h5typer.save_data('mydata.h5', data)

# Update existing file
more_data = {'new_key': 'new_value'}
h5typer.save_data('mydata.h5', more_data, update=True)
```

#### `load_data(filename)`

Load a nested dictionary from an HDF5 file.

**Parameters:**
- `filename` (str): Path to the HDF5 file

**Returns:**
- `dict`: The loaded nested dictionary

**Example:**
```python
import h5typer

data = h5typer.load_data('mydata.h5')
print(data['key'])  # 'value'
```

### Class API

For more control, use the `H5Typer` class directly:

```python
from h5typer import H5Typer

# Create instance
io_handler = H5Typer()

# Save data
io_handler.save_data('output.h5', my_dict)

# Load data
loaded = io_handler.load_data('output.h5')
```

## Supported Types

h5typer automatically handles the following Python types:

| Python Type | HDF5 Storage | Notes |
|-------------|--------------|-------|
| `numpy.ndarray` | Dataset | All dtypes supported, including object arrays |
| `pandas.DataFrame` | Group with datasets | Index, columns, and values preserved |
| `pandas.Series` | Group with datasets | Index and values preserved |
| `dict` | Group | Nested dictionaries become HDF5 groups |
| `str` | Dataset (bytes) | UTF-8 encoded |
| `int`, `float` | Dataset | Stored as numpy scalars |
| `None` | h5py.Empty | Preserved on load |
| `list`, `tuple` | Dataset | Converted to numpy arrays, all elements must be of the same type |

## Type Conversion Details

### NumPy Arrays

```python
# String arrays
str_array = np.array(['a', 'b', 'c'])
# Automatically converted to bytes for HDF5

# Float64 arrays
float_array = np.array([1.0, 5.0, 3.0], dtype=np.float64)
```

### Pandas Objects

```python
# DataFrames - index, columns, and values all preserved
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# Series - index and values preserved
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
```

### None Values

```python
data = {
    'value': None,
}
# None values are preserved through save/load
```

## Advanced Usage

### Nested Structures

```python
data = {
    'level1': {
        'level2': {
            'level3': {
                'deep_array': np.array([1, 2, 3])
            }
        }
    }
}

h5typer.save_data('nested.h5', data)
loaded = h5typer.load_data('nested.h5')
# Structure is preserved
```

### Updating Files

```python
# Initial save
h5typer.save_data('data.h5', {'key1': 'value1'})

# Add more data
h5typer.save_data('data.h5', {'key2': 'value2'}, update=True)

# Load gets both keys
data = h5typer.load_data('data.h5')
# {'key1': 'value1', 'key2': 'value2'}
```

### Integer Keys

```python
# Integer keys are preserved
data = {
    100: 'hundred',
    200: 'two hundred',
    'string_key': 'value'
}

h5typer.save_data('intkeys.h5', data)
loaded = h5typer.load_data('intkeys.h5')
print(loaded[100])  # 'hundred' - key is still an integer
```

## Integration with HiC-SCA

h5typer is used by [HiC-SCA](https://github.com/iQLS-MMS/hic-sca) for all HDF5 I/O operations:

```python
# In HiC-SCA
from hicsca import HiCSCA

# Process Hi-C data
hicsca = HiCSCA("sample.hic", resolutions=[100000])
hicsca.process_all_chromosomes()

# Save results (uses h5typer internally)
hicsca.to_hdf5("results.h5")

# Load results (uses h5typer internally)
hicsca_loaded = HiCSCA.from_hdf5("results.h5")
```

The integration is transparent - HiC-SCA uses h5typer for automatic type mapping of complex nested dictionaries containing numpy arrays, pandas DataFrames, and metadata.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 0.1.0
- Initial release
- Support for numpy arrays, pandas DataFrames/Series, None values
- Nested dictionary support
- Integer key preservation
- Update functionality

## See Also

- **HiC-SCA**: Hi-C Spectral Compartment Analysis package using h5typer
- **h5py**: Low-level HDF5 interface for Python
- **pandas**: Data analysis library with HDF5 support
