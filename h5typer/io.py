"""
h5typer I/O Module

Handles saving and loading nested Python dictionaries to/from HDF5 files.
Provides automatic type mapping between Python objects and HDF5 data types.
Compatible with numpy arrays, pandas DataFrames/Series, and standard Python types.
"""

import numpy as np
import pandas as pd
import h5py
import types
from typing import Dict, Optional, Any
from pathlib import Path


class H5Typer:
    """
    Handles saving and loading of nested Python dictionaries to HDF5.

    Provides automatic type conversion and mapping between Python objects
    and HDF5 data types.
    """

    def __init__(self):
        """Initialize the data I/O handler."""
        # File format version: 1.1 stores MultiIndex levels and mixed-dtype
        # DataFrame columns as separate arrays, and saves Series names
        self.version = 1.1

    @staticmethod
    def _convert_npstr_to_byte(str_array: np.ndarray) -> np.ndarray:
        """Convert numpy string array to bytes for HDF5 compatibility."""
        return np.char.encode(str_array, encoding='utf-8')

    @staticmethod
    def _convert_npbyte_to_str(byte_array: np.ndarray) -> np.ndarray:
        """Convert numpy bytes array to string."""
        return np.char.decode(byte_array, encoding='utf-8')

    @staticmethod
    def _convert_obj_to_dtypes(obj_array: np.ndarray) -> np.ndarray:
        """Convert object dtype arrays to appropriate types."""
        current_array = np.array(obj_array.tolist())
        current_type = type(current_array.dtype)

        type_handlers = {
            str: lambda x: x.encode('utf-8'),
            np.dtypes.StrDType: H5Typer._convert_npstr_to_byte,
            types.NoneType: lambda x: h5py.Empty('f'),
        }

        if current_type == np.dtypes.ObjectDType:
            return obj_array
        elif current_type in type_handlers:
            return type_handlers[current_type](current_array)
        else:
            return current_array

    @staticmethod
    def _split_pandas_index(index: pd.Index) -> tuple:
        """
        Split a pandas Index into HDF5-storable arrays and names.

        A MultiIndex is stored as one array per level (keys 'level_0', 'level_1', ...)
        so each level keeps its own dtype, instead of an array of tuples which numpy
        coerces to a single common dtype (e.g. integers become strings).
        """
        if isinstance(index, pd.MultiIndex):
            levels = {f'level_{i}': index.get_level_values(i).to_numpy()
                      for i in range(index.nlevels)}
            names = {f'level_{i}': level_name for i, level_name in enumerate(index.names)}
            return levels, names

        return index.to_numpy(), index.name

    def _convert_to_h5_compatible(self, obj: Any) -> tuple:
        """
        Convert Python objects to HDF5-compatible format.

        Returns
        -------
        tuple
            (data_type, converted_obj)
        """
        data_type = ''
        converted_obj = obj

        if isinstance(obj, np.ndarray):
            data_type = 'numpy'
            dtype_type = type(obj.dtype)

            if dtype_type == np.dtypes.StrDType:
                converted_obj = self._convert_npstr_to_byte(obj)
            elif dtype_type == np.dtypes.ObjectDType:
                converted_obj = self._convert_obj_to_dtypes(obj)

        elif isinstance(obj, dict):
            data_type = 'dict'

        elif isinstance(obj, pd.core.generic.NDFrame):
            index, name = self._split_pandas_index(obj.index)

            if isinstance(obj, pd.DataFrame):
                data_type = 'pandas dataframe'
                columns, columns_name = self._split_pandas_index(obj.columns)

                # Mixed column dtypes are stored one column per array (keys 'col_0',
                # 'col_1', ...) so numpy does not coerce them to a common dtype
                if len(set(obj.dtypes)) > 1:
                    values = {f'col_{i}': obj.iloc[:, i].to_numpy() for i in range(obj.shape[1])}
                else:
                    values = obj.to_numpy()

                converted_obj = {'values': values, 'index': index, 'name': name,
                                 'columns': columns, 'columns_name': columns_name}

            elif isinstance(obj, pd.Series):
                data_type = 'pandas series'
                converted_obj = {'values': obj.to_numpy(), 'index': index, 'name': name,
                                 'series_name': obj.name}

        elif isinstance(obj, types.NoneType):
            data_type = 'None'
            converted_obj = h5py.Empty('f')

        elif isinstance(obj, str):
            data_type = 'str'
            converted_obj = obj.encode('utf-8')

        elif isinstance(obj, (int, float, np.integer, np.floating)):
            data_type = 'scalar'
            converted_obj = np.array(obj)

        elif isinstance(obj, list) or isinstance(obj, tuple):
            obj = np.array(obj)
            data_type = 'object'
            dtype_type = type(obj.dtype)

            if dtype_type == np.dtypes.StrDType:
                converted_obj = self._convert_npstr_to_byte(obj)
            elif dtype_type == np.dtypes.ObjectDType:
                converted_obj = self._convert_obj_to_dtypes(obj)

        else:
            data_type = 'object'
            converted_obj = np.array(obj)

        return data_type, converted_obj

    def _flatten_dict(self, input_dict: Dict, parent_address: str = "") -> tuple:
        """
        Flatten nested dictionary for HDF5 storage.

        Parameters
        ----------
        input_dict : dict
            Nested dictionary to flatten
        parent_address : str
            Current hierarchical path

        Returns
        -------
        tuple
            (meta_dict, output_dict) containing metadata and flattened data
        """
        output_dict = {}
        meta_dict = {}

        if len(input_dict) == 0:
            meta_dict[parent_address] = 'dict'
        else:
            for key, value in input_dict.items():
                data_type, current_value = self._convert_to_h5_compatible(value)
                current_full_address = parent_address + '/' + str(key)

                if isinstance(current_value, dict):
                    temp_meta_dict, temp_output_dict = self._flatten_dict(
                        current_value, current_full_address
                    )
                    output_dict.update(temp_output_dict)

                    # For pandas types, store metadata at current level
                    if 'pandas' in data_type:
                        meta_dict[current_full_address] = data_type
                    else:
                        meta_dict.update(temp_meta_dict)
                else:
                    output_dict[current_full_address] = current_value
                    meta_dict[current_full_address] = data_type

        if parent_address == "":
            meta_dict = {'/__meta__' + key: val for key, val in meta_dict.items()}

        return meta_dict, output_dict

    @staticmethod
    def _remove_h5_path(h5_file: h5py.File, address: str) -> None:
        """
        Remove the entry at address, or a dataset on its parent path that would
        block creating it (e.g. an old scalar 'x' when writing 'x/y').
        """
        parts = address.strip('/').split('/')

        for i in range(1, len(parts) + 1):
            path = '/'.join(parts[:i])

            if path not in h5_file:
                return

            if i == len(parts) or isinstance(h5_file[path], h5py.Dataset):
                del h5_file[path]
                return

    def save_data(self, filename: str, data_dict: Dict, update: bool = False) -> None:
        """
        Save nested dictionary to HDF5 file.

        Parameters
        ----------
        filename : str
            Path to HDF5 file
        data_dict : dict
            Nested dictionary containing data to save
        update : bool, optional
            If True, update existing file; if False, overwrite (default: False)
        """
        write_mode = 'a' if update else 'w'

        meta_dict, flat_dict = self._flatten_dict(data_dict)

        with h5py.File(filename, write_mode) as h5_file:
            if update:
                # h5py cannot assign over existing entries, so clear what will be rewritten
                for meta_address in meta_dict:
                    data_address = meta_address[len('/__meta__'):]
                    if data_address.strip('/'):
                        self._remove_h5_path(h5_file, data_address)
                        self._remove_h5_path(h5_file, meta_address)

                if '__version__' in h5_file:
                    del h5_file['__version__']

            h5_file.update(flat_dict)
            h5_file.update(meta_dict)
            h5_file['__version__'] = self.version

    def _auto_input_type_converter(self, obj: Any) -> Any:
        """Convert HDF5 objects back to Python types."""
        obj_type = type(obj)

        if obj_type == np.ndarray:
            dtype_type = type(obj.dtype)

            if dtype_type == np.dtypes.BytesDType:
                return self._convert_npbyte_to_str(obj)
            elif dtype_type == np.dtypes.StrDType:
                return obj
            elif (dtype_type == np.dtypes.ObjectDType and obj.size > 0
                  and all(isinstance(x, bytes) for x in obj.flat)):
                # Variable-length strings come back from h5py as bytes objects
                return np.array([x.decode('utf-8') for x in obj.flat]).reshape(obj.shape)

        elif obj_type == bytes:
            return obj.decode('utf-8')
        elif isinstance(obj, h5py._hl.base.Empty):
            return None

        return obj

    def _object_input_handler(self, data_type: str, var_name: str, data_dict: Dict) -> Any:
        """Handle object type reconstruction."""
        data_obj = self._auto_input_type_converter(data_dict[var_name])

        if (data_type == 'object') and (type(data_obj) == np.ndarray):
            return data_obj.tolist()
        
        elif (data_type in ('object', 'scalar')) and isinstance(data_obj, np.generic):
            return data_obj.item()

        return data_obj

    def _str_input_handler(self, data_type: str, var_name: str, data_dict: Dict) -> str:
        """Handle string type reconstruction."""
        return data_dict[var_name].decode('utf-8')

    def _dict_input_handler(self, data_type: str, var_name: str, data_dict: Dict) -> dict:
        """Handle empty dict reconstruction."""
        return {}

    def _pandas_index_input_handler(self, index_address: str, name_address: str, data_dict: Dict) -> pd.Index:
        """Rebuild a pandas Index or MultiIndex saved by _split_pandas_index."""
        if index_address in data_dict:
            index = self._auto_input_type_converter(data_dict[index_address])
            name = self._auto_input_type_converter(data_dict.get(name_address))

            # Older files store a MultiIndex as a single 2D array of tuples
            if len(index.shape) > 1:
                return pd.MultiIndex.from_arrays(index.T, names=name)

            return pd.Index(index, name=name)

        n_levels = 0
        while f'{index_address}/level_{n_levels}' in data_dict:
            n_levels += 1

        levels = [self._auto_input_type_converter(data_dict[f'{index_address}/level_{i}'])
                  for i in range(n_levels)]
        names = [self._auto_input_type_converter(data_dict.get(f'{name_address}/level_{i}'))
                 for i in range(n_levels)]
        names = [n.item() if isinstance(n, np.generic) else n for n in names]

        return pd.MultiIndex.from_arrays(levels, names=names)

    def _pandas_input_handler(self, data_type: str, var_name: str, data_dict: Dict):
        """Handle pandas DataFrame and Series reconstruction."""
        index = self._pandas_index_input_handler(var_name + '/index', var_name + '/name', data_dict)

        if data_type == 'pandas series':
            values = self._auto_input_type_converter(data_dict[var_name + '/values'])
            series_name = self._auto_input_type_converter(data_dict.get(var_name + '/series_name'))

            if isinstance(series_name, np.generic):
                series_name = series_name.item()

            return pd.Series(values, index=index, name=series_name)

        elif data_type == 'pandas dataframe':
            columns = self._pandas_index_input_handler(
                var_name + '/columns', var_name + '/columns_name', data_dict
            )

            if var_name + '/values' in data_dict:
                values = self._auto_input_type_converter(data_dict[var_name + '/values'])
                return pd.DataFrame(values, index=index, columns=columns)

            # Mixed-dtype frames are stored one array per column
            col_arrays = [self._auto_input_type_converter(data_dict[f'{var_name}/values/col_{i}'])
                          for i in range(len(columns))]
            df = pd.DataFrame(dict(enumerate(col_arrays)), index=index)
            df.columns = columns

            return df

        else:
            raise ValueError(f"Invalid pandas data type: {data_type}")

    def _build_dict_recursive(self, name_list: list, value: Any, existing_dict: Optional[Dict] = None) -> Dict:
        """Recursively rebuild nested dictionary structure."""
        if existing_dict is None:
            existing_dict = {}

        current_key = name_list[0]

        if len(name_list) > 1:
            if current_key in existing_dict:
                existing_dict[current_key].update(
                    self._build_dict_recursive(name_list[1:], value, existing_dict[current_key])
                )
            else:
                existing_dict[current_key] = self._build_dict_recursive(name_list[1:], value)
        else:
            existing_dict[current_key] = value

        return existing_dict

    def _reload_from_flat_dict(self, meta_dict: Dict, data_dict: Dict) -> Dict:
        """Reconstruct nested dictionary from flattened HDF5 data."""
        input_handlers = {
            'numpy': self._object_input_handler,
            'object': self._object_input_handler,
            'dict': self._dict_input_handler,
            'str': self._str_input_handler,
            'None': self._object_input_handler,
            'scalar': self._object_input_handler,
            'pandas series': self._pandas_input_handler,
            'pandas dataframe': self._pandas_input_handler,
        }

        processed_dict = {}

        for var_full_name, data_type in meta_dict.items():
            handler = input_handlers.get(data_type, self._object_input_handler)
            var_name_list = var_full_name.split('/')

            # Convert string keys that are integers back to int
            for idx, var_name in enumerate(var_name_list):
                if var_name.isdigit():
                    var_name_list[idx] = int(var_name)

            var_name = var_name_list[0]
            value = handler(data_type, var_full_name, data_dict)

            if len(var_name_list) > 1:
                if var_name in processed_dict:
                    current_value = self._build_dict_recursive(
                        var_name_list[1:], value, processed_dict[var_name]
                    )
                else:
                    current_value = self._build_dict_recursive(var_name_list[1:], value)
            else:
                current_value = value

            processed_dict[var_name] = current_value

        return processed_dict

    def load_data(self, filename: str) -> Dict:
        """
        Load nested dictionary from HDF5 file.

        Parameters
        ----------
        filename : str
            Path to HDF5 file

        Returns
        -------
        dict
            Nested dictionary containing loaded data
        """
        with h5py.File(filename, 'r') as h5_file:
            # Check version
            if '__version__' in h5_file:
                version = h5_file['__version__'][()]
                # Older formats are still readable; only newer ones may not be
                if version > self.version:
                    print(f"Warning: File version {version} is newer than supported version {self.version}")

            # Collect metadata and data
            meta_dict = {}
            data_dict = {}

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    address_list = name.split('/')

                    if address_list[0] == '__meta__':
                        meta_dict['/'.join(address_list[1:])] = obj[()].decode('utf-8')
                    elif name != '__version__':
                        data_dict[name] = obj[()]

            h5_file.visititems(visitor)

        return self._reload_from_flat_dict(meta_dict, data_dict)


# Convenience functions for direct module usage
def save_data(filename: str, data_dict: Dict, update: bool = False) -> None:
    """
    Save nested dictionary to HDF5 file (convenience function).

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    data_dict : dict
        Nested dictionary to save
    update : bool, optional
        Update existing file if True

    Examples
    --------
    >>> data = {'results': {'exp1': np.array([1, 2, 3])}}
    >>> save_data('output.h5', data)
    """
    io_handler = H5Typer()
    io_handler.save_data(filename, data_dict, update=update)


def load_data(filename: str) -> Dict:
    """
    Load nested dictionary from HDF5 file (convenience function).

    Parameters
    ----------
    filename : str
        Path to HDF5 file

    Returns
    -------
    dict
        Loaded nested dictionary

    Examples
    --------
    >>> data = load_data('output.h5')
    """
    io_handler = H5Typer()
    return io_handler.load_data(filename)
