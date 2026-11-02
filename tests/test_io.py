"""
Tests for h5typer I/O functionality.

Tests cover:
- Basic type conversions (int, float, str, None)
- NumPy array handling
- Pandas DataFrame and Series
- Nested dictionaries
- Empty dicts
- Update mode
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from h5typer import H5Typer, save_data, load_data


class TestBasicTypes:
    """Test basic Python type conversions."""

    def test_save_load_scalars(self):
        """Test saving and loading scalar values."""
        data = {
            'int_val': 42,
            'float_val': 3.14,
            'str_val': 'hello world',
            'none_val': None
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            assert loaded['int_val'] == 42
            assert loaded['float_val'] == 3.14
            assert loaded['str_val'] == 'hello world'
            assert loaded['none_val'] is None
        finally:
            os.unlink(temp_file)

    def test_empty_dict(self):
        """Test saving and loading empty dictionaries."""
        data = {'empty': {}}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            assert 'empty' in loaded
            assert loaded['empty'] == {}
        finally:
            os.unlink(temp_file)


class TestNumPyArrays:
    """Test NumPy array handling."""

    def test_numeric_arrays(self):
        """Test various numeric array types."""
        data = {
            'int_array': np.array([1, 2, 3, 4, 5]),
            'float_array': np.array([1.1, 2.2, 3.3]),
            'bool_array': np.array([True, False, True]),
            'complex_array': np.array([1+2j, 3+4j])
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            np.testing.assert_array_equal(loaded['int_array'], data['int_array'])
            np.testing.assert_array_almost_equal(loaded['float_array'], data['float_array'])
            np.testing.assert_array_equal(loaded['bool_array'], data['bool_array'])
            np.testing.assert_array_equal(loaded['complex_array'], data['complex_array'])
        finally:
            os.unlink(temp_file)

    def test_string_arrays(self):
        """Test string array conversion."""
        data = {
            'str_array': np.array(['hello', 'world', 'test'])
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            np.testing.assert_array_equal(loaded['str_array'], data['str_array'])
        finally:
            os.unlink(temp_file)

    def test_multidimensional_arrays(self):
        """Test multidimensional arrays."""
        data = {
            '2d_array': np.array([[1, 2, 3], [4, 5, 6]]),
            '3d_array': np.random.rand(3, 4, 5)
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            np.testing.assert_array_equal(loaded['2d_array'], data['2d_array'])
            np.testing.assert_array_almost_equal(loaded['3d_array'], data['3d_array'])
        finally:
            os.unlink(temp_file)


class TestPandas:
    """Test pandas DataFrame and Series handling."""

    def test_dataframe(self):
        """Test DataFrame save/load."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5.0, 6.0, 7.0, 8.0],
            'C': ['a', 'b', 'c', 'd']
        })

        data = {'dataframe': df}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            pd.testing.assert_frame_equal(loaded['dataframe'], df)
        finally:
            os.unlink(temp_file)

    def test_series(self):
        """Test Series save/load."""
        series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'], name='test_series')

        data = {'series': series}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            pd.testing.assert_series_equal(loaded['series'], series)
        finally:
            os.unlink(temp_file)

    def test_dataframe_with_multiindex(self):
        """Test DataFrame with MultiIndex."""
        index = pd.MultiIndex.from_tuples(
            [('A', 1), ('A', 2), ('B', 1), ('B', 2)],
            names=['letter', 'number']
        )
        df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)

        data = {'multiindex_df': df}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            pd.testing.assert_frame_equal(loaded['multiindex_df'], df)
        finally:
            os.unlink(temp_file)

    def test_multiindex_integer_levels_keep_dtype(self):
        """Test integer MultiIndex levels are not converted to strings."""
        index = pd.MultiIndex.from_arrays(
            [['chr1', 'chr1', 'chr2'], [100000, 200000, 100000]],
            names=['chr', 'pos']
        )
        columns = pd.MultiIndex.from_arrays(
            [['x', 'y'], [1, 2]], names=['group', 'rep']
        )
        df = pd.DataFrame(np.arange(6.0).reshape(3, 2), index=index, columns=columns)
        series = pd.Series([1.0, 2.0, 3.0], index=index)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, {'df': df, 'series': series})
            loaded = load_data(temp_file)

            pd.testing.assert_frame_equal(loaded['df'], df)
            pd.testing.assert_series_equal(loaded['series'], series)
            assert loaded['df'].index.get_level_values('pos').dtype == np.int64
        finally:
            os.unlink(temp_file)


class TestNestedStructures:
    """Test nested dictionary structures."""

    def test_nested_dicts(self):
        """Test deeply nested dictionaries."""
        data = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': np.array([1, 2, 3]),
                        'name': 'deep_value'
                    },
                    'value2': 'hello'
                },
                'value1': 42
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            assert loaded['level1']['value1'] == 42
            assert loaded['level1']['level2']['value2'] == 'hello'
            assert loaded['level1']['level2']['level3']['name'] == 'deep_value'
            np.testing.assert_array_equal(
                loaded['level1']['level2']['level3']['value'],
                data['level1']['level2']['level3']['value']
            )
        finally:
            os.unlink(temp_file)

    def test_mixed_types_nested(self):
        """Test nested structure with mixed types."""
        data = {
            'experiment': {
                'metadata': {
                    'name': 'Test',
                    'date': '2025-10-27',
                    'valid': True
                },
                'results': {
                    'array': np.random.rand(10),
                    'dataframe': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
                    'value': None
                }
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            assert loaded['experiment']['metadata']['name'] == 'Test'
            assert loaded['experiment']['results']['value'] is None
            pd.testing.assert_frame_equal(
                loaded['experiment']['results']['dataframe'],
                data['experiment']['results']['dataframe']
            )
        finally:
            os.unlink(temp_file)


class TestIntegerKeys:
    """Test integer keys in dictionaries."""

    def test_integer_keys(self):
        """Test that integer keys are preserved."""
        data = {
            100: 'hundred',
            200: 'two hundred',
            'string_key': 'value'
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            assert loaded[100] == 'hundred'
            assert loaded[200] == 'two hundred'
            assert loaded['string_key'] == 'value'
            assert isinstance(list(loaded.keys())[0], int)
        finally:
            os.unlink(temp_file)


class TestUpdateMode:
    """Test update mode functionality."""

    def test_update_existing_file(self):
        """Test updating an existing HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            # Initial save
            data1 = {'key1': 'value1', 'array1': np.array([1, 2, 3])}
            save_data(temp_file, data1)

            # Update with new data
            data2 = {'key2': 'value2', 'array2': np.array([4, 5, 6])}
            save_data(temp_file, data2, update=True)

            # Load and verify both datasets exist
            loaded = load_data(temp_file)

            assert 'key1' in loaded
            assert 'key2' in loaded
            assert loaded['key1'] == 'value1'
            assert loaded['key2'] == 'value2'
            np.testing.assert_array_equal(loaded['array1'], np.array([1, 2, 3]))
            np.testing.assert_array_equal(loaded['array2'], np.array([4, 5, 6]))
        finally:
            os.unlink(temp_file)

    def test_update_overwrites_existing_keys(self):
        """Test update mode replaces existing keys, including type changes."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, {'a': 1, 'b': {'c': 2}, 'keep': 'k'})
            save_data(temp_file, {'a': 5, 'b': 3}, update=True)
            save_data(temp_file, {'a': {'z': 9}}, update=True)

            loaded = load_data(temp_file)

            assert loaded == {'a': {'z': 9}, 'b': 3, 'keep': 'k'}
        finally:
            os.unlink(temp_file)

    def test_overwrite_mode(self):
        """Test that default mode overwrites existing file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            # Initial save
            data1 = {'key1': 'value1'}
            save_data(temp_file, data1)

            # Overwrite with new data (update=False is default)
            data2 = {'key2': 'value2'}
            save_data(temp_file, data2)

            # Load and verify only new data exists
            loaded = load_data(temp_file)

            assert 'key1' not in loaded
            assert 'key2' in loaded
            assert loaded['key2'] == 'value2'
        finally:
            os.unlink(temp_file)


class TestClassAPI:
    """Test the H5Typer class API."""

    def test_class_save_load(self):
        """Test using H5Typer class directly."""
        io_handler = H5Typer()

        data = {
            'value': 42,
            'array': np.array([1, 2, 3])
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            io_handler.save_data(temp_file, data)
            loaded = io_handler.load_data(temp_file)

            assert loaded['value'] == 42
            np.testing.assert_array_equal(loaded['array'], data['array'])
        finally:
            os.unlink(temp_file)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_array(self):
        """Test empty numpy arrays."""
        data = {'empty_array': np.array([])}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            np.testing.assert_array_equal(loaded['empty_array'], np.array([]))
        finally:
            os.unlink(temp_file)

    def test_unicode_strings(self):
        """Test Unicode string handling."""
        data = {
            'unicode': 'Hello 世界 🌍',
            'unicode_array': np.array(['こんにちは', 'مرحبا', '你好'])
        }

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            save_data(temp_file, data)
            loaded = load_data(temp_file)

            assert loaded['unicode'] == 'Hello 世界 🌍'
            np.testing.assert_array_equal(loaded['unicode_array'], data['unicode_array'])
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
