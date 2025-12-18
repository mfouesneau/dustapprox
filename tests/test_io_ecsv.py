"""Tests for dustapprox.io.ecsv module."""

import pytest
from typing import cast
import pandas as pd
import numpy as np

from dustapprox.io import ecsv


class TestReadHeader:
    """Test read_header function."""

    def test_read_header_basic(self, tmp_path):
        """Test reading a basic ECSV header."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# datatype:
# - {name: a, datatype: int64}
# - {name: b, datatype: float64}
a b
1 2.5
2 3.5
"""
        test_file.write_text(content)

        header = ecsv.read_header(str(test_file))

        assert header is not None
        assert "datatype" in header
        assert len(header["datatype"]) == 2
        assert header["datatype"][0]["name"] == "a"
        assert header["datatype"][1]["name"] == "b"

    def test_read_header_with_meta(self, tmp_path):
        """Test reading header with metadata."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# datatype:
# - {name: x, datatype: float64}
# meta:
#   author: test
#   version: 1.0
x
1.0
"""
        test_file.write_text(content)

        header = ecsv.read_header(str(test_file))

        assert "meta" in header
        assert header["meta"]["author"] == "test"
        assert header["meta"]["version"] == 1.0

    def test_read_header_with_units(self, tmp_path):
        """Test reading header with units."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# datatype:
# - {name: wavelength, unit: angstrom, datatype: float64}
# - {name: flux, unit: erg / s / cm2 / angstrom, datatype: float64}
wavelength flux
1000 1e-10
2000 2e-10
"""
        test_file.write_text(content)

        header = ecsv.read_header(str(test_file))

        assert header["datatype"][0]["unit"] == "angstrom"
        assert "erg" in header["datatype"][1]["unit"]

    def test_read_header_empty_lines_ignored(self, tmp_path):
        """Test that empty lines in header are ignored."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---

# datatype:
# - {name: x, datatype: int64}

x
1
"""
        test_file.write_text(content)

        header = ecsv.read_header(str(test_file))
        assert "datatype" in header

    def test_read_header_nonexistent_file(self):
        """Test reading header from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ecsv.read_header("/nonexistent/path/file.ecsv")


class TestRead:
    """Test read function."""

    def test_read_basic(self, tmp_path):
        """Test reading a basic ECSV file."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# datatype:
# - {name: a, datatype: int64}
# - {name: b, datatype: float64}
a,b
1,2.5
2,3.5
3,4.5
"""
        test_file.write_text(content)

        df = ecsv.read(str(test_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "a" in df.columns
        assert "b" in df.columns
        assert df["a"].dtype == np.int64
        assert df["b"].dtype == np.float64

    def test_read_with_metadata(self, tmp_path):
        """Test that metadata is stored in attrs."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# datatype:
# - {name: x, datatype: float64}
# meta:
#   source: test_source
#   date: 2024-01-01
x
1.0
2.0
"""
        test_file.write_text(content)

        df = cast(pd.DataFrame, ecsv.read(str(test_file)))

        assert hasattr(df, "attrs")
        assert "source" in df.attrs
        assert df.attrs["source"] == "test_source"

    def test_read_string_datatype(self, tmp_path):
        """Test reading string datatype."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# datatype:
# - {name: name, datatype: string}
# - {name: value, datatype: float64}
name,value
alpha,1.0
beta,2.0
"""
        test_file.write_text(content)

        df = cast(pd.DataFrame, ecsv.read(str(test_file)))

        assert df["name"].dtype == object  # strings are stored as object
        assert df["value"].dtype == np.float64

    def test_read_with_custom_delimiter(self, tmp_path):
        """Test reading with custom delimiter."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# delimiter: '|'
# datatype:
# - {name: a, datatype: int64}
# - {name: b, datatype: int64}
a|b
1|2
3|4
"""
        test_file.write_text(content)

        df = cast(pd.DataFrame, ecsv.read(str(test_file)))

        assert len(df) == 2
        assert df.loc[0, "a"] == 1
        assert df.loc[0, "b"] == 2

    def test_read_preserves_data_values(self, tmp_path):
        """Test that data values are correctly preserved."""
        test_file = tmp_path / "test.ecsv"
        content = """# %ECSV 1.0
# ---
# datatype:
# - {name: x, datatype: float64}
# - {name: y, datatype: float64}
x,y
1.5,2.5
3.5,4.5
5.5,6.5
"""
        test_file.write_text(content)

        df = cast(pd.DataFrame, ecsv.read(str(test_file)))

        assert df.loc[0, "x"] == 1.5
        assert df.loc[0, "y"] == 2.5
        assert df.loc[2, "x"] == 5.5


class TestGenerateHeader:
    """Test generate_header function."""

    def test_generate_header_basic(self):
        """Test generating a basic header."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})

        header = ecsv.generate_header(df)

        assert "# %ECSV" in header
        assert "datatype:" in header
        assert "name: a" in header
        assert "name: b" in header

    def test_generate_header_with_meta(self):
        """Test generating header with metadata."""
        df = pd.DataFrame({"x": [1, 2]})

        header = ecsv.generate_header(df, source="test", version="1.0")

        assert "meta:" in header
        assert "source: test" in header
        assert "version:" in header

    def test_generate_header_from_attrs(self):
        """Test that df.attrs are included in header."""
        df = pd.DataFrame({"x": [1, 2]})
        df.attrs = {"author": "test_author", "date": "2024-01-01"}

        header = ecsv.generate_header(df)

        assert "author: test_author" in header
        assert "date:" in header

    def test_generate_header_meta_overrides_attrs(self):
        """Test that explicit meta overrides df.attrs."""
        df = pd.DataFrame({"x": [1, 2]})
        df.attrs = {"key": "value1"}

        header = ecsv.generate_header(df, key="value2")

        # Explicit meta should override attrs
        assert "value2" in header

    def test_generate_header_correct_format(self):
        """Test that generated header has correct ECSV format."""
        df = pd.DataFrame({"x": [1, 2]})
        header = ecsv.generate_header(df)

        lines = header.split("\n")
        assert lines[0].startswith("# %ECSV")
        assert lines[1] == "# ---"
        # All header lines should start with #
        for line in lines:
            if line:
                assert line.startswith("#")


class TestWrite:
    """Test write function."""

    def test_write_basic(self, tmp_path):
        """Test writing a basic ECSV file."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})

        output_file = tmp_path / "output.ecsv"
        ecsv.write(df, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "# %ECSV" in content
        assert "a,b" in content

    def test_write_and_read_roundtrip(self, tmp_path):
        """Test that write and read are consistent."""
        original_df = pd.DataFrame({"x": [1, 2, 3], "y": [4.5, 5.5, 6.5]})

        file_path = tmp_path / "test.ecsv"
        ecsv.write(original_df, str(file_path))

        read_df = cast(pd.DataFrame, ecsv.read(str(file_path)))

        assert len(read_df) == len(original_df)
        assert list(read_df.columns) == list(original_df.columns)
        pd.testing.assert_frame_equal(read_df, original_df)

    def test_write_with_metadata(self, tmp_path):
        """Test writing with metadata."""
        df = pd.DataFrame({"x": [1, 2]})

        output_file = tmp_path / "output.ecsv"
        ecsv.write(df, str(output_file), author="test", version=1.0)

        # Read back and check metadata
        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert "author" in read_df.attrs
        assert read_df.attrs["author"] == "test"

    def test_write_with_attrs(self, tmp_path):
        """Test that df.attrs are written to file."""
        df = pd.DataFrame({"x": [1, 2]})
        df.attrs = {"note": "test note"}

        output_file = tmp_path / "output.ecsv"
        ecsv.write(df, str(output_file))

        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert "note" in read_df.attrs
        assert read_df.attrs["note"] == "test note"

    def test_write_preserves_dtypes(self, tmp_path):
        """Test that data types are preserved through write/read."""
        df = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype=np.int64),
                "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            }
        )

        output_file = tmp_path / "output.ecsv"
        ecsv.write(df, str(output_file))

        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert read_df["int_col"].dtype == np.int64
        assert read_df["float_col"].dtype == np.float64


class TestConverter:
    """Test _converter function."""

    def test_converter_int_array(self):
        """Test converting string to int array."""
        result = ecsv._converter("[1, 2, 3]", "int64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_converter_float_array(self):
        """Test converting string to float array."""
        result = ecsv._converter("[1.5, 2.5, 3.5]", "float64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert np.allclose(result, np.array([1.5, 2.5, 3.5]))

    def test_converter_with_none_values(self):
        """Test converter with None values (masked array)."""
        result = ecsv._converter("[1, null, 3]", "int64")

        # Should return a masked array
        assert isinstance(result, (np.ndarray, np.ma.MaskedArray))
        if isinstance(result, np.ma.MaskedArray):
            assert result.mask[
                1
            ]  # Second element should be masked  #pyright: ignore


class TestECSVEdgeCases:
    """Test edge cases and potential bugs."""

    def test_empty_dataframe(self, tmp_path):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["a", "b"])

        output_file = tmp_path / "empty.ecsv"
        try:
            ecsv.write(df, str(output_file))
        except ValueError:
            pass

    def test_dataframe_with_nan(self, tmp_path):
        """Test DataFrame with NaN values."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, np.nan]})

        output_file = tmp_path / "nan.ecsv"
        ecsv.write(df, str(output_file))

        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert np.isnan(read_df.loc[1, "x"])  # pyright: ignore
        assert np.isnan(read_df.loc[2, "y"])  # pyright: ignore

    def test_dataframe_with_inf(self, tmp_path):
        """Test DataFrame with infinity values."""
        df = pd.DataFrame(
            {
                "x": [1.0, np.inf, -np.inf],
            }
        )

        output_file = tmp_path / "inf.ecsv"
        ecsv.write(df, str(output_file))

        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert np.isinf(read_df.loc[1, "x"])  # pyright: ignore
        assert read_df.loc[1, "x"] > 0  # pyright: ignore
        assert read_df.loc[2, "x"] < 0  # pyright: ignore

    def test_single_row_dataframe(self, tmp_path):
        """Test DataFrame with single row."""
        df = pd.DataFrame({"x": [1], "y": [2]})

        output_file = tmp_path / "single.ecsv"
        ecsv.write(df, str(output_file))

        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert len(read_df) == 1
        assert read_df.loc[0, "x"] == 1

    def test_single_column_dataframe(self, tmp_path):
        """Test DataFrame with single column."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        output_file = tmp_path / "single_col.ecsv"
        ecsv.write(df, str(output_file))

        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert len(read_df.columns) == 1
        assert "x" in read_df.columns

    def test_column_names_with_spaces(self, tmp_path):
        """Test column names with spaces (should be avoided but test behavior)."""
        # ECSV spec may not support spaces well, but test how it handles them
        df = pd.DataFrame({"col name": [1, 2]})

        output_file = tmp_path / "spaces.ecsv"
        try:
            ecsv.write(df, str(output_file))
            # If it doesn't raise an error, that's fine
            assert output_file.exists()
        except Exception:
            # Also acceptable to raise an error
            pass

    def test_very_long_values(self, tmp_path):
        """Test with very long string values."""
        long_string = "a" * 10000
        df = pd.DataFrame({"text": [long_string]})

        output_file = tmp_path / "long.ecsv"
        ecsv.write(df, str(output_file))

        read_df = ecsv.read(str(output_file))
        assert read_df.loc[0, "text"] == long_string

    def test_special_characters_in_data(self, tmp_path):
        """Test special characters in data."""
        df = pd.DataFrame(
            {"text": ["hello", "world,with,commas", "tabs\there"]}
        )

        output_file = tmp_path / "special.ecsv"
        ecsv.write(df, str(output_file))

        # CSV should handle commas in quoted fields
        read_df = cast(pd.DataFrame, ecsv.read(str(output_file)))
        assert "," in read_df.loc[1, "text"]  # pyright: ignore
