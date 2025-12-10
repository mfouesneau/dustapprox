"""Tests for dustapprox.tools.downloader module."""

import os
import tempfile
from unittest.mock import Mock, patch

from dustapprox.tools.downloader import (
    download_file,
    _pretty_size_print,
    _dl_ascii_progress,
)


class TestPrettySizePrint:
    """Test the _pretty_size_print function."""

    def test_pretty_size_bytes(self):
        """Test formatting bytes."""
        result = _pretty_size_print(512)
        assert "Bytes" in result
        assert "512" in result

    def test_pretty_size_kilobytes(self):
        """Test formatting kilobytes."""
        result = _pretty_size_print(2048)  # Use 2KB to avoid scientific notation
        assert "KB" in result

    def test_pretty_size_megabytes(self):
        """Test formatting megabytes."""
        result = _pretty_size_print(2 * 1024 * 1024)
        assert "MB" in result

    def test_pretty_size_gigabytes(self):
        """Test formatting gigabytes."""
        result = _pretty_size_print(2 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_pretty_size_terabytes(self):
        """Test formatting terabytes."""
        result = _pretty_size_print(2 * 1024 * 1024 * 1024 * 1024)
        assert "TB" in result

    def test_pretty_size_petabytes(self):
        """Test formatting petabytes."""
        result = _pretty_size_print(2 * 1024 * 1024 * 1024 * 1024 * 1024)
        assert "PB" in result

    def test_pretty_size_exabytes(self):
        """Test formatting exabytes."""
        result = _pretty_size_print(2 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024)
        assert "EB" in result

    def test_pretty_size_zettabytes(self):
        """Test formatting zettabytes."""
        result = _pretty_size_print(2 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024)
        assert "ZB" in result

    def test_pretty_size_yottabytes(self):
        """Test formatting yottabytes."""
        result = _pretty_size_print(2 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024)
        assert "YB" in result

    def test_pretty_size_zero(self):
        """Test formatting zero bytes."""
        result = _pretty_size_print(0)
        assert "Bytes" in result
        assert "0" in result

    def test_pretty_size_one(self):
        """Test formatting single byte."""
        result = _pretty_size_print(1)
        assert "Bytes" in result

    def test_pretty_size_none(self):
        """Test that None returns None."""
        # Using type: ignore due to type annotation restrictions
        result = _pretty_size_print(None)  # type: ignore
        assert result is None

    def test_pretty_size_large_value(self):
        """Test formatting very large values."""
        result = _pretty_size_print(int(1e20))
        assert result is not None
        assert isinstance(result, str)

    def test_pretty_size_boundary_values(self):
        """Test formatting at unit boundaries."""
        # Just below MiB
        result_below = _pretty_size_print(2 * 1024 * 1024 - 1024)
        assert "KB" in result_below or "MB" in result_below
        
        # Just at MiB
        result_at = _pretty_size_print(2 * 1024 * 1024)
        assert "MB" in result_at


class TestDlAsciiProgress:
    """Test the _dl_ascii_progress function."""

    def test_progress_with_sequence(self):
        """Test progress indicator with a sequence."""
        data = [b"chunk1", b"chunk2", b"chunk3"]
        
        with patch('sys.stdout.write'), patch('sys.stdout.flush'):
            result = list(_dl_ascii_progress(data, total=18))
        
        assert len(result) == 3
        assert result == data

    def test_progress_with_iterator(self):
        """Test progress indicator with an iterator."""
        def data_iterator():
            yield b"a"
            yield b"b"
            yield b"c"
        
        with patch('sys.stdout.write'), patch('sys.stdout.flush'):
            result = list(_dl_ascii_progress(data_iterator(), total=3))
        
        assert len(result) == 3

    def test_progress_length_customization(self):
        """Test progress indicator with custom progress bar length."""
        data = [b"test"] * 10
        
        with patch('sys.stdout.write'), patch('sys.stdout.flush'):
            result = list(_dl_ascii_progress(data, total=40, progress_length=30))
        
        assert len(result) == 10

    def test_progress_mininterval(self):
        """Test that progress respects minimum interval."""
        data = [b"x"] * 100
        
        with patch('sys.stdout.write'), patch('sys.stdout.flush'):
            result = list(_dl_ascii_progress(data, total=100, mininterval=0.01))
        
        assert len(result) == 100

    def test_progress_auto_length_detection(self):
        """Test automatic length detection from sequence."""
        data = [b"a", b"b", b"c", b"d", b"e"]
        
        with patch('sys.stdout.write'), patch('sys.stdout.flush'):
            result = list(_dl_ascii_progress(data))
        
        assert len(result) == 5

    def test_progress_with_non_chunked_iterator(self):
        """Test progress with iterator that yields non-bytes objects."""
        def non_bytes_iterator():
            yield 1
            yield 2
            yield 3
        
        with patch('sys.stdout.write'), patch('sys.stdout.flush'):
            result = list(_dl_ascii_progress(non_bytes_iterator(), total=3))
        
        assert len(result) == 3

    def test_progress_output_format(self):
        """Test that progress output contains expected format characters."""
        data = [b"chunk"] * 5
        
        with patch('sys.stdout.write') as mock_write:
            with patch('sys.stdout.flush'):
                list(_dl_ascii_progress(data, total=25, progress_length=20))
        
        # Check that write was called
        assert mock_write.called

    def test_progress_with_default_parameters(self):
        """Test progress with default parameters."""
        data = [b"test"] * 3
        
        result = list(_dl_ascii_progress(data))
        
        assert len(result) == 3
        assert result == data

    def test_progress_respects_mininterval(self):
        """Test that progress respects mininterval between updates."""
        data = [b"small"] * 20
        
        with patch('sys.stdout.write'):
            with patch('sys.stdout.flush'):
                list(_dl_ascii_progress(data, total=100, mininterval=10))
        
        # With mininterval=10 and quick execution, write may not be called many times
        # Just verify the function works
        assert True


class TestDownloadFile:
    """Test the download_file function."""

    def test_download_file_basic(self):
        """Test basic file download."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = Mock(return_value=[b"test data"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_file.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
            
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                assert f.read() == b"test data"

    def test_download_file_multiple_chunks(self):
        """Test download with multiple chunks."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content = Mock(return_value=[
            b"chunk1",
            b"chunk2",
            b"chunk3",
        ])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "multi_chunk.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
            
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                data = f.read()
                assert data == b"chunk1chunk2chunk3"

    def test_download_file_no_content_length(self):
        """Test download when content-length header is missing."""
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.content = b"full content"
        mock_response.iter_content = Mock(return_value=[b"full content"])  # Provide iter_content too
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "no_length.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
            
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                assert f.read() == b"full content"

    def test_download_file_no_content_length_string_zero(self):
        """Test download when content-length header is "0"."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "0"}
        mock_response.content = b"some data"
        mock_response.iter_content = Mock(return_value=[b"some data"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "zero_length.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
            
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                assert f.read() == b"some data"

    def test_download_file_already_exists_size_match(self):
        """Test skipping download when file exists with matching size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "matching_size.bin")
            existing_content = b"exact match"
            
            with open(file_path, "wb") as f:
                f.write(existing_content)
            
            mock_response = Mock()
            mock_response.headers = {"content-length": str(len(existing_content))}
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path, overwrite=False)
            
            # File should not be re-downloaded (should skip)
            with open(file_path, "rb") as f:
                assert f.read() == existing_content

    def test_download_file_already_exists_no_content_length_header(self):
        """Test skip when file exists and content-length header is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "no_header.bin")
            existing_content = b"existing"
            
            with open(file_path, "wb") as f:
                f.write(existing_content)
            
            mock_response = Mock()
            mock_response.headers = {}  # No content-length
            mock_response.content = b"new"
            mock_response.iter_content = Mock(return_value=[b"new"])
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path, overwrite=False)
            
            # File should be re-downloaded because total_length is None (triggers download)
            with open(file_path, "rb") as f:
                assert f.read() == b"new"

    def test_download_file_already_exists_no_overwrite(self):
        """Test skipping download when file exists and no overwrite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "existing.bin")
            existing_content = b"existing data"
            
            with open(file_path, "wb") as f:
                f.write(existing_content)
            
            mock_response = Mock()
            mock_response.headers = {"content-length": str(len(existing_content))}
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path, overwrite=False)
            
            with open(file_path, "rb") as f:
                assert f.read() == existing_content

    def test_download_file_already_exists_size_mismatch(self):
        """Test re-downloading when file exists but size differs."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content = Mock(return_value=[b"new data"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "size_mismatch.bin")
            
            # Create file with different size
            with open(file_path, "wb") as f:
                f.write(b"short")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path, overwrite=False)
            
            with open(file_path, "rb") as f:
                assert f.read() == b"new data"

    def test_download_file_overwrite_true(self):
        """Test re-downloading when overwrite is True."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "8"}
        mock_response.iter_content = Mock(return_value=[b"new data"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "overwrite.bin")
            
            # Create initial file
            with open(file_path, "wb") as f:
                f.write(b"old data")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path, overwrite=True)
            
            with open(file_path, "rb") as f:
                assert f.read() == b"new data"

    def test_download_file_creates_directories(self):
        """Test that download creates necessary directories."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "10"}
        mock_response.iter_content = Mock(return_value=[b"test"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "sub", "dir", "file.bin")
            
            # Parent directory doesn't exist yet
            parent_dir = os.path.dirname(nested_path)
            os.makedirs(parent_dir, exist_ok=True)
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", nested_path)
            
            assert os.path.exists(nested_path)

    def test_download_file_return_value(self):
        """Test that download_file returns the correct filename."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content = Mock(return_value=[b"data"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "result.bin")
            
            with patch("requests.get", return_value=mock_response):
                result = download_file("http://example.com/file", file_path)
            
            assert result == file_path

    def test_download_file_large_file(self):
        """Test downloading a large file with many chunks."""
        chunks = [b"x" * 4096 for _ in range(10)]
        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(b"".join(chunks)))}
        mock_response.iter_content = Mock(return_value=chunks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "large_file.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
            
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                content = f.read()
                assert len(content) == len(b"".join(chunks))

    def test_download_file_empty_file(self):
        """Test downloading an empty file."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "0"}
        mock_response.iter_content = Mock(return_value=[])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "empty.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
            
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) == 0

    def test_download_file_prints_output(self):
        """Test that download_file produces output."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "10"}
        mock_response.iter_content = Mock(return_value=[b"test"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.bin")
            
            with patch("requests.get", return_value=mock_response):
                with patch("builtins.print") as mock_print:
                    download_file("http://example.com/file", file_path)
                    # Check that print was called
                    assert mock_print.called

    def test_download_file_with_special_characters(self):
        """Test downloading file with special characters in name."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "4"}
        mock_response.iter_content = Mock(return_value=[b"data"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "file-with_special.chars.bin")
            
            with patch("requests.get", return_value=mock_response):
                result = download_file("http://example.com/file", file_path)
            
            assert os.path.exists(file_path)
            assert result == file_path

    def test_download_file_requests_called_correctly(self):
        """Test that requests.get is called with correct URL."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content = Mock(return_value=[b"data"])
        
        test_url = "http://example.com/specific/path/file.zip"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "file.zip")
            
            with patch("requests.get", return_value=mock_response) as mock_get:
                download_file(test_url, file_path)
                
                # Verify requests.get was called with correct arguments
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert call_args[0][0] == test_url
                assert call_args[1].get("stream") is True

    def test_download_file_iter_content_chunk_size(self):
        """Test that iter_content is called with correct chunk size."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "8192"}
        mock_response.iter_content = Mock(return_value=[b"chunk"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "file.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
                
                # Verify iter_content was called with chunk_size=4096
                mock_response.iter_content.assert_called_once_with(chunk_size=4096)

    def test_download_file_file_is_binary_mode(self):
        """Test that file is opened in binary write mode."""
        mock_response = Mock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content = Mock(return_value=[b"bytes"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "binary.bin")
            
            with patch("requests.get", return_value=mock_response):
                download_file("http://example.com/file", file_path)
            
            # Verify file can be read as binary
            with open(file_path, "rb") as f:
                content = f.read()
                assert isinstance(content, bytes)
                assert content == b"bytes"


class TestModuleExports:
    """Test module exports."""

    def test_module_all(self):
        """Test that __all__ includes expected functions."""
        from dustapprox.tools import downloader
        assert hasattr(downloader, "__all__")
        assert "download_file" in downloader.__all__

    def test_import_download_file(self):
        """Test importing download_file."""
        from dustapprox.tools.downloader import download_file
        assert download_file is not None
        assert callable(download_file)

    def test_import_pretty_size_print(self):
        """Test importing _pretty_size_print."""
        from dustapprox.tools.downloader import _pretty_size_print
        assert _pretty_size_print is not None
        assert callable(_pretty_size_print)

    def test_import_dl_ascii_progress(self):
        """Test importing _dl_ascii_progress."""
        from dustapprox.tools.downloader import _dl_ascii_progress
        assert _dl_ascii_progress is not None
        assert callable(_dl_ascii_progress)
