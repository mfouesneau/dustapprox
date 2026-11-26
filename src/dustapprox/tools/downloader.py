"""A simple tool to download files from URLs"""

import requests
import sys
import os
import time
from typing import Sequence, Iterator, Union
from io import TextIOWrapper

__all__ = ["download_file"]


def _pretty_size_print(num_bytes: int) -> str:
    """
    Output number of bytes in a human readable format

    parameters
    ----------
    num_bytes: int
        number of bytes to convert

    returns
    -------
    output: str
        string representation of the size with appropriate unit scale
    """
    if num_bytes is None:
        return

    KiB = 1024
    MiB = KiB * KiB
    GiB = KiB * MiB
    TiB = KiB * GiB
    PiB = KiB * TiB
    EiB = KiB * PiB
    ZiB = KiB * EiB
    YiB = KiB * ZiB

    if num_bytes > YiB:
        output = "%.3g YB" % (num_bytes / YiB)
    elif num_bytes > ZiB:
        output = "%.3g ZB" % (num_bytes / ZiB)
    elif num_bytes > EiB:
        output = "%.3g EB" % (num_bytes / EiB)
    elif num_bytes > PiB:
        output = "%.3g PB" % (num_bytes / PiB)
    elif num_bytes > TiB:
        output = "%.3g TB" % (num_bytes / TiB)
    elif num_bytes > GiB:
        output = "%.3g GB" % (num_bytes / GiB)
    elif num_bytes > MiB:
        output = "%.3g MB" % (num_bytes / MiB)
    elif num_bytes > KiB:
        output = "%.3g KB" % (num_bytes / KiB)
    else:
        output = "%.3g Bytes" % (num_bytes)

    return output


def _dl_ascii_progress(
    iterseq: Union[Sequence, Iterator],
    total: int = 100,
    progress_length: int = 50,
    mininterval: float = 2,
    buffer: TextIOWrapper = sys.stdout,  # pyright: ignore / dumb
):
    """A simplistic progress indicator in ascii format applicable to a sequence

    writes to sys.stdout

    Parameters
    ----------
    iterseq: Sequence
        sequence to iter over
    total: int
        length of the sequence (default is 100 or len(iterseq) when possible)
    progress_length: int
        number of characters used by the indicator
    mininterval: float
        how long to wait before updating the indicator (default 0.5 seconds)
    buffer: TextIOWrapper
        where to write the indicator to (default sys.stdout)
    """
    dl = 0
    message_length = 0
    try:
        total = len(iterseq)  # pyright: ignore / exception caught
    except TypeError:
        pass

    last_print_t = time.time()

    for chunk in iterseq:
        try:
            dl += len(chunk)
        except TypeError:
            dl += 1
        cur_t = time.time()
        if cur_t - last_print_t >= mininterval:
            # unknown/zero total (e.g. no content-length): no fraction to show
            done = (
                min(progress_length, int(progress_length * dl / total))
                if total
                else 0
            )
            message = "\r[{}{}] ({})".format(
                "=" * done,
                " " * (progress_length - done),
                _pretty_size_print(dl),
            )
            clear = " " * (max(1, message_length - len(message)))
            sys.stdout.write(message + clear)
            message_length = len(message)
            sys.stdout.flush()
        yield (chunk)
    buffer.write("\n")
    buffer.flush()


def download_file(
    link: str,
    file_name: str,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> str:
    """Download a file on disk from url

    Parameters
    ----------
    link: str
        url of the file
    file_name: str
        path and filename of the download location
    overwrite: bool
        set to re-download (default False)
    timeout: float
        seconds to wait for the server (connect and between bytes)

    Returns
    -------
    Returns the filename of the data

    Raises
    ------
    requests.HTTPError
        if the server answers with an error status (e.g. 404), instead of
        saving the error page as the file.
    """
    response = requests.get(link, stream=True, timeout=timeout)
    response.raise_for_status()
    content_length = response.headers.get("content-length")
    total_length = int(content_length) if content_length else None
    if os.path.exists(file_name) and not overwrite:
        # without a content-length we cannot tell a partial file: re-download
        if (total_length is not None) and (
            os.stat(file_name).st_size == total_length
        ):
            print(f"file '{file_name}' already downloaded.")
            return file_name

    with open(file_name, "wb") as f:
        print(f"Downloading '{file_name}'", end="")
        print(f" ({_pretty_size_print(total_length) or 'unknown size':s})")

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            for data in _dl_ascii_progress(
                response.iter_content(chunk_size=4096), total=total_length
            ):
                f.write(data)
    return file_name
