"""
The ECSV format allows for specification of key table and column meta-data, in
particular the data type and unit.

The "Comma-Separated-Values" format is probably the most common text-based method
for representing data tables. The proposed standard in APE6 leverages this
universality by using this format for serializing the actual data values.

For science applications, pure CSV has a serious shortcoming in the complete
lack of support for table metadata. This is frequently not a showstopper but
represents a serious limitation which has motivated alternate standards in the
astronomical community.

The proposed Enhanced CSV (ECSV) format has the following overall structure:

A header section which consists of lines that start with the # character and
provide the table definition and data format via a YAML-encoded data structure.
An initial line in the header section which identifies the file as ECSV and
provides a version number.  A CSV-formatted data section in which the first line
contains the column names and subsequent lines contains the data values.
Version 1.0 of the ECSV format specification and the reference Python
implementation assumes ASCII-encoded header and data sections. Support for
unicode (in particular UTF-8) may be added in subsequent versions.

.. note::

   "Comma-Separated-Values" (CSV) is a misleading name since tab-separated or
   whitespace-separated tabular data generally fall in this category.
   We should use "Character-Separated-Values", which keep the same acronym.


example of file::

    # %ECSV 1.0
    # ---
    # datatype:
    # - {name: a, unit: m / s, datatype: int64, format: '%03d'}
    # - {name: b, unit: km, datatype: int64, description: This is column b}
    a b
    1 2
    4 3

.. seealso::

    `Enhanced Character Separated Values table format description
    <https://github.com/astropy/astropy-APEs/blob/main/APE6.rst>`_ from the
    Astropy project.

"""
import textwrap
import re
import yaml
import pandas as pd
import numpy as np
from typing import Union
from io import TextIOWrapper


__ECSV_VERSION__ = '1.0'


def read_header(fname: str) -> dict:
    """ read the header of ECSV file as a dictionary

    Parameters
    ----------
    fname:  str
        The name of the file to read.

    Returns
    -------
    data: dict
        the header of the file.
    """

    def process_header_lines(fname: str, comment: str = '#') -> str:
        """ Return header lines if non-blank and starting with the comment char
            Empty lines are discarded.
        """
        re_comment = re.compile(comment)

        with open(fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                match = re_comment.match(line)
                if match:
                    out = line[match.end():]
                    if out:
                        yield out
                else:
                    return

    header = yaml.load(textwrap.dedent('\n'.join(process_header_lines(fname))),
                        yaml.SafeLoader)

    return header

def read(fname: str, **kwargs) -> pd.DataFrame:
    """ Read the content of an Enhanced Character Separated Values

    Parameters
    ----------
    fname:  str
        The name of the file to read.

    Returns
    -------
    data:  pd.DataFrame
        The data read from the file.
    """
    header = read_header(fname)

    dtype_mapper = {'string': str}

    dtype = {k['name']: np.dtype(dtype_mapper.get(k['datatype'], k['datatype']))
                                    for k in header['datatype']}

    delimiter = header.get('delimiter', kwargs.pop('delimiter', ','))
    comment = kwargs.pop('comment', '#')

    df = pd.read_csv(fname, delimiter=delimiter, dtype=dtype, comment=comment, **kwargs)
    df.attrs.update(header.get('meta', {}))
    return df


def generate_header(df: pd.DataFrame, **meta) -> str:
    """ Generates the yaml equivalent string for the ECSV header

    Parameters
    ----------
    df: pd.DataFrame
        data to be written to the file.
    meta: dict
        meta data to be written to the header.
        Typically keywords, comments, history and so forth should be part of meta.
        df.attrs will be automatically added to the meta data.

    Returns
    -------
    header: str
        the header corresponding to the data.
    """
    dtypes = [{'name': name, 'datatype': str(dt)}
                for name, dt in df.dtypes.to_dict().items()]
    meta_ = df.attrs.copy()
    meta_.update(meta)
    h = {'delimiter': ',', 'datatype': dtypes, 'meta': meta_}
    preamble = ['# %ECSV {0:s}'.format(__ECSV_VERSION__), '# ---']
    lines = ['# ' + line for line in yaml.dump(h, sort_keys=False).split('\n') if line]
    return '\n'.join(preamble + lines)


def write(df: pd.DataFrame,
          fname: Union[str, TextIOWrapper],
          mode: str = 'w', **meta):
    """ output data into ecsv file

    Parameters
    ----------
    df: pd.DataFrame
        data to be written to the file.
    fname: str
        the name of the file to write.
    mode: str
        the mode to open the file.
    meta: dict
        meta data to be written to the header.
    """
    if hasattr(fname, 'write'):
       fname.write(generate_header(df, **meta) + '\n')
       df.to_csv(fname, index=False)
    else:
        with open(fname, mode) as fout:
            fout.write(generate_header(df, **meta) + '\n')
            df.to_csv(fout, index=False)