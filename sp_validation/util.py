# -*- coding: utf-8 -*-

"""
  
:Name: util.py

:Description: This script contains utility methods.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import sys
import os

import math
import numpy as np

try:
    import vos.commands as vosc
except ImportError:  # pragma: no cover
    import_fail = True
else:
    import_fail = False



def millify(n):
    """Millify

    Return human-readible names of large numbers

    Parameters
    ----------
    n : int
        input number

    Returns
    -------
    s : string
        output name
    """

    millnames = ['',' Thousand',' Million',' Billion',' Trillion']

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        )
    )

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def equi_num_bins(x, n_bin):
    """ Equi Num Bins

    Returns (n_bin+1) equi-numbered bin edges of x. They define n_bin bins,
    each of which contains an equal number of points of x

    Parameters
    ----------
    x : list
        input data
    n_bins : int
        number of bins

    Returns
    -------
    numpy.array :
        equi-numbered bin array

    """
    xeqn = np.interp(
        np.linspace(0, len(x), n_bin + 1),
        np.arange(len(x)),
        np.sort(x)
    )

    return xeqn


def transform_nan(value):
    """Transform Nan

    Transform a nan to a very large number.

    Parameters
    ----------
    value : float
        input value

    Returns
    -------
    res : float
        output value
    """

    large = 1e30

    if np.isnan(value) or np.isinf(value):
        res = 1e30
    else:
        res = value

    return res


class vosError(Exception):
    """VOS Error

    Generic error that is raised by the vosHandler.

    """
    pass


class vosHandler:
    """VOS Handler

    This class manages the use of VOS commands.

    Parameters
    ----------
    command : str
        VOS command name

    """

    def __init__(self, command):

        self._check_vos_install()
        self._avail_commands = tuple(vosc.__all__)
        self.command = command

    @staticmethod
    def _check_vos_install():
        """Check VOS Install

        Check if VOS is correctly installed.

        """
        if import_fail:
            raise ImportError(
                'vos package not found, install with \'pip install vos\''
            )

    @property
    def command(self):
        """Command

        This method sets the VOS command property.

        """
        return self._command

    @command.setter
    def command(self, value):

        if value not in self._avail_commands:
            raise ValueError('vos command must be one of {}'
                             ''.format(self._avail_commands))

        self._command = getattr(vosc, value)

    def __call__(self, *args, **kwargs):
        """Call Method

        This method allows class instances to be called as functions.

        """
        try:
            self._command()

        except Exception:
            raise vosError(f'Error in VOs command: {self._command.__name__}')


def download(source, target, verbose=False):
    """Download

    Download file from vos.

    Parameters
    ----------
    source : string
        source path on vos
    target : string
        target path
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    status : bool
        status, True/False or success/failure
    """

    cmd = 'vcp'

    if not os.path.exists(target):
        sys.argv = [cmd, source, target]
        if verbose:
            print('Downloading file {} to {}...'.format(source, target))
        vcp = vosHandler(cmd)

        vcp()
        if verbose:
            print('Download finished.')
    else:
        if verbose:
            print('Target file {} exists, skipping download.'.format(target))

