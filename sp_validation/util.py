"""UTIL.

:Description: This script contains utility methods.

:Author: Martin Kilbinger

:Author: Martin Kilbinge <martin.kilblinger@cea.fr>

"""

import sys
import os

import math
import numpy as np
from scipy import stats

try:
    import vos.commands as vosc
except ImportError:  # pragma: no cover
    import_fail = True
else:
    import_fail = False


def millify(n):
    """Millify.

    Return human-readible names of large numbers

    Parameters
    ----------
    n : int
        input number

    Returns
    -------
    str
        output name

    """
    millnames = ['', ' Thousand', ' Million', ' Billion', ' Trillion']

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        )
    )

    return '{n / 10**(3 * millidx):.0f}{millnames[millidx])}'


def equi_num_bins(values, n_bin):
    """Equi Num Bins.

    Returns (n_bin+1) equi-numbered bin edges of values. They define n_bin
    bins, each of which contains an equal number of points of values.

    Parameters
    ----------
    values : list
        input data
    n_bins : int
        number of bins

    Returns
    -------
    numpy.array :
        equi-numbered bin array

    """
    xeqn = np.interp(
        np.linspace(0, len(values), n_bin + 1),
        np.arange(len(values)),
        np.sort(values)
    )

    return xeqn


def transform_nan(value):
    """Transform Nan.

    Transform a ``nan`` to a very large number.

    Parameters
    ----------
    value : float
        input value

    Returns
    -------
    float
        output value
    """
    large = 1e30

    if np.isnan(value) or np.isinf(value):
        res = 1e30
    else:
        res = value

    return res


class vosError(Exception):
    """VOS Error.

    Generic error that is raised by the vosHandler.

    """

    pass


class vosHandler:
    """VOS Handler.

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
        """Check VOS Install.

        Check if VOS is correctly installed.

        """
        if import_fail:
            raise ImportError(
                'vos package not found, install with \'pip install vos\''
            )

    @property
    def command(self):
        """Command.

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
        """Call Method.

        This method allows class instances to be called as functions.

        """
        try:
            self._command()

        except Exception:
            raise vosError(f'Error in VOs command: {self._command.__name__}')


def download(source, target, verbose=False):
    """Download.

    Download file from vos.

    Parameters
    ----------
    source : str
        source path on vos
    target : str
        target path
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    bool
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


def compute_bins_func_2d(x, y, n_bin, mix, weights=None):
    """Compute Bins Func 2D.

    Compute bins in x, y, err, for 2D model

    Parameters
    ----------
    x : 2D numpy.ndarray
        x_1, x_2-values
    y : 2D numpy.ndarray
        y_1, y_2-values
    n_bin : int
        number of bins to create
    mix : bool
        mixing of component if True
    weights  : numpy.ndarrayarray of double, optional, default=None
        weights of x points

    Returns
    -------
    numpy.ndarray(float, 2, n_bin)
        bin centers in x_1, x_2
    numpy.ndarray(float, 2, 2, n_bin)
        binned values of y_1, y_2 corresponding to x_1, x_2 bins
    numpy.ndarray(float, 2, 2, n_bin)
        binned errors of y_1, y_2 corresponding to x_1, x_2 bins

    """
    # Compute bins in x

    # Initialise bins and edges for x
    x_bin = np.zeros(shape=(2, n_bin))
    x_edges = np.zeros(shape=(2, n_bin + 1))

    # Loop over both components, compute equi-numbered bins
    for comp in (0, 1):
        xeqn = equi_num_bins(x[comp], n_bin)
        res = stats.binned_statistic(x[comp], x[comp], 'mean', bins=xeqn)
        x_bin[comp] = res.statistic
        x_edges[comp] = res.bin_edges

    # Compute bins in y and errors

    # Initialise
    y_bin = np.zeros(shape=(2, 2, n_bin))
    err_bin = np.zeros(shape=(2, 2, n_bin))

    # Loop over both components corresponding to x (comp_x),
    # and y and err (comp_y)
    for comp_x in (0, 1):
        for comp_y in (0, 1):

            # No mixing and different y-/x-component: not used
            if not mix and (comp_x != comp_y):
                continue

            # 1d y bins
            if weights is None:
                y_bin[comp_y][comp_x] = stats.binned_statistic(
                    x[comp_x],
                    y[comp_y],
                    'mean',
                    bins=x_edges[comp_x]
                ).statistic
            else:
                yw = stats.binned_statistic(
                    x[comp_x],
                    y[comp_y] * weights,
                    'sum',
                    bins=x_edges[comp_x]
                ).statistic
                w = stats.binned_statistic(
                    x[comp_x],
                    weights,
                    'sum',
                    bins=x_edges[comp_x]
                ).statistic
                y_bin[comp_y][comp_x] = yw / w

            # 1d numbers
            n = stats.binned_statistic(
                x[comp_x],
                y[comp_y],
                'count',
                bins=x_edges[comp_x]
            ).statistic

            # 1d errors of the mean = standard deviation devided by sqrt
            # of the numbers
            err_bin[comp_y][comp_x] = stats.binned_statistic(
                x[comp_x],
                y[comp_y],
                'std',
                bins=x_edges[comp_x]
            ).statistic / np.sqrt(n)

    return x_bin, y_bin, err_bin


def func_bias_2d_full(params, x1, x2, order='lin', mix=False):
    """Func Bias 2D Full.

    Function of 2D bias model evaluated on full 2D grid.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1 : list of float
        first component of x-values
    x2 : list of float
        second component of x-values
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    2D np.array of float
        first component the 2D model y1(x1, x2) on the (x1, x2)-grid
    2D np.array of float
        second component the 2D model, y2(x1, x2) on the (x1, x2)-grid

    """
    len1 = len(x1)
    len2 = len(x2)

    # Initialise both components y1, y2 as 2D arrays
    y1 = np.zeros(shape=(len1, len2))
    y2 = np.zeros(shape=(len1, len2))

    # Create 2D mesh for input x1, x2 values
    v1, v2 = np.meshgrid(x1, x2, indexing='ij')

    # Compute both components y1, y2 over the meash
    y1, y2 = func_bias_2d(params, v1, v2, order=order, mix=mix)

    return y1, y2


def func_bias_2d(params, x1_data, x2_data, order='lin', mix=False):
    """Func Bias 2D.

    Function of 2D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1_data : float or list of float
        first component of x-values of the data
    x2_data : float or list of float
        second component of x-values of the data
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    float or list of float
        first component the 2D model, y1(x1, x2). Dimension
        is equal to x1_data and x2_data
    float or list of float
        second component the 2D model, y2(x1, x2). Dimension
        is equal to x1_data and x2_data

    """
    # Get affine parameters
    a11 = params['a11'].value
    a22 = params['a22'].value
    c1 = params['c1'].value
    c2 = params['c2'].value

    # Compute y-values for affine model
    y1_model = a11 * x1_data + c1
    y2_model = a22 * x2_data + c2

    if order == 'quad':
        # Add quadratic part
        q111 = params['q111'].value
        q222 = params['q222'].value
        y1_model += q111 * x1_data ** 2
        y2_model += q222 * x2_data ** 2

    if mix:
        # Add linear mixing part
        a12 = params['a12'].value
        y1_model += a12 * x2_data
        y2_model += a12 * x1_data

        if order == 'quad':
            # Add quadratic mixing part
            q112 = params['q112'].value
            q122 = params['q122'].value
            q212 = params['q212'].value
            q211 = params['q211'].value
            y1_model += q112 * x1_data * x2_data + q122 * x2_data ** 2
            y2_model += q212 * x1_data * x2_data + q211 * x1_data ** 2

    return y1_model, y2_model


def log_command(argv, name=None, close_no_return=True):                         
    """Log Command.                                                             
                                                                                
    Write command with arguments to a file or stdout.                           
    Choose name = 'sys.stdout' or 'sys.stderr' for output on sceen.             

    MKDEBUG copied from shapepipe:cfis
                                                                                
    Parameters                                                                  
    ----------                                                                  
    argv : list
        Command line arguments                                                  
    name : str                                                                  
        Output file name (default: 'log_<command>')                             
    close_no_return : bool                                                      
        If True (default), close log file. If False, keep log file open         
        and return file handler                                                 
                                                                                
    Returns                                                                     
    -------                                                                     
    filehandler                                                                 
        log file handler (if close_no_return is False)                          
                                                                                
    """                                                                         
    if name is None:                                                            
        name = 'log_' + os.path.basename(argv[0])                               
                                                                                
    if name == 'sys.stdout':                                                    
        f = sys.stdout                                                          
    elif name == 'sys.stderr':                                                  
        f = sys.stderr                                                          
    else:                                                                       
        f = open(name, 'w')                                                     
                                                                                
    for a in argv:                                                              
                                                                                
        # Quote argument if special characters                                  
        if ']' in a or ']' in a:                                                
            a = f'\"{a}\"'                                                      
                                                                                
        print(a, end='', file=f)                                                
        print(' ', end='', file=f)                                              
                                                                                
    print('', file=f)                                                           
                                                                                
    if not close_no_return:                                                     
        return f                                                                
                                                                                
    if name != 'sys.stdout' and name != 'sys.stderr':                           
        f.close() 
