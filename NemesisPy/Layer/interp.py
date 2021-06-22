#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.interpolate import interp1d

def interp(X_data, Y_data, X, ITYPE=1):
    """
    Routine for 1D interpolation using the SciPy library.

    Inputs
    ------
    @param X_data: 1D array

    @param Y_data: 1D array

    @param X: real

    @param ITYPE: int
        1=linear interpolation
        2=quadratic spline interpolation
        3=cubic spline interpolation
    """
    if ITYPE == 1:
        interp = interp1d
        f = interp1d(X_data, Y_data, kind='linear', fill_value='extrapolate')
        Y = f(X)

    elif ITYPE == 2:
        interp = interp1d
        f = interp(X_data, Y_data, kind='quadratic', fill_value='extrapolate')
        Y = f(X)

    elif ITYPE == 3:
        interp = interp1d
        f = interp(X_data, Y_data, kind='cubic', fill_value='extrapolate')
        Y = f(X)

    return Y