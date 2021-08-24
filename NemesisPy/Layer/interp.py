#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.interpolate import interp1d
import numpy as np

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

def interpg(X_data, Y_data, X):
    """
    Routine for 1D interpolation

    Inputs
    ------
    @param X_data: 1D array

    @param Y_data: 1D array

    @param X: real

    @param ITYPE: int
        1=linear interpolation
    """

    from NemesisPy import find_nearest

    NX = len(X)
    Y = np.zeros(NX)
    J = np.zeros(NX,dtype='int32')
    F = np.zeros(NX)
    for IX in range(NX):

        j = 0
        while X_data[j]<X[IX]:
            j = j + 1
        
        if j==0:
            j = 1
        J[IX] = j - 1
        F[IX] = (X[IX]-X_data[j-1])/(X_data[j]-X_data[j-1])
        Y[IX] = (1.0-F[IX])*Y_data[j-1] + F[IX]*Y_data[j]

    return Y,J,F

