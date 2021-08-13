# NAME:
#       utils.py (nemesislib)
#
# DESCRIPTION:
#
#	This library contains useful functions to perform calculations using the NEMESIS
#       radiative transfer algorithm
#
# CATEGORY:
#
#	NEMESIS
# 
# MODIFICATION HISTORY: Juan Alday 15/03/2021

import numpy as np
from struct import *
import pylab
import sys,os,errno,shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as font_manager
import matplotlib as mpl

###############################################################################################

###############################################################################################

def unique(array):

    """
    FUNCTION NAME : unique()

    DESCRIPTION : Find the unique values in a list

    INPUTS : 

        array :: List of numbers

    OPTIONAL INPUTS: none
            
    OUTPUTS : 
 
        array_uniq :: Array containing the unique values 

    CALLING SEQUENCE:

        array_uniq = unique(array)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

#    # insert the list to the set 
#    list_set = set(array) 
#    # convert the set to the list 
#    array_uniq = (list(list_set)) 

    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in array: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x)

    array_uniq = unique_list

    return array_uniq


###############################################################################################

def file_lines(fname):

    """
    FUNCTION NAME : file_lines()

    DESCRIPTION : Returns the number of lines in a given file

    INPUTS : 
 
        fname :: Name of the file

    OPTIONAL INPUTS: none
            
    OUTPUTS : 
 
        nlines :: Number of lines in file

    CALLING SEQUENCE:

        nlines = file_lines(fname)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

###############################################################################################

def find_nearest(array, value):

    """
    FUNCTION NAME : find_nearest()

    DESCRIPTION : Find the closest value in an array

    INPUTS : 

        array :: List of numbers
        value :: Value to search for

    OPTIONAL INPUTS: none
            
    OUTPUTS : 
 
        closest_value :: Closest number to value in array
        index :: Index of closest_value within array

    CALLING SEQUENCE:

        closest_value,index = find_nearest(array,value)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

###############################################################################################

def gauss_lobatto(n, n_digits):

    """
    Computes the Gauss-Lobatto quadrature [1]_ points and weights.

    The Gauss-Lobatto quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P'_(n-1)`
    and the weights `w_i` are given by:

    .. math::
        &w_i = \frac{2}{n(n-1) \left[P_{n-1}(x_i)\right]^2},\quad x\neq\pm 1\\
        &w_i = \frac{2}{n(n-1)},\quad x=\pm 1

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_lobatto
    >>> x, w = gauss_lobatto(3, 5)
    >>> x
    [-1, 0, 1]
    >>> w
    [0.33333, 1.3333, 0.33333]
    >>> x, w = gauss_lobatto(4, 5)
    >>> x
    [-1, -0.44721, 0.44721, 1]
    >>> w
    [0.16667, 0.83333, 0.83333, 0.16667]

    See Also
    ========

    gauss_legendre,gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
    .. [2] http://people.math.sfu.ca/~cbm/aands/page_888.htm
    """
    from sympy.core import S, Dummy, pi
    from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                        hermite_poly, jacobi_poly)
    from sympy.polys.rootoftools import RootOf

    x = Dummy("x")
    p = legendre_poly(n-1, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in pd.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S(1)/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))

    xi.insert(0, -1)
    xi.append(1)
    w.insert(0, (S(2)/(n*(n-1))).n(n_digits))
    w.append((S(2)/(n*(n-1))).n(n_digits))
    return xi, w

###############################################################################################

def ngauss(npx,x,ng,iamp,imean,ifwhm,MakePlot=False):


    """
        FUNCTION NAME : ngauss()
        
        DESCRIPTION : 

            Create a function which is the sum of multiple gaussians
 
        INPUTS :
      
            npx :: Number of points in x-array
            x(npx) :: Array specifying the points at which the function must be calculated
            ng :: Number of gaussians
            iamp(ng) :: Amplitude of each of the gaussians
            imean(ng) :: Center x-point of the gaussians
            ifwhm(ng) :: FWHM of the gaussians

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            fun(npx) :: Function at each x-point

        CALLING SEQUENCE:
        
            fun = ngauss(npx,x,ng,iamp,imean,ifwhm)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    fun  = np.zeros([npx])
    isigma = ifwhm/(2.0*np.sqrt(2.*np.log(2.)))
    for i in range(npx):
        for j in range(ng):
            fun[i] = fun[i] + iamp[j] * np.exp( -(x[i]-imean[j])**2.0/(2.0*isigma[j]**2.)  )


    #Make plot if keyword is specified
    if MakePlot == True:
        axis_font = {'size':'20'}
        cm = plt.cm.get_cmap('RdYlBu')
        fig = plt.figure(figsize=(15,8))
        wavemin = x.min()
        wavemax = x.max()
        ax = plt.axes()
        ax.set_xlim(wavemin,wavemax)
        ax.tick_params(labelsize=20)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel('x',**axis_font)
        plt.ylabel('f(x)',**axis_font)
        im = ax.plot(x,fun)
        plt.grid()
        plt.show()    
    
    return fun

###############################################################################################

def lognormal_dist(x,mu,sigma,MakePlot=False):


    """
        FUNCTION NAME : lognormal()
        
        DESCRIPTION : 

            Calculate the value of the log-normal distribution at the points x
 
        INPUTS :
      
            x :: Points at which the function must be evaluated
            mu :: Mean of the distribution
            sigma :: Variance of the distribution

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            fun :: Function at each x-point

        CALLING SEQUENCE:
        
            fun = lognormal(x,mu,sigma)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2021)

    """

    fun = 1.0 / (x * sigma * np.sqrt(2.*np.pi)) * np.exp( - (np.log(x)-np.log(mu))**2. / (2.*sigma**2.) )
    
    return fun


    ###############################################################################################

def gamma_dist(x,a,b,MakePlot=False):


    """
        FUNCTION NAME : gamma_dist()
        
        DESCRIPTION : 

            Calculate the value of the modified gamma distribution presented in Hansen and Travis (1974)
            Equation 2.56: n(r) = r**( (1-3*b)/b ) * np.exp(-r/(a*b))
 
        INPUTS :
      
            x :: Points at which the function must be evaluated
            a :: Effective radius of the distribution
            b :: Effective variance of the distribution

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            fun :: Function at each x-point

        CALLING SEQUENCE:
        
            fun = gamma_dist(x,a,b)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2021)

    """

    fun = x**( (1-3*b)/b ) * np.exp(-x/(a*b))

    return fun