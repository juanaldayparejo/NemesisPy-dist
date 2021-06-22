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
