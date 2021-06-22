from NemesisPy.Profile import *
from NemesisPy.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Measurement:

    def __init__(self, NGEOM=1, FWHM=0.0, LATITUDE=0.0, LONGITUDE=0.0):

        """
        Inputs
        ------
        @param NGEOM: int,
            Number of observing geometries
        @param FWHM: int,
            Full-width at half-maximum of the instrument     
        @param LATITUDE: int,
            Planetocentric latitude at centre of the field of view  
        @param LONGITUDE: int,
            Planetocentric longitude at centre of the field of view          
        
        Attributes
        ----------
        @attribute NCONV: 1D array
            Number of convolution spectral points in each spectrum
        @attribute VCONV: 2D array
            Convolution spectral points (wavelengths/wavenumbers) in each spectrum
        @attribute MEAS: 2D array
            Measured spectrum for each geometry
        @attribute ERRMEAS: 2D array
            Noise in the measured spectrum for each geometry        
        @attribute NAV: 1D array
            For each geometry, number of individual geometries need to be calculated
            and averaged to reconstruct the field of view 
        @attribute FLAT: 2D array
            Latitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute FLON: 2D array
            Longitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute SOL_ANG: 2D array
            Solar indicent angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute EMISS_ANG: 2D array
            Emission angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute AZI_ANG: 2D array
            Azimuth angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute WGEOM: 2D array
            Weights of each point for the averaging of the FOV (when NAV > 1)

        Methods
        -------
        Measurement.edit_NCONV
        Measurement.edit_VCONV
        Measurement.edit_MEAS
        Measurement.edit_ERRMEAS
        Measurement.edit_NAV
        Measurement.edit_FLAT
        Measurement.edit_FLON
        Measurement.edit_SOLANG
        Measurement.edit_EMISSANG
        Measurement.edit_AZIANG
        Measurement.edit_WGEOM     
        """

        #Input parameters
        self.NGEOM = NGEOM
        self.FWHM = FWHM
        self.LATITUDE = LATITUDE        
        self.LONGITUDE = LONGITUDE

        # Input the following profiles using the edit_ methods.
        self.NCONV = None # np.zeros(NGEOM)
        self.VCONV = None # np.zeros(NCONV,NGEOM)
        self.MEAS =  None # np.zeros(NCONV,NGEOM)
        self.ERRMEAS = None # np.zeros(NCONV,NGEOM)
        self.NAV = None # np.zeros(NGEOM)     
        self.FLAT = None # np.zeros(NGEOM,NAV)
        self.FLON = None # np.zeros(NGEOM,NAV)
        self.SOL_ANG = None # np.zeros(NGEOM,NAV)
        self.EMISS_ANG = None # np.zeros(NGEOM,NAV)
        self.AZI_ANG = None # np.zeros(NGEOM,NAV)
        self.WGEOM = None # np.zeros(NGEOM,NAV)

    def edit_NCONV(self, NCONV_array):
        """
        Edit the number of convolution wavelengths/wavenumbers in each geometry
        @param NCONV_array: 1D array
            Number of convolution wavelengths/wavenumbers in each geometry
        """
        NCONV_array = np.array(NCONV_array,dtype='int32')
        assert len(NCONV_array) == self.NGEOM, 'NCONV should have NGEOM elements'
        self.NCONV = NCONV_array

    def edit_VCONV(self, VCONV_array):
        """
        Edit the convolution wavelengths/wavenumbers array in each geometry
        @param VCONV_array: 1D array
            Convolution wavelengths/wavenumbers in each geometry
        """
        VCONV_array = np.array(VCONV_array)
        try:
            assert VCONV_array.shape == (self.NCONV.max(), self.NGEOM),\
                'VCONV should be NCONV by NGEOM.'
        except:
            assert VCONV_array.shape == (self.NCONV.max(),) and self.NGEOM==1,\
                'VCONV should be NP by NVMR.'

        self.VCONV = VCONV_array