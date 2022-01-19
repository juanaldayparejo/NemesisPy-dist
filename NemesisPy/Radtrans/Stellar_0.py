from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Stellar_0:

    def __init__(self, SOLEXIST=True, DIST=1.0, RADIUS=1.0, ISPACE=0, NCONV=2):

        """
        Inputs
        ------
        @param SOLEXIST: log,
            Flag indicating whether sunlight needs to be included (SOLEXIST=True) or not (SOLEXIST=False)
        @param DIST: float,
            Distance between star and planet (AU) 
        @param RADIUS: float,
            Radius of the star (km)       
        @param ISPACE: int,
            Spectral units in which the solar spectrum is defined (0) Wavenumber (1) Wavelength              
        @param NCONV: int,
            Number of spectral points in which the stellar spectrum is defined
        
        Attributes
        ----------
        @attribute VCONV: 1D array
            Wavelength array at which the stellar file is defined
        @attribute SOLSPEC: 1D array
            Stellar power spectrum (W/(cm-1) or W/um)
        @attribute SOLFLUX: 1D array
            Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)
        @attribute STELLARDATA: str
            String indicating where the STELLAR data files are stored

        Methods
        -------
        Stellar_0.edit_SOLSPEC
        """

        from NemesisPy import Nemesis_Path

        #Input parameters
        self.SOLEXIST = SOLEXIST
        self.DIST = DIST
        self.RADIUS = RADIUS
        self.ISPACE = ISPACE
        self.NCONV = NCONV

        # Input the following profiles using the edit_ methods.
        self.VCONV = None # np.zeros(NCONV)
        self.SOLSPEC = None # np.zeros(NCONV)
        self.SOLFLUX = None #np.zeros(NCONV)

        self.STELLARDATA = Nemesis_Path()+'NemesisPy/Data/stellar/'

    def edit_VCONV(self, VCONV):
        """
        Edit the solar spectrum 
        @param VCONV: 1D array
            Array defining the wavelengths at which the solar spectrum is defined
        """
        VCONV_array = np.array(VCONV)
        assert len(VCONV_array) == self.NCONV, 'VCONV should have NCONV elements'
        self.VCONV = VCONV_array

    def edit_SOLSPEC(self, SOLSPEC):
        """
        Edit the solar spectrum 
        @param SOLSPEC: 1D array
            Array defining the solar spectrum
        """
        SOLSPEC_array = np.array(SOLSPEC)
        assert len(SOLSPEC_array) == self.NCONV, 'SOLSPEC should have NCONV elements'
        self.SOLSPEC = SOLSPEC_array

    def read_sol(self, runname, MakePlot=False):
        """
        Read the solar spectrum from the .sol file
        @param runname: str
            Name of the NEMESIS run
        """

        from NemesisPy import file_lines
        
        #Opening file
        f = open(runname+'.sol','r')
        s = f.readline().split()
        solname = s[0]
        f.close()

        nlines = file_lines(self.STELLARDATA+solname)

        #Reading buffer
        ibuff = 0
        with open(self.STELLARDATA+solname,'r') as fsol:
            for curline in fsol:
                if curline.startswith("#"):
                    ibuff = ibuff + 1
                else:
                    break

        nvsol = nlines - ibuff - 2
        
        #Reading file
        fsol = open(self.STELLARDATA+solname,'r')
        for i in range(ibuff):
            s = fsol.readline().split()
    
        s = fsol.readline().split()
        ispace = int(s[0])
        s = fsol.readline().split()
        solrad = float(s[0])
        vsol = np.zeros(nvsol)
        rad = np.zeros(nvsol)
        for i in range(nvsol):
            s = fsol.readline().split()
            vsol[i] = float(s[0])
            rad[i] = float(s[1])
    
        fsol.close()

        self.RADIUS = solrad
        self.ISPACE = ispace
        self.NCONV = nvsol
        self.edit_VCONV(vsol)
        self.edit_SOLSPEC(rad)

        if MakePlot==True:
            fig,ax1=plt.subplots(1,1,figsize=(8,3))
            ax1.plot(vsol,rad)
            #ax1.set_yscale('log')
            plt.tight_layout()
            plt.show()
    
        return solrad,nvsol,vsol,rad


    def calc_solar_flux(self):
        """
        Calculate the stellar flux at the planet's distance
        """

        area = 4.*np.pi*(self.DIST * unit["AU"] * 100. )**2.
        self.SOLFLUX = self.SOLSPEC / area   #W cm-2 (cm-1)-1 or W cm-2 um-1