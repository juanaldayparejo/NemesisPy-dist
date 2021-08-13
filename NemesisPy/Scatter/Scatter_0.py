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

Scattering Class. Includes the absorption and scattering properties of aerosol particles.
"""

class Scatter_0:

    def __init__(self, ISPACE=0, NMU=5, NF=2, NPHI=101, NDUST=1,SOL_ANG=0.0,EMISS_ANG=0.0,AZI_ANG=0.0):

        """
        Inputs
        ------
        @param ISPACE: int,
            Flag indicating the spectral units
            (0) Wavenumber (cm-1)
            (1) Wavelength (um)
        @param NDUST: int,
            Number of aerosol populations included in the atmosphere
        @param NMU: int,
            Number of zenith ordinates to perform the scattering calculations                
        @param NF: int,
            Number of Fourier components to perform the scattering calculations in the azimuth direction
        @param NPHI: int,
            Number of azimuth ordinates to perform the scattering calculations using Fourier analysis
        @param SOL_ANG: float,
            Observation solar angle (degrees)
        @param EMISS_ANG: float,
            Observation emission angle (degrees)
        @param AZI_ANG: float,
            Observation azimuth angle (degrees)
            
        Attributes
        ----------
        @attribute NWAVE: int,
            Number of wavelengths used to define its spectral properties 
        @attribute NTHETA: int,
            Number of angles used to define the scattering phase function of the aerosols 
        @attribute WAVE: 1D array,
            Wavelengths at which the spectral properties of the aerosols are defined      
        @attribute KEXT: 2D array,
            Extinction cross section of each of the aerosol populations at each wavelength (cm2)
        @attribute SGLALB: 2D array,
            Single scattering albedo of each of the aerosol populations at each wavelength
        @attribute KABS: 2D array,
            Absorption cross section of each of the aerosol populations at each wavelength (cm2)
        @attribute KSCA: 2D array,
            Scattering cross section of each of the aerosol populations at each wavelength (cm2)
        @attribute PHASE: 3D array,
            Scattering phase function of each of the aerosol populations at each wavelength
        @attribute F: 2D array,
            Parameter defining the relative contribution of G1 and G2 of the double Henyey-Greenstein phase function
            See Irvine (1965)
        @attribute G1: 2D array,
            Parameter defining the first assymetry factor of the double Henyey-Greenstein phase function
            See Irvine (1965)
        @attribute G2: 2D array,
            Parameter defining the second assymetry factor of the double Henyey-Greenstein phase function
            See Irvine (1965)
        @attribute MU: 1D array,
            Cosine of the zenith angles corresponding to the Gauss-Lobatto quadrature points
        @attribute WTMU: 1D array,
            Quadrature weights of the Gauss-Lobatto quadrature points

        Methods
        -------
        Scatter_0.calc_GAUSS_LOBATTO()
        Scatter_0.fit_hg()
        """

        #Input parameters
        self.NMU = NMU
        self.NF = NF
        self.NPHI = NPHI

        # Input the following profiles using the edit_ methods.
        self.NWAVE = None
        self.NTHETA = None
        self.WAVE = None #np.zeros(NWAVE)
        self.KEXT = None #np.zeros(NWAVE,NDUST)
        self.KABS = None #np.zeros(NWAVE,NDUST)
        self.KSCA = None #np.zeros(NWAVE,NDUST)
        self.SGLALB = None #np.zeros(NWAVE,NDUST)
        self.THETA = None #np.zeros(NTHETA)
        self.PHASE = None #np.zeros(NWAVE,NTHETA,NDUST)

        self.MU = None # np.zeros(NCONV)
        self.WTMU = None # np.zeros(NCONV)

        #Henyey-Greenstein phase function parameters
        self.G1 = None  #np.zeros(NWAVE,NDUST)
        self.G2 = None #np.zeros(NWAVE,NDUST)
        self.F = None #np.zeros(NWAVE,NDUST)

        self.calc_GAUSS_LOBATTO()

    def calc_GAUSS_LOBATTO(self):
        """
        Calculate the Gauss-Lobatto quadrature points and weights.
        """

        from NemesisPy import gauss_lobatto

        nzen = 2*self.NMU    #The gauss_lobatto function calculates both positive and negative angles, and Nemesis just uses the posiive
        ndigits = 12
        x,w = gauss_lobatto(nzen,ndigits)
        self.MU = x[self.NMU:nzen]
        self.WTMU = w[self.NMU:nzen]

    def read_xsc(self,runname,MakePlot=False):
        """
        Read the aerosol properties from the .xsc file
        """

        from NemesisPy import file_lines

        #reading number of lines in file
        nlines = file_lines(runname+'.xsc')
        nwave = int((nlines-1)/ 2)

        #Reading file
        f = open(runname+'.xsc','r')
    
        s = f.readline().split()
        naero = int(s[0])
    
        wave = np.zeros([nwave])
        ext_coeff = np.zeros([nwave,naero])
        sglalb = np.zeros([nwave,naero])
        for i in range(nwave):
            s = f.readline().split()
            wave[i] = float(s[0])
            for j in range(naero):
                ext_coeff[i,j] = float(s[j+1])
            s = f.readline().split()
            for j in range(naero):
                sglalb[i,j] = float(s[j])

        f.close()

        self.NDUST = naero
        self.NWAVE = nwave
        self.WAVE = wave
        self.KEXT = ext_coeff
        self.SGLALB = sglalb
        self.KSCA = self.SGLALB * self.KEXT
        self.KABS = self.KEXT - self.KSCA

        if MakePlot==True:

            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6))

            for i in range(self.NDUST):

                ax1.plot(self.WAVE,self.KEXT[:,i],label='Dust population '+str(i+1))
                ax2.plot(self.WAVE,self.SGLALB[:,i])

            ax1.legend()
            ax1.grid()
            ax2.grid()
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax1.set_ylabel('k$_{ext}$ (cm$^2$)')
            ax2.set_ylabel('Single scattering albedo')

            plt.tight_layout()
            plt.show()

    def read_hgphase(self,MakePlot=False):
        """
        Read the Henyey-Greenstein phase function parameters stored in the hgphaseN.dat files
        """

        from NemesisPy import file_lines
        
        if self.NWAVE==None:
            uwave = 1
            self.NWAVE = file_lines('hgphase1.dat')
        else:
            uwave = 0
            nwave = file_lines('hgphase1.dat')
            if nwave!=self.NWAVE:
                sys.exit('error reading hgphase1.dat :: NWAVE needs to be the same in .xsc and hgphase files')

        wave = np.zeros(self.NWAVE) 
        g1 = np.zeros([self.NWAVE,self.NDUST])
        g2 = np.zeros([self.NWAVE,self.NDUST])
        fr = np.zeros([self.NWAVE,self.NDUST])
        for i in range(self.NDUST):
            f = open('hgphase'+str(i+1)+'.dat','r')
            for j in range(self.NWAVE):
                s = f.readline().split()
                wave[j] = float(s[0])
                fr[j,i] = float(s[1])
                g1[j,i] = float(s[2])
                g2[j,i] = float(s[3])
            f.close()

        if uwave==1:
            self.WAVE = wave

        self.G1 = g1
        self.G2 = g2
        self.F = fr

    def calc_hgphase(self,Theta=None,MakePlot=False):
        """
        Calculate the phase function at Theta angles given the double Henyey-Greenstein parameters
        """

        if Theta[0]!=None:
            self.NTHETA = len(Theta)
            self.THETA = Theta

        phase = np.zeros([self.NWAVE,self.NTHETA,self.NDUST])
        for i in range(self.NTHETA):

            t1 = (1.-self.G1**2.)/(1. - 2.*self.G1*np.cos(self.THETA[i]/180.*np.pi) + self.G1**2.)**1.5
            t2 = (1.-self.G2**2.)/(1. - 2.*self.G2*np.cos(self.THETA[i]/180.*np.pi) + self.G2**2.)**1.5
            phase[:,i,:] = self.F * t1 + (1.0 - self.F) * t2

        self.PHASE = phase

    def calc_tau_dust(self,WAVEC,Layer,MakePlot=False):
        """
        Calculate the CIA opacity in each atmospheric layer

        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations
        """

        from scipy import interpolate
        from NemesisPy import find_nearest

        if((WAVEC.min()<self.WAVE.min()) & (WAVEC.max()>self.WAVE.min())):
            sys.exit('error in Scatter_0() :: Spectral range for calculation is outside of range in which the Aerosol properties are defined')

        #Calculating the opacity at each vertical layer for each dust population
        NWAVEC = len(WAVEC)
        TAUDUST = np.zeros([NWAVEC,Layer.NLAY,self.NDUST])
        TAUCLSCAT = np.zeros([NWAVEC,Layer.NLAY,self.NDUST])
        for i in range(self.NDUST):

            #Interpolating the cross sections to the correct grid
            f = interpolate.interp1d(self.WAVE,np.log(self.KEXT[:,i]))
            kext = np.exp(f(WAVEC))
            f = interpolate.interp1d(self.WAVE,np.log(self.KSCA[:,i]))
            ksca = np.exp(f(WAVEC))

            #Calculating the opacity at each layer
            for j in range(Layer.NLAY):
                DUSTCOLDENS = Layer.CONT[j,i] * 1.0e-4   #particles/cm2
                TAUDUST[:,j,i] =  kext * DUSTCOLDENS
                TAUCLSCAT[:,j,i] = ksca * DUSTCOLDENS

        return TAUDUST,TAUCLSCAT