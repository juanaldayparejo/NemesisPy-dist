from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import miepython

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

Scattering Class. Includes the absorption and scattering properties of aerosol particles.
"""

class Scatter_0:

    def __init__(self, ISPACE=0, ISCAT=0, IRAY=0, IMIE=0, NMU=5, NF=2, NPHI=101, NDUST=1, SOL_ANG=0.0, EMISS_ANG=0.0, AZI_ANG=0.0, NTHETA=91, THETA=np.linspace(0.,180.,91)):

        """
        Inputs
        ------
        @param ISPACE: int,
            Flag indicating the spectral units
            (0) Wavenumber (cm-1)
            (1) Wavelength (um)
        @param ISCAT: int,
            Flag indicating the type of scattering calculation that must be performed 
            (0) Thermal emission calculation (no scattering)
            (1) Multiple scattering
            (2) Internal scattered radiation field is calculated first (required for limb-scattering calculations)
            (3) Single scattering in plane-parallel atmosphere
            (4) Single scattering in spherical atmosphere
        @param IRAY int,
            Flag indicating the type of Rayleigh scattering to include in the calculations:
            (0) Rayleigh scattering optical depth not included
            (1) Rayleigh optical depths for gas giant atmosphere
            (2) Rayleigh optical depth suitable for CO2-dominated atmosphere
            (>2) Rayleigh optical depth suitable for a N2-O2 atmosphere
        @param IMIE int,
            Flag indicating how the aerosol phase function needs to be computed (only relevant for ISCAT>0):
            (0) Phase function is computed from the associated Henyey-Greenstein parameters stored in G1,G2
            (1) Phase function is computed from the Mie-Theory parameters stored in PHASE
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
        @attribute WLPOL: 2D array
            Weights of the Legendre polynomials used to model the phase function
        @attribute MU: 1D array,
            Cosine of the zenith angles corresponding to the Gauss-Lobatto quadrature points
        @attribute WTMU: 1D array,
            Quadrature weights of the Gauss-Lobatto quadrature points
        @attribute ALPHA: real,
            Scattering angle (degrees) computed from the observing angles

        Methods
        -------
        Scatter_0.calc_GAUSS_LOBATTO()
        Scatter_0.read_xsc()
        Scatter_0.write_xsc()
        Scatter_0.read_hgphase()
        Scatter_0.calc_hgphase()
        Scatter_0.calc_phase()
        Scatter_0.calc_tau_dust()
        Scatter_0.calc_tau_rayleighj()
        Scatter_0.calc_tau_rayleighv()
        Scatter_0.read_refind_file()
        Scatter_0.read_refind()
        Scatter_0.miescat()
        Scatter_0.initialise_arrays()
        """

        #Input parameters
        self.NMU = NMU
        self.NF = NF
        self.NPHI = NPHI
        self.ISPACE = ISPACE
        self.ISCAT = ISCAT
        self.SOL_ANG = SOL_ANG
        self.EMISS_ANG = EMISS_ANG
        self.AZI_ANG = AZI_ANG
        self.NDUST = NDUST
        self.IRAY = IRAY
        self.IMIE = IMIE

        # Input the following profiles using the edit_ methods.
        self.NWAVE = None
        self.NTHETA = NTHETA
        self.WAVE = None #np.zeros(NWAVE)
        self.KEXT = None #np.zeros(NWAVE,NDUST)
        self.KABS = None #np.zeros(NWAVE,NDUST)
        self.KSCA = None #np.zeros(NWAVE,NDUST)
        self.SGLALB = None #np.zeros(NWAVE,NDUST)
        self.THETA = THETA
        self.PHASE = None #np.zeros(NWAVE,NTHETA,NDUST)

        self.MU = None # np.zeros(NCONV)
        self.WTMU = None # np.zeros(NCONV)

        #Henyey-Greenstein phase function parameters
        self.G1 = None  #np.zeros(NWAVE,NDUST)
        self.G2 = None #np.zeros(NWAVE,NDUST)
        self.F = None #np.zeros(NWAVE,NDUST)

        #Legendre polynomials phase function parameters
        self.NLPOL = None #int
        self.WLPOL = None #np.zeros(NWAVE,NLPOL,NDUST)

        #Refractive index of a given aerosol population
        self.NWAVER = None 
        self.WAVER = None #np.zeros(NWAVER)
        self.REFIND_REAL = None #np.zeros(NWAVER)
        self.REFIND_IM = None #np.zeros(NWAVER)

        self.calc_GAUSS_LOBATTO()

    def initialise_arrays(self,NDUST,NWAVE,NTHETA):
        """
        Initialise arrays for storing the scattering properties of the aerosols
        """

        self.NDUST = NDUST
        self.NWAVE = NWAVE
        self.NTHETA = NTHETA
        self.WAVE = np.zeros(self.NWAVE)
        self.KEXT = np.zeros((self.NWAVE,self.NDUST))
        self.KSCA = np.zeros((self.NWAVE,self.NDUST))
        self.KABS = np.zeros((self.NWAVE,self.NDUST))
        self.SGLALB = np.zeros((self.NWAVE,self.NDUST))
        self.PHASE = np.zeros((self.NWAVE,self.NTHETA,self.NDUST))
        self.G1 = np.zeros((self.NWAVE,self.NDUST))
        self.G2 = np.zeros((self.NWAVE,self.NDUST))
        self.F = np.zeros((self.NWAVE,self.NDUST))

    def calc_GAUSS_LOBATTO(self):
        """
        Calculate the Gauss-Lobatto quadrature points and weights.
        """

        from NemesisPy import gauss_lobatto

        nzen = 2*self.NMU    #The gauss_lobatto function calculates both positive and negative angles, and Nemesis just uses the posiive
        ndigits = 12
        x,w = gauss_lobatto(nzen,ndigits)
        self.MU = np.array(x[self.NMU:nzen],dtype='float64')
        self.WTMU = np.array(w[self.NMU:nzen],dtype='float64')

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
        self.PHASE = np.zeros((self.NWAVE,self.NTHETA,self.NDUST))

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

    def write_xsc(self,runname,MakePlot=False):
        """
        Write the aerosol scattering and absorving properties to the .xsc file
        """

        f = open(runname+'.xsc','w')
        f.write('%i \n' % (self.NDUST))

        for i in range(self.NWAVE):
            str1 = str('{0:7.6f}'.format(self.WAVE[i]))
            str2 = ''
            for j in range(self.NDUST):
                str1 = str1+'\t'+str('{0:7.6e}'.format(self.KEXT[i,j]))
                str2 = str2+'\t'+str('{0:7.6f}'.format(self.SGLALB[i,j]))

            f.write(str1+'\n')
            f.write(str2+'\n')

        f.close()
        

    def read_hgphase(self,NDUST=None):
        """
        Read the Henyey-Greenstein phase function parameters stored in the hgphaseN.dat files
        """

        from NemesisPy import file_lines
       
        if NDUST!=None:
            self.NDUST = NDUST

        #Getting the number of wave points
        nwave = file_lines('hgphase1.dat')
        self.NWAVE = nwave

        wave = np.zeros(nwave)
        g1 = np.zeros((self.NWAVE,self.NDUST))
        g2 = np.zeros((self.NWAVE,self.NDUST))
        fr = np.zeros((self.NWAVE,self.NDUST))

        for IDUST in range(self.NDUST):

            filename = 'hgphase'+str(IDUST+1)+'.dat'

            f = open(filename,'r')
            for j in range(self.NWAVE):
                s = f.readline().split()
                wave[j] = float(s[0])
                fr[j,IDUST] = float(s[1])
                g1[j,IDUST] = float(s[2])
                g2[j,IDUST] = float(s[3])
            f.close()

        self.WAVE = wave
        self.G1 = np.array(g1,dtype='float64')
        self.G2 =  np.array(g2,dtype='float64')
        self.F =  np.array(fr,dtype='float64')

    def write_hgphase(self):
        """
        Write the Henyey-Greenstein phase function parameters into the hgphaseN.dat files
        """

        for IDUST in range(self.NDUST):

            filename = 'hgphase'+str(IDUST+1)+'.dat'

            f = open(filename,'w')
            for j in range(self.NWAVE):

                f.write('%10.7f \t %10.7f \t %10.7f \t %10.7f \n' % (self.WAVE[j],self.F[j,IDUST],self.G1[j,IDUST],self.G2[j,IDUST]))

            f.close()


    def calc_hgphase(self,Theta):
        """
        Calculate the phase function at Theta angles given the double Henyey-Greenstein parameters
        @param Theta: 1D array or real scalar
            Scattering angle (degrees)
        """

        if np.isscalar(Theta)==True:
            ntheta = 1
            Thetax = [Theta]
        else:
            Thetax = Theta

        #Re-arranging the size of Thetax to be (NTHETA,NWAVE,NDUST)
        Thetax = np.repeat(Thetax[:,np.newaxis],self.NWAVE,axis=1)
        Thetax = np.repeat(Thetax[:,:,np.newaxis],self.NDUST,axis=2)

        t1 = (1.-self.G1**2.)/(1. - 2.*self.G1*np.cos(Thetax/180.*np.pi) + self.G1**2.)**1.5
        t2 = (1.-self.G2**2.)/(1. - 2.*self.G2*np.cos(Thetax/180.*np.pi) + self.G2**2.)**1.5
        
        phase = self.F * t1 + (1.0 - self.F) * t2
        phase = np.transpose(phase,axes=[1,0,2])

        return phase

    def interp_phase(self,Theta):
        """
        Interpolate the phase function at Theta angles fiven the phase function in the Scatter class

        Input
        ______

        @param Theta: 1D array
            Scattering angle (degrees)


        Output
        _______

        @param phase(NWAVE,NTHETA,NDUST) : 3D array
            Phase function interpolated at the correct Theta angles

        """

        from scipy.interpolate import interp1d

        s = interp1d(self.THETA,self.PHASE,axis=1)
        phase = s(Theta)

        return phase

    def calc_phase(self,Theta,Wave):
        """
        Calculate the phase function of each aerosol type at a given  scattering angle Theta and a given set of Wavelengths/Wavenumbers
        If IMIE=0 in the Scatter class, then the phase function is calculated using the Henyey-Greenstein parameters.
        If IMIE=1 in the Scatter class, then the phase function is interpolated from the values stored in the PHASE array
        If IMIE=2 in the Scatter class, then the phase function is calculated using Legendre Polynomials

        Input
        ______

        @param Theta: real or 1D array
            Scattering angle (degrees)
        @param Wave: 1D array
            Wavelengths (um) or wavenumbers (cm-1) ; It must be the same units as given by the ISPACE

        Outputs
        ________

        @param phase(NWAVE,NTHETA,NDUST) : 3D array
            Phase function at each wavelength, angle and for each aerosol type

        """

        from scipy.interpolate import interp1d

        nwave = len(Wave)

        if np.isscalar(Theta)==True:
            Thetax = [Theta]
        else:
            Thetax = Theta

        ntheta = len(Thetax)

        phase2 = np.zeros((nwave,ntheta,self.NDUST))

        if self.IMIE==0:
            
            #Calculating the phase function at the wavelengths defined in the Scatter class
            phase1 = self.calc_hgphase(Thetax)

        elif self.IMIE==1:

            #Interpolate the phase function to the Scattering angle at the wavelengths defined in the Scatter class
            phase1 = self.interp_phase(Thetax)

        elif self.IMIE==2:

            #Calculating the phase function at the wavelengths defined in the Scatter class
            #using the Legendre polynomials
            phase1 = self.lpphase(Thetax)

        else:
            sys.exit('error :: IMIE value not valid in Scatter class')


        #Interpolating the phase function to the wavelengths defined in Wave
        s = interp1d(self.WAVE,phase1,axis=0)
        phase = s(Wave)

        return phase


    def calc_phase_ray(self,Theta):
        """
        Calculate the phase function of Rayleigh scattering at a given scattering angle (Dipole scattering)

        Input
        ______

        @param Theta: real or 1D array
            Scattering angle (degrees)

        Outputs
        ________

        @param phase(NTHETA) : 1D array
            Phase function at each angle

        """

        phase = 0.75 * ( 1.0 + np.cos(Theta/180.*np.pi) * np.cos(Theta/180.*np.pi) )

        return phase


    def read_phase(self,NDUST=None):
        """
        Read a file with the format of the PHASE*.DAT using the format required by NEMESIS

        Optional inputs
        ----------------

        @NDUST: int
            If included, then several files from 1 to NDUST will be read with the name format being PHASE*.DAT
        """

        if NDUST!=None:
            self.NDUST = NDUST

        mwave = 5000
        mtheta = 361
        kext = np.zeros((mwave,self.NDUST))
        sglalb = np.zeros((mwave,self.NDUST))
        phase = np.zeros((mwave,mtheta,self.NDUST))

        for IDUST in range(self.NDUST):

            filename = 'PHASE'+str(IDUST+1)+'.DAT'         

            f = open(filename,'r')
            s = f.read()[0:1000].split()
            f.close()
            #Getting the spectral unit
            if s[0]=='wavenumber':
                self.ISPACE = 0
            elif s[1]=='wavelength':
                self.ISPACE = 1

            #Calculating the wave array
            vmin = float(s[1])
            vmax = float(s[2])
            delv = float(s[3])
            nwave = int(s[4])
            nphase = int(s[5])
            wave = np.linspace(vmin,vmax,nwave)

            #Reading the rest of the information
            f = open(filename,'r')
            s = f.read()[1000:].split()
            f.close()
            i0 = 0
            #Reading the phase angle
            theta = np.zeros(nphase)
            for i in range(nphase):
                theta[i] = s[i0]
                i0 = i0 + 1

            #Reading the data
            wave1 = np.zeros(nwave)
            kext1 = np.zeros(nwave)
            sglalb1 = np.zeros(nwave)
            phase1 = np.zeros((nwave,nphase))
            for i in range(nwave):

                wave1[i]=s[i0]
                i0 = i0 + 1
                kext1[i] = float(s[i0])
                i0 = i0 + 1
                sglalb1[i] = float(s[i0])
                i0 = i0 + 1

                for j in range(nphase):
                    phase1[i,j] = float(s[i0])
                    i0 = i0 + 1

            kext[0:nwave,IDUST] = kext1[:]
            sglalb[0:nwave,IDUST] = sglalb1[:]
            phase[0:nwave,0:nphase,IDUST] = phase1[:,:]
            
        #Filling the parameters in the class based on the information in the files
        self.NWAVE = nwave
        self.NTHETA = nphase
        self.WAVE = wave1
        self.THETA = theta
        self.KEXT = np.zeros((self.NWAVE,self.NDUST))
        self.KSCA = np.zeros((self.NWAVE,self.NDUST))
        self.KABS = np.zeros((self.NWAVE,self.NDUST))
        self.SGLALB = np.zeros((self.NWAVE,self.NDUST))
        self.PHASE = np.zeros((self.NWAVE,self.NTHETA,self.NDUST)) 

        self.KEXT[:,:] = kext[0:self.NWAVE,0:self.NDUST]
        self.SGLALB[:,:] = sglalb[0:self.NWAVE,0:self.NDUST]
        self.KSCA[:,:] = self.KEXT[:,:] * self.SGLALB[:,:]
        self.KABS[:,:] = self.KEXT[:,:] - self.KSCA[:,:]
        self.PHASE[:,:,:] = phase[0:self.NWAVE,0:self.NTHETA,0:self.NDUST]


    def write_phase(self,IDUST):
        """
        Write a file with the format of the PHASE*.DAT using the format required by NEMESIS

        Inputs
        ----------------

        @IDUST: int
            Aerosol population whose properties will be written in the PHASE.DAT file
        """

        f = open('PHASE'+str(IDUST+1)+'.DAT','w')
        
        #First buffer
        if self.ISPACE==0:
            wavetype='wavenumber'
        elif self.ISPACE==1:
            wavetype='wavelength'

        str1 = "{:<512}".format(' %s  %8.2f  %8.2f  %8.4f  %4i  %4i' % (wavetype,self.WAVE.min(),self.WAVE.max(),self.WAVE[1]-self.WAVE[0],self.NWAVE,self.NTHETA))
 
        #Second buffer
        comment = 'Mie scattering  - Particle size distribution not known'
        str2 = "{:<512}".format(' %s' % (comment)  )

        #Third buffer
        strxx = ''
        for i in range(self.NTHETA):
            strx = ' %8.3f' % (self.THETA[i])
            strxx = strxx+strx

        str3 = "{:<512}".format(strxx)
        if len(str3)>512:
            sys.exit('error writing PHASEN.DAT file :: File format does not support so many scattering angles (NTHETA)')

        #Fourth buffer
        str4 = ''
        for i in range(self.NWAVE):
            strxx = ''
            strx1 = ' %8.6f %12.5e %12.5e' % (self.WAVE[i],self.KEXT[i,IDUST],self.SGLALB[i,IDUST])
            strx2 = ''
            for j in range(self.NTHETA):
                strx2 = strx2+' %10.4f' % (self.PHASE[i,j,IDUST])
            strxx = "{:<512}".format(strx1+strx2)
            if len(strxx)>512:
                sys.exit('error while writing PHASEN.DAT :: File format does not support so many scattering angles (NTHETA)')
            str4=str4+strxx

        f.write(str1+str2+str3+str4)
        f.close()


    def read_lpphase(self,NDUST=None):
        """
        Read the weights of the Legendre polynomials used to model the phase function (stored in the lpphaseN.dat files)
        These files are assumed to be pickle files with the correct format
        """

        import pickle

        if NDUST!=None:
            self.NDUST = NDUST

        #Reading the first file to read dimensions of the data
        filen = open('lpphase1.dat','rb')
        wave = pickle.load(filen)
        wlegpol = pickle.load(filen)

        self.NWAVE = len(wave)
        self.NLPOL = wlegpol.shape[1]

        wlpol = np.zeros((self.NWAVE,self.NLPOL,self.NDUST))
        for IDUST in range(self.NDUST):
            filen = open('lpphase'+str(IDUST+1)+'.dat','rb')
            wave = pickle.load(filen)
            wlegpol = pickle.load(filen)            
            wlpol[:,:,IDUST] = wlegpol[:,:]

        self.WAVE = wave
        self.WLPOL = wlpol

    def calc_lpphase(self,Theta):
        """
        Calculate the phase function at Theta angles given the weights of the Legendre polynomials
        @param Theta: 1D array or real scalar
            Scattering angle (degrees)
        """

        from scipy.special import legendre

        if np.isscalar(Theta)==True:
            ntheta = 1
            Thetax = [Theta]
        else:
            Thetax = Theta

        ntheta = len(Thetax)
        phase = np.zeros([self.NWAVE,ntheta,self.NDUST])

        for IDUST in range(self.NDUST):
            for IL in range(self.NLPOL):
                leg = legendre(IL)
                P_n = leg(np.cos(Thetax/180.*np.pi))
                for IWAVE in range(self.NWAVE):
                    phase[IWAVE,:,IDUST] = phase[IWAVE,:,IDUST] + P_n[:] * self.WLPOL[IWAVE,IL,IDUST]
        
        return phase

    def calc_tau_dust(self,WAVEC,Layer,MakePlot=False):
        """
        Calculate the aerosol opacity in each atmospheric layer

        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAUDUST(NWAVE,NLAY,NDUST) :: Aerosol opacity for each aerosol type and each layer (from extinction coefficients)
        TAUCLSCAT(NWAVE,NLAY,NDUST) :: Aerosol scattering opacity for each aerosol type and each layer
        dTAUDUSTdq(NWAVE,NLAY,NDUST) :: Rate of change of the aerosol opacity with the dust abundance
        dTAUCLSCATdq(NWAVE,NLAY,NDUST) :: Rate of change of the aerosol scattering opacity with dust abundance
        """

        from scipy import interpolate
        from NemesisPy import find_nearest

        if((WAVEC.min()<self.WAVE.min()) & (WAVEC.max()>self.WAVE.min())):
            sys.exit('error in Scatter_0() :: Spectral range for calculation is outside of range in which the Aerosol properties are defined')

        #Calculating the opacity at each vertical layer for each dust population
        NWAVEC = len(WAVEC)
        TAUDUST = np.zeros([NWAVEC,Layer.NLAY,self.NDUST])
        TAUCLSCAT = np.zeros([NWAVEC,Layer.NLAY,self.NDUST])
        dTAUDUSTdq = np.zeros([NWAVEC,Layer.NLAY,self.NDUST])
        dTAUCLSCATdq = np.zeros([NWAVEC,Layer.NLAY,self.NDUST])
        for i in range(self.NDUST):

            #Interpolating the cross sections to the correct grid
            f = interpolate.interp1d(self.WAVE,np.log(self.KEXT[:,i]))
            kext = np.exp(f(WAVEC))
            f = interpolate.interp1d(self.WAVE,np.log(self.KSCA[:,i]))
            ksca = np.exp(f(WAVEC))

            #Calculating the opacity at each layer
            for j in range(Layer.NLAY):
                DUSTCOLDENS = Layer.CONT[j,i]  #particles/m2
                TAUDUST[:,j,i] =  kext * 1.0e-4 * DUSTCOLDENS 
                TAUCLSCAT[:,j,i] = ksca * 1.0e-4 * DUSTCOLDENS
                dTAUDUSTdq[:,j,i] = kext * 1.0e-4 #dtau/dAMOUNT (m2)
                dTAUCLSCATdq[:,j,i] = ksca * 1.0e-4 #dtau/dAMOUNT (m2)

        return TAUDUST,TAUCLSCAT,dTAUDUSTdq,dTAUCLSCATdq

    def calc_tau_rayleighj(self,ISPACE,WAVEC,Layer,MakePlot=False):
        """
        Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
        for Gas Giant atmospheres using data from Allen (1976) Astrophysical Quantities

        @ISPACE: int
            Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
        dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer

        """

        AH2=13.58E-5
        BH2 = 7.52E-3
        AHe= 3.48E-5
        BHe = 2.30E-3
        fH2 = 0.864
        k = 1.37971e-23
        P0=1.01325e5
        T0=273.15

        if ISPACE==0:
            LAMBDA = 1./WAVEC * 1.0e-2  #Wavelength in metres
            x = 1.0/(LAMBDA*1.0e6)
        else:
            LAMBDA = WAVEC * 1.0e-6 #Wavelength in metres
            x = 1.0/(LAMBDA*1.0e6)

        nH2 = AH2*(1.0+BH2*x*x)
        nHe = AHe*(1.0+BHe*x*x)

        #calculate the Jupiter air's refractive index at STP (Actually n-1)
        nAir = fH2*nH2 + (1-fH2)*nHe

        #H2,He Seem pretty isotropic to me?...Hence delta = 0.
        #Penndorf (1957) quotes delta=0.0221 for H2 and 0.025 for He.
        #(From Amundsen's thesis. Amundsen assumes delta=0.02 for H2-He atmospheres
        delta = 0.0
        temp = 32*(np.pi**3.)*nAir**2.
        N0 = P0/(k*T0)

        x = N0*LAMBDA*LAMBDA
        faniso = (6.0+3.0*delta)/(6.0 - 7.0*delta)

        #Calculating the scattering cross sections in m2
        k_rayleighj = temp*faniso/(3.*(x**2)) #(NWAVE)

        #Calculating the Rayleigh opacities in each layer
        tau_ray = np.zeros([len(WAVEC),Layer.NLAY])
        dtau_ray = np.zeros([len(WAVEC),Layer.NLAY])
        for ilay in range(Layer.NLAY):
            tau_ray[:,ilay] = k_rayleighj[:] * Layer.TOTAM[ilay] #(NWAVE,NLAY) 
            dtau_ray[:,ilay] = k_rayleighj[:] #dTAURAY/dTOTAM (m2)

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            for i in range(Layer.NLAY):
                ax1.plot(WAVEC,tau_ray[:,i])
            ax1.grid()
            plt.tight_layout()
            plt.show()

        return tau_ray,dtau_ray

    def calc_tau_rayleighv(self,ISPACE,WAVEC,Layer,MakePlot=False):
        """
        Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
        for CO2-domunated atmospheres using data from Allen (1976) Astrophysical Quantities

        @ISPACE: int
            Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
        dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer

        """

        if ISPACE==0:
            LAMBDA = 1./WAVEC * 1.0e-2 * 1.0e6  #Wavelength in microns
            x = 1.0/(LAMBDA*1.0e6)
        else:
            LAMBDA = WAVEC #Wavelength in microns

        C = 8.8e-28   #provided by B. Bezard

        #Calculating the scattering cross sections in m2
        k_rayleighv = C/LAMBDA**4. * 1.0e-4 #(NWAVE)
        

        #Calculating the Rayleigh opacities in each layer
        tau_ray = np.zeros((len(WAVEC),Layer.NLAY))
        dtau_ray = np.zeros((len(WAVEC),Layer.NLAY))
        for ilay in range(Layer.NLAY):
            tau_ray[:,ilay] = k_rayleighv[:] * Layer.TOTAM[ilay] #(NWAVE,NLAY) 
            dtau_ray[:,ilay] = k_rayleighv[:] #dTAURAY/dTOTAM (m2)

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            for i in range(Layer.NLAY):
                ax1.plot(WAVEC,tau_ray[:,i])
            ax1.grid()
            plt.tight_layout()
            plt.show()

        return tau_ray,dtau_ray

    def calc_tau_rayleighv2(self,ISPACE,WAVEC,Layer,MakePlot=False):
        """
        Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
        for CO2-dominated atmospheres using Ityaksov, Linnartz, Ubachs 2008, 
        Chemical Physics Letters, 462, 31-34

        @ISPACE: int
            Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
        dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer

        """

        if ISPACE==0:
            LAMBDA = 1./WAVEC * 1.0e-2 * 1.0e6  #Wavelength in microns
            x = 1.0/(LAMBDA*1.0e6)
        else:
            LAMBDA = WAVEC #Wavelength in microns

        #dens = 1.01325d6 / (288.15 * 1.3803e-16)
        dens = 2.5475605e+19

        #wave in microns -> cm
        lam = LAMBDA*1.0e-4

        #King factor (taken from Ityaksov et al.)
        f_king = 1.14 + (25.3e-12)/(lam*lam)

        nu2 = 1./lam/lam
        term1 = 5799.3 / (16.618e9-nu2) + 120.05/(7.9609e9-nu2) + 5.3334 / (5.6306e9-nu2) + 4.3244 / (4.6020e9-nu2) + 1.218e-5 / (5.84745e6 - nu2)
        
        #refractive index
        n = 1.0 + 1.1427e3*term1

        factor1 = ( (n*n-1)/(n*n+2.0) )**2.

        k_rayleighv = (24.*np.pi**3./lam**4./dens**2.) * factor1 * f_king  #cm2
        k_rayleighv = k_rayleighv * 1.0e-4

        #Calculating the Rayleigh opacities in each layer
        tau_ray = np.zeros((len(WAVEC),Layer.NLAY))
        dtau_ray = np.zeros((len(WAVEC),Layer.NLAY))
        for ilay in range(Layer.NLAY):
            tau_ray[:,ilay] = k_rayleighv[:] * Layer.TOTAM[ilay] #(NWAVE,NLAY) 
            dtau_ray[:,ilay] = k_rayleighv[:] #dTAURAY/dTOTAM (m2)

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            for i in range(Layer.NLAY):
                ax1.plot(WAVEC,tau_ray[:,i])
            ax1.grid()
            plt.tight_layout()
            plt.show()

        return tau_ray,dtau_ray



    def read_refind(self,aeroID):
        """
        Read a file of the refractive index from the NEMESIS aerosol database 

        @aeroID: str
            ID of the aerosol type

        Outputs
        ________

        @ISPACE: int
            Flag indicating whether the refractive index is expressed in Wavenumber (0) of Wavelength (1)
        @NWAVER: int
            Number of spectral point
        @WAVER: 1D array
            Wavenumber (cm-1) / Wavelength (um) array
        @REFIND_REAL: 1D array
            Real part of the refractive index
        @REFIND_IM: 1D array
            Imaginary part of the refractive index
        """

        from NemesisPy import aerosol_info

        wave_aero = aerosol_info[str(aeroID)]["wave"]
        refind_real_aero1 = aerosol_info[str(aeroID)]["refind_real"]
        refind_im_aero1 = aerosol_info[str(aeroID)]["refind_im"]

        self.NWAVER = len(wave_aero)
        self.WAVER = wave_aero
        self.REFIND_REAL = refind_real_aero1
        self.REFIND_IM = refind_im_aero1


    def read_refind_file(self,filename,MakePlot=False):
        """
        Read a file of the refractive index using the format required by NEMESIS

        @filename: str
            Name of the file where the data is stored

        Outputs
        ________

        @ISPACE: int
            Flag indicating whether the refractive index is expressed in Wavenumber (0) of Wavelength (1)
        @NWAVER: int
            Number of spectral point
        @WAVER: 1D array
            Wavenumber (cm-1) / Wavelength (um) array
        @REFIND_REAL: 1D array
            Real part of the refractive index
        @REFIND_IM: 1D array
            Imaginary part of the refractive index
        """

        from NemesisPy import file_lines

        nlines = file_lines(filename)

        #Reading buffer
        ibuff = 0
        with open(filename,'r') as fsol:
            for curline in fsol:
                if curline.startswith("#"):
                    ibuff = ibuff + 1
                else:
                    break

        #Reading buffer
        f = open(filename,'r')
        for i in range(ibuff):
            s = f.readline().split()

        #Reading file
        s = f.readline().split()
        nwave = int(s[0])
        s = f.readline().split()
        ispace = int(s[0])

        wave = np.zeros(nwave)
        refind_real = np.zeros(nwave)
        refind_im = np.zeros(nwave)
        for i in range(nwave):
            s = f.readline().split()
            wave[i] = float(s[0])
            refind_real[i] = float(s[1])
            refind_im[i] = float(s[2])

        f.close()

        return ispace,nwave,wave,refind_real,refind_im

    def miescat(self,IDUST,psdist,pardist,MakePlot=False,rdist=None,Ndist=None,WaveNorm=None):

        """
        Function to calculate the extinction coefficient, single scattering albedo and phase functions
        for different aerosol populations using Mie Theory.

        Inputs
        ________

        @param IDUST: int
            Integer indicating to which aerosol population this calculation corresponds to (from 0 to NDUST-1)
        @param psdist: int
            Flag indicating the particle size distribution
                0 :: Single particle size
                1 :: Log-normal distribution
                2 :: Standard gamma distribution
                -1 :: Other input particle size distribution (using optional inputs)
        @param pardist: int
            Particle size distribution parameters
                psdist == 0 :: pardist(0) == particle size in microns
                psdist == 1 :: pardist(0) == mu
                               pardist(1) == sigma
                psdist == 2 :: pardist(0) == a
                               pardist(1) == b               

        Optional inputs
        ________________

        rdist :: Array of particle sizes (to be used if pardist==-1)
        Ndist :: Density of particles of each particle size (to be used if pardist==-1)
        WaveNorm :: Wavelength/Wavenumber at which the cross sections will be normalised

        Outputs
        ________

        Updated KEXT,KABS,KSCA,PHASE in the Scatter_0 class

        """

        #from scipy.integrate import simpson
        from scipy.integrate import simps
        from scipy.stats import lognorm
        from scipy import interpolate
        from NemesisPy import lognormal_dist,find_nearest
        from copy import copy

        iNorm = 0
        if WaveNorm!=None:

            if self.WAVE.min()>WaveNorm:
                sys.exit('error in miescat :: WaveNorm must be within WAVE.min() and WAVE.max()')
            if self.WAVE.max()<WaveNorm:
                sys.exit('error in miescat :: WaveNorm must be within WAVE.min() and WAVE.max()')

            iNorm = 1

        #First we determine the number of particle sizes to later perform the integration
        ######################################################################################

        if psdist==0:   #Single particle size
            nr = 1
            rd = np.zeros(nr)   #Array of particle sizes
            Nd = np.zeros(nr)   #Weight (tipycally density) of each particle size for the integration
            rd[0] = pardist[0]
            Nd[0] = 1.0
        elif psdist==1: #Log-normal distribution
            mu = pardist[0]
            sigma = pardist[1]
            r0 = lognorm.ppf(0.00000001, sigma, 0.0, mu)
            r1 = lognorm.ppf(0.99999999, sigma, 0.0, mu)
            rmax = np.exp( np.log(mu) - sigma**2.)
            delr = (rmax-r0)/50.
            nr = int((r1-r0)/delr) + 1
            rd = np.linspace(r0,r1,nr) #Array of particle sizes
            Nd = lognormal_dist(rd,mu,sigma) #Density of each particle size for the integration
        elif psdist==2: #Standard gamma distribution
            A = pardist[0]
            B = pardist[1]
            rmax = (1-3*B)/B * A * B
            nmax = rmax**((1-3*B)/B) * np.exp(-rmax/(A*B))
            r = np.linspace(0.0001,A*50.,2001)
            n = r**((1-3*B)/B) * np.exp(-r/(A*B)) / nmax

            cumdist = np.zeros(len(r))
            for i in range(len(r)):
                if i==0:
                    cumdist[i] = n[i]
                else:
                    cumdist[i] = cumdist[i-1] + n[i]
            cumdist = cumdist / cumdist.max()

            r1,ir1 = find_nearest(cumdist,0.0000001)
            r2,ir2 = find_nearest(cumdist,0.9999999)
            delr = (rmax-r[ir1])/30.
            nr = int((r[ir2]-r[ir1])/delr) + 1
            rd = np.linspace(r[ir1],r[ir2],nr)
            Nd = rd**((1-3*B)/B) * np.exp(-rd/(A*B)) / nmax

        elif psdist==-1:
            if rdist[0]==None:
                sys.exit('error in miescat :: If psdist=-1 rdist and Ndist must be filled')
            if Ndist[0]==None:
                sys.exit('error in miescat :: If psdist=-1 rdist and Ndist must be filled')
            nr = len(rdist)
            rd = np.zeros(nr)   #Array of particle sizes
            Nd = np.zeros(nr)   #Weight (tipycally density) of each particle size for the integration
            rd[:] = rdist
            Nd[:] = Ndist
        else:
            sys.exit('error in miescat :: Type of distribution not yet implemented')

        #Change the units of the spectral points if they are in wavenumbers
        ######################################################################################

        if self.ISPACE==0:  #Wavenumber
            wave1 = copy(self.WAVE)
            wavel = 1.0e4/wave1   #Wavelength array
            isort = np.argsort(wavel)
            wavel = wavel[isort]
            #refind_real1 = refind_real[isort]
            #refind_im1 = refind_im[isort]
            #refind = refind_real1 - 1.0j * refind_im1
        elif self.ISPACE==1:  #Wavelength
            wavel = copy(self.WAVE)
            #refind = refind_real - 1.0j * refind_im
        else:
            sys.exit('error in miescat :: ISPACE must be either 0 or 1')


        #Interpolating the refractive index to the correct wavelength/wavenumber grid
        #####################################################################################

        refind_real = np.zeros(self.NWAVE)
        refind_im = np.zeros(self.NWAVE)

        f = interpolate.interp1d(self.WAVER,self.REFIND_REAL)
        refind_real[:] = f(wavel)

        f = interpolate.interp1d(self.WAVER,self.REFIND_IM)
        refind_im[:] = f(wavel)

        refind = refind_real - 1.0j * refind_im 

        #We calculate the scattering and absorption properties of each particle size in the distribution
        ######################################################################################################

        kext = np.zeros([self.NWAVE,nr])
        ksca = np.zeros([self.NWAVE,nr])
        phase = np.zeros([self.NWAVE,self.NTHETA,nr])
        for ir in range(nr):

            r0 = rd[ir]
            x = np.zeros(self.NWAVE)
            x[:] = 2.*np.pi*r0/(wavel)
            qext, qsca, qback, g = miepython.mie(refind,x)

            ksca[:,ir] = qsca * np.pi * (r0/1.0e4)**2.   #Cross section in cm2
            kext[:,ir] = qext * np.pi * (r0/1.0e4)**2.   #Cross section in cm2

            mu = np.cos(self.THETA/180.*np.pi)
            #In miepython the phase function is normalised to the single scattering albedo
            #For the integration over particle size distributions, we need to follow the 
            #formula 2.47 in Hansen and Travis (1974), which does not use a normalised 
            #version of the phase function. Therefore, we need to correct it.
            for iwave in range(self.NWAVE):
                unpolar = miepython.i_unpolarized(refind[iwave],x[iwave],mu)
                phase[iwave,:,ir] = unpolar
                phase[iwave,:,ir] = unpolar / (qsca[iwave]/qext[iwave]) / (wavel[iwave]*1.0e-4)**2. * np.pi * ksca[iwave,ir] 

        #Now integrating over particle size to find the mean scattering and absorption properties
        ###########################################################################################

        if nr>1:
            kext1 = np.zeros([self.NWAVE,nr])
            ksca1 = np.zeros([self.NWAVE,nr])
            phase1 = np.zeros([self.NWAVE,self.NTHETA,nr])
            for ir in range(nr):
                kext1[:,ir] = kext[:,ir] * Nd[ir]
                ksca1[:,ir] = ksca[:,ir] * Nd[ir]
                phase1[:,:,ir] = phase[:,:,ir] * Nd[ir]

            #Integrating the arrays
            #kextout = simpson(kext1,x=rd,axis=1)
            #kscaout = simpson(ksca1,x=rd,axis=1)
            #phaseout = simpson(phase1,x=rd,axis=2)
            kextout = simps(kext1,x=rd,axis=1)
            kscaout = simps(ksca1,x=rd,axis=1) 
            phaseout = simps(phase1,x=rd,axis=2)

            #Integrating the particle size distribution
            #pnorm = simpson(Nd,x=rd)
            pnorm = simps(Nd,x=rd)

            #Calculating the mean properties
            xext = np.zeros(self.NWAVE)
            xsca = np.zeros(self.NWAVE)
            xabs = np.zeros(self.NWAVE)
            xext[:] = kextout/pnorm
            xsca[:] = kscaout/pnorm
            xabs = xext - xsca
            xphase = np.zeros([self.NWAVE,self.NTHETA])
            xphase[:,:] = phaseout/pnorm

        else:
            xext = np.zeros(self.NWAVE)
            xsca = np.zeros(self.NWAVE)
            xabs = np.zeros(self.NWAVE)
            xext[:] = kext[:,0]  
            xsca[:] = ksca[:,0]
            xabs = xext - xsca
            xphase = np.zeros([self.NWAVE,self.NTHETA])
            xphase = phase[:,:,0]

        #Normalising the phase function following equation 2.49 in Hansen and Travis (1974)
        #This normalisation reconciles the calculations between NemesisPy and Makephase (Fortran Nemesis)
        for i in range(self.NTHETA):
            xphase[:,i] = xphase[:,i] * (wavel * 1.0e-4)**2. / (xsca * np.pi)

        #Sorting again the arrays if ispace=0
        if self.ISPACE==0:  #Wavenumber
            wave = 1.0e4/wavel
            isort = np.argsort(wave)
            wave = wave[isort]
            xext = xext[isort]
            xsca = xsca[isort]
            xabs = xabs[isort]
            xphase[:,:] = xphase[isort,:]

        if iNorm==1:
            wave0,iwave = find_nearest(self.WAVE,WaveNorm)
            xext = xext / xext[iwave]
            xabs = xabs / xext[iwave]
            xsca = xsca / xext[iwave]

        self.KEXT[:,IDUST] = xext
        self.KABS[:,IDUST] = xabs
        self.KSCA[:,IDUST] = xsca
        self.SGLALB[:,IDUST] = xsca/xext
        self.PHASE[:,:,IDUST] = xphase[:,:]
 
        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(14,7))

            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1,rowspan=2)
            ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2,rowspan=1)
            ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2,rowspan=1)

            ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1,rowspan=1)
            ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1,rowspan=1)
            ax6 = plt.subplot2grid((3, 3), (2, 2), colspan=1,rowspan=1)

            if nr>1:

                import matplotlib

                colormap = 'viridis'
                norm = matplotlib.colors.Normalize(vmin=Nd.min(),vmax=Nd.max())
                c_m = plt.cm.get_cmap(colormap,360)
                # create a ScalarMappable and initialize a data structure
                s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
                s_m.set_array([])

                #Plotting the extinction cross section 
                iwave0 = 0
                iwave1 = int(self.NWAVE/2-1)
                iwave2 = self.NWAVE-1
                for ir in range(nr):
                    ax1.scatter(rd[ir],Nd[ir],c=s_m.to_rgba([Nd[ir]]),edgecolors='black')
                    ax2.semilogy(self.WAVE,kext1[:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                    ax3.semilogy(self.WAVE,ksca1[:,ir]/kext1[:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                    ax4.plot(self.THETA,phase1[iwave0,:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                    ax5.plot(self.THETA,phase1[iwave1,:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                    ax6.plot(self.THETA,phase1[iwave2,:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                #ax2.semilogy(self.WAVE,xext,c='tab:red')
                #ax3.semilogy(self.WAVE,xsca/xext,c='tab:red')
                #ax4.plot(self.THETA,xphase[iwave0,:],c='tab:red')
                #ax5.plot(self.THETA,xphase[iwave1,:],c='tab:red')
                #ax6.plot(self.THETA,xphase[iwave2,:],c='tab:red')
                if self.ISPACE==0:
                    ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
                    ax3.set_xlabel('Wavenumber (cm$^{-1}$)')
                else:
                    ax2.set_xlabel('Wavelength ($\mu$m)')
                    ax3.set_xlabel('Wavelength ($\mu$m)')
                ax1.set_ylabel('Density distribution')
                ax2.set_ylabel('Extinction cross section (cm$^2$)')
                ax3.set_ylabel('Single scattering albedo')
                ax4.set_ylabel('Phase function')
                ax5.set_ylabel('Phase function')
                ax6.set_ylabel('Phase function')

            ax1.set_xlabel('Particle size radii ($\mu$m)')
            ax4.set_xlabel('Angle (deg)')
            ax5.set_xlabel('Angle (deg)')
            ax6.set_xlabel('Angle (deg)')
            

            cax = plt.axes([0.92, 0.15, 0.02, 0.7])   #Bottom
            cbar2 = plt.colorbar(s_m,cax=cax,orientation='vertical')
            cbar2.set_label('Density distribution')

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax4.grid()
            ax5.grid()
            ax6.grid()


            plt.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.95, wspace=0.35, hspace=0.35)


    def miescat_k(self,IDUST,psdist,pardist,MakePlot=False,rdist=None,Ndist=None,WaveNorm=None):

        """
        Function to calculate the extinction coefficient, single scattering albedo and phase functions
        for different aerosol populations using Mie Theory.

        Inputs
        ________

        @param IDUST: int
            Integer indicating to which aerosol population this calculation corresponds to (from 0 to NDUST-1)
        @param psdist: int
            Flag indicating the particle size distribution
                0 :: Single particle size
                1 :: Log-normal distribution
                -1 :: Other input particle size distribution (using optional inputs)
        @param pardist: int
            Particle size distribution parameters
                psdist == 0 :: pardist(0) == particle size in microns
                psdist == 1 :: pardist(0) == mu
                               pardist(1) == sigma

        Optional inputs
        ________________

        rdist :: Array of particle sizes (to be used if pardist==-1)
        Ndist :: Density of particles of each particle size (to be used if pardist==-1)
        WaveNorm :: Wavelength/Wavenumber at which the cross sections will be normalised

        Outputs
        ________

        Updated KEXT,KABS,KSCA,PHASE in the Scatter_0 class

        """

        from scipy.integrate import simpson
        from scipy.stats import lognorm
        from scipy import interpolate
        from NemesisPy import lognormal_dist,find_nearest
        from copy import copy

        iNorm = 0
        if WaveNorm!=None:

            if self.WAVE.min()>WaveNorm:
                sys.exit('error in miescat :: WaveNorm must be within WAVE.min() and WAVE.max()')
            if self.WAVE.max()<WaveNorm:
                sys.exit('error in miescat :: WaveNorm must be within WAVE.min() and WAVE.max()')

            iNorm = 1

        #First we determine the number of particle sizes to later perform the integration
        ######################################################################################

        if psdist==0:   #Single particle size
            nr = 1
            rd = np.zeros(nr)   #Array of particle sizes
            Nd = np.zeros(nr)   #Weight (tipycally density) of each particle size for the integration
            rd[0] = pardist[0]
            Nd[0] = 1.0
        elif psdist==1: #Log-normal distribution
            mu = pardist[0]
            sigma = pardist[1]
            r0 = lognorm.ppf(0.0001, sigma, 0.0, mu)
            r1 = lognorm.ppf(0.9999, sigma, 0.0, mu)
            rmax = np.exp( np.log(mu) - sigma**2.)
            delr = (rmax-r0)/10.
            nr = int((r1-r0)/delr) + 1
            rd = np.linspace(r0,r1,nr) #Array of particle sizes
            Nd = lognormal_dist(rd,mu,sigma) #Density of each particle size for the integration
        elif psdist==2: #Standard gamma distribution
            sys.exit('error in miescat :: Standard gamma distribution has not yet been implemented')
        elif psdist==-1:
            if rdist[0]==None:
                sys.exit('error in miescat :: If psdist=-1 rdist and Ndist must be filled')
            if Ndist[0]==None:
                sys.exit('error in miescat :: If psdist=-1 rdist and Ndist must be filled')
            nr = len(rdist)
            rd = np.zeros(nr)   #Array of particle sizes
            Nd = np.zeros(nr)   #Weight (tipycally density) of each particle size for the integration
            rd[:] = rdist
            Nd[:] = Ndist
        else:
            sys.exit('error in miescat :: Type of distribution not yet implemented')

        #Interpolating the refractive index to the correct wavelength/wavenumber grid
        #####################################################################################

        refind_real = np.zeros(self.NWAVE)
        refind_im = np.zeros(self.NWAVE)

        f = interpolate.interp1d(self.WAVER,self.REFIND_REAL)
        refind_real[:] = f(self.WAVE)

        f = interpolate.interp1d(self.WAVER,self.REFIND_IM)
        refind_im[:] = f(self.WAVE)

        #Second we change the units of the spectral points if they are in wavenumbers
        ######################################################################################

        if self.ISPACE==0:  #Wavenumber
            wave1 = copy(self.WAVE)
            wave = 1.0e4/wave1
            isort = np.argsort(wave)
            wave = wave[isort]
            refind_real1 = refind_real[isort]
            refind_im1 = refind_im[isort]
            refind = refind_real1 - 1.0j * refind_im1
        elif self.ISPACE==1:  #Wavelength
            refind = refind_real - 1.0j * refind_im
        else:
            sys.exit('error in miescat :: ISPACE must be either 0 or 1')

        #We calculate the scattering and absorption coefficients of each particle size in the distribution
        ######################################################################################################

        kext = np.zeros([self.NWAVE,nr])
        ksca = np.zeros([self.NWAVE,nr])
        for ir in range(nr):

            r0 = rd[ir]
            x = np.zeros(self.NWAVE)
            x[:] = 2.*np.pi*r0/(self.WAVE)
            qext, qsca, qback, g = miepython.mie(refind,x)

            ksca[:,ir] = qsca * np.pi * (r0/1.0e4)**2.   #Cross section in cm2
            kext[:,ir] = qext * np.pi * (r0/1.0e4)**2.   #Cross section in cm2

        #Now integrating over particle size to find the mean scattering and absorption properties
        ###########################################################################################

        if nr>1:
            kext1 = np.zeros([self.NWAVE,nr])
            ksca1 = np.zeros([self.NWAVE,nr])
            for ir in range(nr):
                kext1[:,ir] = kext[:,ir] * Nd[ir]
                ksca1[:,ir] = ksca[:,ir] * Nd[ir]

            #Integrating the arrays
            kextout = simpson(kext1,x=rd,axis=1)
            kscaout = simpson(ksca1,x=rd,axis=1)

            #Integrating the particle size distribution
            pnorm = simpson(Nd,x=rd)

            #Calculating the mean properties
            xext = np.zeros(self.NWAVE)
            xsca = np.zeros(self.NWAVE)
            xabs = np.zeros(self.NWAVE)
            xext[:] = kextout/pnorm
            xsca[:] = kscaout/pnorm
            xabs = xext - xsca

        else:
            xext = np.zeros(self.NWAVE)
            xsca = np.zeros(self.NWAVE)
            xabs = np.zeros(self.NWAVE)
            xext[:] = kext[:,0]  
            xsca[:] = ksca[:,0]
            xabs = xext - xsca

        #Sorting again the arrays if ispace=0
        if self.ISPACE==0:  #Wavenumber
            wave = 1.0e4/wave
            isort = np.argsort(wave)
            wave = wave[isort]
            xext = xext[isort]
            xsca = xsca[isort]
            xabs = xabs[isort]

        if iNorm==1:
            wave0,iwave = find_nearest(self.WAVE,WaveNorm)
            xext = xext / xext[iwave]
            xabs = xabs / xext[iwave]
            xsca = xsca / xext[iwave]

        self.KEXT[:,IDUST] = xext
        self.KABS[:,IDUST] = xabs
        self.KSCA[:,IDUST] = xsca
        self.SGLALB[:,IDUST] = xsca/xext
