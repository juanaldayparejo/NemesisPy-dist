from NemesisPy.Data import *
from scipy.interpolate import interp1d
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

class Surface_0:

    def __init__(self, GASGIANT=False, LOWBC=1, GALB=1.0, NEM=2, NLOCATIONS=1):

        """
        Inputs
        ------
        @param GASGIANT: log,
            Flag indicating whether the planet has surface (GASGIANT=False) or not (GASGIANT=True)

        @param LOWBC: int,
            Flag indicating the lower boundary condition.
                0 :: Thermal emission only (i.e. no reflection)
                1 :: Lambertian surface
                2 :: Hapke surface
 
        @param GALB: int,
            Ground albedo
            
        @param NEM: int,
            Number of spectral points defining the emissivity of the surface   
        
        Attributes
        ----------

        @attribute TSURF: real
            Surface temperature (K)

        Attributes for Thermal emission from surface:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        @attribute VEM: 1D array
            Wavelengths at which the emissivity and other surface parameters are defined

        @attribute EMISSIVITY: 1D array
            Surface emissitivity 


        Attributes for Lambertian surface:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        None. The albedo of a lambertian surface is given in the .set file by GALB, 
        and if GALB<0, it is calculated as 1.0 - EMISSIVITY.


        Attributes for Hapke surface:
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The method implemented here is derived in Hapke (2012) - Theory of Reflectance and Emittance Spectroscopy
        In particular, it is derived in chapter 12.3.1 of that book. We also assume that the scattering phase function
        of the surface is given by a Double Henyey-Greenstein function. 

        @attribute SGLALB : 1D array
            Single scattering albedo w

        @attribute K : 1D array
            Porosity coefficient 

        @attribute BS0 : 1D array
            Amplitude of opposition effect (0 to 1)    

        @attribute hs : 1D array
            Width of the opposition surge

        @attribute BC0 : 1D array
            Amplitude of the coherent backscatter opposition effect (0 to 1)

        @attribute hc : 1D array
            Width of the backscatter function

        @attribute ROUGHNESS : 1D array
            Roughness mean slope angle (degrees)

        @attribute G1 : 1D array
            Asymmetry factor of the first Henyey-Greenstein function defining the phase function

        @attribute G2 : 1D array
            Asymmetry factor of the second Henyey-Greenstein function defining the phase function
        
        @attribute F: 2D array,
            Parameter defining the relative contribution of G1 and G2 of the double Henyey-Greenstein phase function



        Methods
        -------
        Surface_0.edit_EMISSIVITY
        Surface_0.read_sur()
        Surface_0.read_hap()


        """

        #Input parameters
        self.NLOCATIONS = NLOCATIONS
        self.GASGIANT = GASGIANT
        self.LOWBC = LOWBC
        self.GALB = GALB
        self.NEM = NEM

        # Input the following profiles using the edit_ methods.
        self.TSURF = None 
        self.VEM = None #(NEM)
        self.EMISSIVITY = None #(NEM) 

        #Hapke parameters
        self.SGLALB = None #(NEM)
        self.BS0 = None #(NEM)
        self.hs = None #(NEM)
        self.BC0 = None #(NEM)
        self.hc = None #(NEM)
        self.K = None #(NEM)
        self.ROUGHNESS = None #(NEM)
        self.G1 = None #(NEM)
        self.G2 = None #(NEM)
        self.F = None #(NEM)

    def edit_EMISSIVITY(self, EMISSIVITY_array):
        """
        Edit the surface emissivity at each of the lat/lon points
        @param EMISSIVITY_array: 3D array
            Array defining the surface emissivity at each of the points
        """
        EMISSIVITY_array = np.array(EMISSIVITY_array)
        assert len(EMISSIVITY_array) == self.NEM , \
            'EMISSIVITY should have NEM elements'
        self.EMISSIVITY = EMISSIVITY_array  



    def calc_phase_angle(self,EMISS_ANG,SOL_ANG,AZI_ANG):
        """
        Calculate the phase angle based on the emission, incident and azimuth angles

        Inputs
        ------
        @param EMISS_ANG: 1D array or scalar
            Emission angle (deg)
        @param SOL_ANG: 1D array or scalar
            Solar zenith or incident angle (deg)
        @param AZI_ANG: 1D array or scalar
            Azimuth angle (deg)

        Outputs
        -------
        @param PHASE_ANG: 1D array or scalar
            Phase angle (deg)
        """

        #First of all let's calculate the scattering phase angle
        mu = np.cos(EMISS_ANG/180.*np.pi)   #Cosine of the reflection angle
        mu0 = np.cos(SOL_ANG/180.*np.pi)    #Coside of the incidence angle

        cg = mu * mu0 + np.sqrt(1. - mu**2.) * np.sqrt(1.-mu0**2.) * np.cos(AZI_ANG/180.*np.pi)
        iin = np.where(cg>1.0)
        cg[iin] = 1.0 
        g = np.arccos(cg)/np.pi*180.   #Scattering phase angle (degrees) (NTHETA)

        return g

    ##################################################################################################################
    ##################################################################################################################
    #                                                   EMISSIVITY
    ##################################################################################################################
    ##################################################################################################################

    def read_sur(self, runname):
        """
        Read the surface emissivity from the .sur file
        @param runname: str
            Name of the Nemesis run
        """
        
        #Opening file
        f = open(runname+'.sur','r')
        nem = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
    
        vem = np.zeros([nem])
        emissivity = np.zeros([nem])
        for i in range(nem):
            tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
            vem[i] = tmp[0]
            emissivity[i] = tmp[1]

        self.NEM = nem
        self.VEM = vem
        self.EMISSIVITY = emissivity

    ##################################################################################################################

    def write_sur(self, runname):
        """
        Write the surface emissivity into the .sur file
        @param runname: str
            Name of the Nemesis run
        """
        
        #Opening file
        f = open(runname+'.sur','w')
        f.write('%i \n' % (self.NEM))
        for i in range(self.NEM):
            f.write('%7.4e \t %7.4e \n' % (self.VEM[i],self.EMISSIVITY[i]))
        f.close()

    ##################################################################################################################


    def calc_radground(self,ISPACE,WAVE=None):
        """
        Calculate the thermal emission from the surface

        @param ISPACE: int
            Units of wavelength array
            (0) Wavenumber in cm-1 (1) Wavelength in microns

        @param WAVE: 1D array
            Wavelength or wavenumber array
        """

        from NemesisPy import planck

        if WAVE is None:
            WAVE = self.VEM

        bbsurf = planck(ISPACE,WAVE,self.TSURF)

        f = interp1d(self.VEM,self.EMISSIVITY)
        emissivity = f(WAVE)

        radground = bbsurf * emissivity

        return radground


    ##################################################################################################################
    ##################################################################################################################
    #                                               LAMBERT REFLECTANCE
    ##################################################################################################################
    ##################################################################################################################


    def calc_Lambert_BRDF(self,ALBEDO,SOL_ANG):
        """
        Calculate the reflectance distribution function for a Lambertian surface.

        @param ALBEDO: 1D array (NWAVE)
            Lambert albedo
        @param SOL_ANG: 1D array (NTHETA)
            Solar zenith angle (degrees)

        @param BRDF: 2D array (NWAVE,NTHETA)
            Bidirectional reflectance
        """

        from NemesisPy.nemesisf import hapke

        #Inserting angles into array if they are a scalar
        if np.isscalar(SOL_ANG)==True:
            SOL_ANG = np.array([SOL_ANG])
        else:
            SOL_ANG = np.array(SOL_ANG)

        if np.isscalar(ALBEDO)==True:
            ALBEDO = np.array([ALBEDO])
        else:
            ALBEDO = np.array(ALBEDO)

        NTHETA = len(SOL_ANG)
        NWAVE = len(ALBEDO)

        BRDF = np.repeat(ALBEDO[:,np.newaxis],NTHETA,axis=1) / np.pi * np.cos(SOL_ANG/180.*np.pi)

        return BRDF

    ##################################################################################################################

    def calc_albedo(self):
        """
        Calculate the Lambert albedo of the surface based on the value on the class

        If GALB<0.0 then the Lambert albedo is calculated from the surface emissivity
        """

        if self.GALB>0.0:
            ALBEDO = np.ones(self.NEM)*self.GALB
        else:
            ALBEDO = np.zeros(self.NEM)
            ALBEDO[:] = 1.0 - self.EMISSIVITY[:]

        return ALBEDO


    ##################################################################################################################
    ##################################################################################################################
    #                                         HAPKE BIDIRECTIONAL-REFLECTANCE
    ##################################################################################################################
    ##################################################################################################################    


    def read_hap(self, runname):
        """
        Read the Hapke parameters of the surface from the .hap file
        @param runname: str
            Name of the Nemesis run
        """
        
        #Opening file
        f = open(runname+'.hap','r')

        #Reading number of wavelengths
        nem = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
    
        #Defining all fields
        vem = np.zeros(nem)
        sglalb = np.zeros(nem)
        k = np.zeros(nem)
        bso = np.zeros(nem)
        hs = np.zeros(nem)
        bco = np.zeros(nem)
        hc = np.zeros(nem)
        roughness = np.zeros(nem)
        g1 = np.zeros(nem)
        g2 = np.zeros(nem)
        fhg = np.zeros(nem)

        #Reading Hapke parameters
        for i in range(nem):
            tmp = np.fromfile(f,sep=' ',count=11,dtype='float')
            vem[i] = tmp[0]
            sglalb[i] = tmp[1]
            k[i] = tmp[2]
            bso[i] = tmp[3]
            hs[i] = tmp[4]
            bco[i] = tmp[5]
            hc[i] = tmp[6]
            roughness[i] = tmp[7]
            g1[i] = tmp[8]
            g2[i] = tmp[9]
            fhg[i] = tmp[10]

        f.close()

        #Storing parameters in the class
        self.NEM = nem
        self.VEM = vem
        self.SGLALB = sglalb
        self.K = k
        self.BS0 = bso
        self.hs = hs
        self.BC0 = bco
        self.hc = hc
        self.ROUGHNESS = roughness
        self.G1 = g1
        self.G2 = g2
        self.F = fhg

    ##################################################################################################################

    def write_hap(self,runname):
        """
        Read the Hapke parameters stored in the class into the .hap file
        @param runname: str
            Name of the Nemesis run
        """

        f = open(runname+'.hap','w')
        f.write('%i \n' % (self.NEM))
        for i in range(self.NEM):
            f.write('%7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \t %7.4e \n' % \
                (self.VEM[i],self.SGLALB[i],self.K[i],self.BS0[i],self.hs[i],self.BC0[i],self.hc[i],self.ROUGHNESS[i],self.G1[i],self.G2[i],self.F[i]))
        f.close()

    ##################################################################################################################

    def calc_Hapke_BRDF(self,EMISS_ANG,SOL_ANG,AZI_ANG,WAVE=None):
        """
        Calculate the bidirectional-reflectance distribution function for a Hapke surface.
        The method used here is described in Hapke (2012): Theory of Reflectance and Emittance
        Spectroscopy, in chapter 12.3.1 (disk-resolved photometry)

        @param EMISS_ANG: float
            Emission angle (degrees)
        @param SOL_ANG: float
            Solar zenith angle (degrees)
        @param AZI_ANG: float
            Azimuth angle (degrees)
        """

        from NemesisPy.nemesisf import hapke

        #Inserting angles into array if they are a scalar
        if np.isscalar(EMISS_ANG)==True:
            EMISS_ANG = np.array([EMISS_ANG])
            SOL_ANG = np.array([SOL_ANG])
            AZI_ANG = np.array([AZI_ANG])
        else:
            EMISS_ANG = np.array(EMISS_ANG)
            SOL_ANG = np.array(SOL_ANG)
            AZI_ANG = np.array(AZI_ANG)

        NTHETA = len(EMISS_ANG)

        #Interpolating surface values if wavelength array is specified
        if WAVE is not None:
            s = interp1d(self.VEM,self.SGLALB)
            SGLALB = s(WAVE)
            s = interp1d(self.VEM,self.K)
            K = s(WAVE)
            s = interp1d(self.VEM,self.BS0)
            BS0 = s(WAVE)
            s = interp1d(self.VEM,self.hs)
            hs = s(WAVE)
            s = interp1d(self.VEM,self.BC0)
            BC0 = s(WAVE)
            s = interp1d(self.VEM,self.hc)
            hc = s(WAVE)
            s = interp1d(self.VEM,self.ROUGHNESS)
            ROUGHNESS = s(WAVE)
            s = interp1d(self.VEM,self.G1)
            G1 = s(WAVE)
            s = interp1d(self.VEM,self.G2)
            G2 = s(WAVE)
            s = interp1d(self.VEM,self.F)
            F = s(WAVE)
        else:
            SGLALB = self.SGLALB
            K = self.K
            BS0 = self.BS0
            hs = self.hs
            BC0 = self.BC0
            hc = self.hc
            ROUGHNESS = self.ROUGHNESS
            G1 = self.G1
            G2 = self.G2
            F = self.F

        #Calling the fortran module to calculate Hapke's BRDF
        BRDF = hapke.hapke_brdf(SGLALB,K,BS0,hs,BC0,hc,ROUGHNESS,G1,G2,F,\
                                SOL_ANG,EMISS_ANG,AZI_ANG)

        return BRDF

    ##################################################################################################################

    def calc_Hapke_BRDF1(self,EMISS_ANG,SOL_ANG,AZI_ANG):
        """
        Calculate the bidirectional-reflectance distribution function for a Hapke surface.
        The method used here is described in Hapke (2012): Theory of Reflectance and Emittance
        Spectroscopy, in chapter 12.3.1 (disk-resolved photometry)

        @param EMISS_ANG: float
            Emission angle (degrees)
        @param SOL_ANG: float
            Solar zenith angle (degrees)
        @param AZI_ANG: float
            Azimuth angle (degrees)
        """

        #Inserting angles into array if they are a scalar
        if np.isscalar(EMISS_ANG)==True:
            EMISS_ANG = np.array([EMISS_ANG])
            SOL_ANG = np.array([SOL_ANG])
            AZI_ANG = np.array([AZI_ANG])
        else:
            EMISS_ANG = np.array(EMISS_ANG)
            SOL_ANG = np.array(SOL_ANG)
            AZI_ANG = np.array(AZI_ANG)

        NTHETA = len(EMISS_ANG)

        #Making all quantities have the same dimensions 
        EMISS_ANG = np.repeat(EMISS_ANG[:,np.newaxis],self.NEM,axis=1)  #(NTHETA,NWAVE)
        SOL_ANG = np.repeat(SOL_ANG[:,np.newaxis],self.NEM,axis=1)      #(NTHETA,NWAVE)
        AZI_ANG = np.repeat(AZI_ANG[:,np.newaxis],self.NEM,axis=1)      #(NTHETA,NWAVE)

        #First of all let's calculate the scattering phase angle
        mu = np.cos(EMISS_ANG/180.*np.pi)   #Cosine of the reflection angle
        mu0 = np.cos(SOL_ANG/180.*np.pi)    #Coside of the incidence angle

        cg = mu * mu0 + np.sqrt(1. - mu**2.) * np.sqrt(1.-mu0**2.) * np.cos(AZI_ANG/180.*np.pi) 
        g = np.arccos(cg)/np.pi*180.   #Scattering phase angle (degrees) (NTHETA)

        #Calculate some of the parameters from the Hapke formalism
        gamma = self.calc_Hapke_gamma()  #(NWAVE)
        r0 = self.calc_Hapke_r0(gamma)  #(NWAVE)
        theta_bar = self.calc_Hapke_thetabar(r0) #(NWAVE)

        #Making all quantities have the same dimensions
        gamma = np.repeat(gamma[np.newaxis,:],NTHETA,axis=0)         #(NTHETA,NWAVE)
        r0 = np.repeat(r0[np.newaxis,:],NTHETA,axis=0)               #(NTHETA,NWAVE)
        theta_bar = np.repeat(theta_bar[np.newaxis,:],NTHETA,axis=0) #(NTHETA,NWAVE)

        E1e = self.calc_Hapke_E1(EMISS_ANG,theta_bar)    #(NTHETA,NWAVE)
        E2e = self.calc_Hapke_E2(EMISS_ANG,theta_bar)    #(NTHETA,NWAVE)
        E1i = self.calc_Hapke_E1(SOL_ANG,theta_bar)      #(NTHETA,NWAVE)
        E2i = self.calc_Hapke_E2(SOL_ANG,theta_bar)      #(NTHETA,NWAVE)
        chi = self.calc_Hapke_chi(theta_bar)             #(NTHETA,NWAVE)
        f = np.exp(-2.0*np.tan(AZI_ANG/2.0/180.*np.pi))  #(NTHETA,NWAVE)

        #Calculating the effective incidence and reflection angles
        mu0eff, mueff = self.calc_Hapke_eff_angles(SOL_ANG,EMISS_ANG,AZI_ANG,theta_bar,E1e,E1i,E2e,E2i,chi)

        #Calculating the nu functions
        nue = self.calc_Hapke_nu(EMISS_ANG,theta_bar,E1e,E2e,chi)
        nui = self.calc_Hapke_nu(SOL_ANG,theta_bar,E1i,E2i,chi)

        #Calculating the Shadowing function S
        if SOL_ANG<=EMISS_ANG:
            S = mueff/nue * mu0/nui * chi / (1.0 - f + f*chi*mu0/nui)
        else:
            S = mueff/nue * mu0/nui * chi / (1.0 - f + f*chi*mu/nue)

        #Calculating the phase function at the scattering angles
        phase = self.calc_hgphase(g)   #(NWAVE,NTHETA)

        #Calculating the shadow-hiding opposition function Bs
        Bs = self.BS0 / ( 1.0 + (1.0/self.hs) + np.tan( g/2./180.*np.pi) )

        #Calculating the backscatter anfular function Bc
        Bc = self.BC0 / ( 1.0 + (1.3 + self.K) * ( (1.0/self.hc*np.tan( g/2./180.*np.pi)) + (1.0/self.hc*np.tan( g/2./180.*np.pi))**2.0 ) )

        #Calculating the Ambartsumian–Chandrasekhar H function
        H0e = self.calc_Hapke_H(mu0eff/self.K,r0)
        He = self.calc_Hapke_H(mueff/self.K,r0)

        #Calculating the bidirectional reflectance
        BRDF = self.K * self.SGLALB / (4.0*np.pi) * mu0eff / (mu0eff + mueff) * \
            ( phase*(1.0+Bs) + (H0e*He-1.0) ) * (1.0+Bc) * S

        return BRDF


    ##################################################################################################################

    def calc_Hapke_H(self,x,r0):
        """
        Calculate the Ambartsumian–Chandrasekhar H function of the Hapke formalism (Hapke, 2012; p. 333)

        Inputs
        ------

        @param SGLALB: 1D array (NWAVE)
            Single scattering albedo

        @param x: 1D array
            Value at which the H-function must be evaluated

        @param r0: 1D array (NWAVE)
            r0 parameter from the Hapke formalism

        Outputs
        -------

        @param H: 1D array or real scalar
            H function

        """

        H = 1.0 / ( 1.0 - self.SGLALB/2.0*x * (r0 + (1.0 - 2.0*r0*x)/2.0*np.log((1.0+x)/x)) )

        return H

##################################################################################################################

    def calc_Hapke_thetabar(self,r0):
        """
        Calculate the theta_bar parameter of the Hapke formalism (Hapke, 2012; p. 333)
        This parameter is the corrected roughness mean slope angle

        Inputs
        ------

        @param ROUGHNESS: 1D array or real scalar
            Roughness mean slope angle (degrees)

        @param r0: 1D array or real scalar
            Diffusive reflectance

        Outputs
        -------

        @param theta_bar: 1D array or real scalar
            Corrected Roughness mean slope angle (degrees)

        """

        theta_bar = self.ROUGHNESS * (1.0 - r0)

        return theta_bar

##################################################################################################################

    def calc_Hapke_gamma(self):
        """
        Calculate the gamma parameter of the Hapke formalism (Hapke, 2012; p. 333)
        This parameter is just a factor calculated from the albedo

        Inputs
        ------

        @param SGLALB: 1D array or real scalar
            Single scattering albedo

        Outputs
        -------

        @param gamma: 1D array or real scalar
            Gamma factor

        """

        gamma = np.sqrt(1.0 - self.SGLALB)

        return gamma

##################################################################################################################

    def calc_Hapke_r0(self,gamma):
        """
        Calculate the r0 parameter of the Hapke formalism (Hapke, 2012; p. 333)
        This parameter is called the diffusive reflectance

        Inputs
        ------

        @param gamma: 1D array or real scalar
            Gamma factor

        Outputs
        -------

        @param r0: 1D array or real scalar
            Diffusive reflectance

        """

        r0 = (1.0 - gamma)/(1.0 + gamma)

        return r0

    ##################################################################################################################

    def calc_Hapke_eff_angles(self,i,e,phi,theta_bar,E1e,E1i,E2e,E2i,chi):
        """
        Calculate the effective incidence and reflection angles 

        Inputs
        ------

        @param i: 1D array (NTHETA)
            Incidence angle (degrees)  

        @param e: 1D array (NTHETA)
            Reflection angle (degrees)

        @param phi: 1D array (NTHETA)
            Azimuth angle (degrees)

        @param theta_bar: 1D array (NEM)
            Corrected roughness mean slope angle (degrees)

        @param E1e,E1i,E2e,E2i,chi: 2D array (NWAVE,NTHETA)
            Several different coefficients from the Hapke formalism         

        Outputs
        -------

        @param mu0_eff: 1D array or real scalar
            Cosine of the effective incidence angle

        @param mu_eff: 1D array or real scalar
            Cosine of the effective reflection angle

        """

        #Calculating some initial parameters
        irad = i / 180. * np.pi  
        erad = e / 180. * np.pi
        phirad = phi / 180. * np.pi 
        tbarrad = theta_bar / 180. * np.pi

        i1 = np.where(i<=e)
        i2 = np.where(e<i)

        mu0eff = np.zeros(i.shape)
        mueff = np.zeros(e.shape)

        mu0eff[i1] = chi[i1] * ( np.cos(irad[i1]) + np.sin(irad[i1]) * np.tan(tbarrad[i1]) * \
                (np.cos(phirad[i1]) * E2e[i1] + np.sin(phirad[i1]/2.)**2. *E2i[i1]) / (2.0 - E1e[i1] - phirad[i1]/np.pi*E1i[i1])  )

        mueff[i1] = chi[i1] * ( np.cos(erad[i1]) + np.sin(erad[i1]) * np.tan(tbarrad[i1]) * \
                (E2e[i1] - np.sin(phirad[i1]/2.)**2. *E2i[i1]) / (2.0 - E1e[i1] - phirad[i1]/np.pi*E1i[i1])  )

        mu0eff[i2] = chi[i2] * ( np.cos(irad[i2]) + np.sin(irad[i2]) * np.tan(tbarrad[i2]) * \
            (E2i[i2] - np.sin(phirad[i2]/2.)**2. *E2e[i2]) / (2.0 - E1i[i2] - phirad[i2]/np.pi*E1e[i2])  )

        mueff[i2] = chi[i2] * ( np.cos(erad[i2]) + np.sin(erad[i2]) * np.tan(tbarrad[i2]) * \
            (np.cos(phirad[i2]) * E2i[i2] + np.sin(phirad[i2]/2.)**2. *E2e[i2]) / (2.0 - E1i[i2] - phirad[i2]/np.pi*E1e[i2])  )

        #There are two possible cases
        '''
        if i<=e:

            mu0eff = chi * ( np.cos(irad) + np.sin(irad) * np.tan(tbarrad) * \
                (np.cos(phirad) * E2e + np.sin(phirad/2.)**2. *E2i) / (2.0 - E1e - phirad/np.pi*E1i)  )

            mueff = chi * ( np.cos(erad) + np.sin(erad) * np.tan(tbarrad) * \
                (E2e - np.sin(phirad/2.)**2. *E2i) / (2.0 - E1e - phirad/np.pi*E1i)  )

        elif i>e:

            mu0eff = chi * ( np.cos(irad) + np.sin(irad) * np.tan(tbarrad) * \
                (E2i - np.sin(phirad/2.)**2. *E2e) / (2.0 - E1i - phirad/np.pi*E1e)  )

            mueff = chi * ( np.cos(erad) + np.sin(erad) * np.tan(tbarrad) * \
                (np.cos(phirad) * E2i + np.sin(phirad/2.)**2. *E2e) / (2.0 - E1i - phirad/np.pi*E1e)  )
        '''


        return mu0eff, mueff

    ##################################################################################################################

    def calc_Hapke_nu(self,x,theta_bar,E1x,E2x,chi):
        """
        Calculate the nu function from the Hapke formalism (Hapke 2012 p.333) 

        Inputs
        ------

        @param x: 1D array or real scalar
            Incidence or reflection angles (degrees)

        @param theta_bar: 1D array or real scalar
            Corrected roughness mean slope angle (degrees)

        @param E1x,E1x,chi: 1D array or real scalar
            Several different coefficients from the Hapke formalism (evaluated at the angle x)       

        Outputs
        -------

        @param nu: 1D array or real scalar
            Nu parameter from the Hapke formalism

        """

        #Calculating some initial parameters
        xrad = x / 180. * np.pi
        tbarrad = theta_bar / 180. * np.pi

        nu = chi * ( np.cos(xrad) + np.sin(xrad) * np.tan(tbarrad) * \
            (E2x) / (2.0 - E1x)  )

        return nu


##################################################################################################################

    def calc_Hapke_E1(self,x,theta_bar):
        """
        Calculate the E1 function of the Hapke formalism (Hapke, 2012; p. 333)

        Inputs
        ------

        @param x: 1D array (NTHETA)
            Angle (degrees)

        @param theta_bar: 1D array (NWAVE)
            Mean slope angle (degrees)

        Outputs
        -------

        @param E1: 2D array (NWAVE,NTHETA)
            Parameter E1 in the Hapke formalism

        """

        E1 = np.exp(-2.0/np.pi * 1.0/np.tan(theta_bar/180.*np.pi) * 1./np.tan(x/180.*np.pi))

        return E1


##################################################################################################################

    def calc_Hapke_E2(self,x,theta_bar):
        """
        Calculate the E2 function of the Hapke formalism (Hapke, 2012; p. 333)

        Inputs
        ------

        @param x: 1D array (NTHETA)
            Angle (degrees)

        @param theta_bar: 1D (NWAVE)
            Mean slope angle (degrees)

        Outputs
        -------

        @param E2: 1D array (NTHETA)
            Parameter E2 in the Hapke formalism

        """

        E2 = np.exp(-1.0/np.pi * 1.0/np.tan(theta_bar/180.*np.pi)**2. * 1./np.tan(x/180.*np.pi)**2.)

        return E2

##################################################################################################################

    def calc_Hapke_chi(self,theta_bar):
        """
        Calculate the chi function of the Hapke formalism (Hapke, 2012; p. 333)

        Inputs
        ------

        @param theta_bar: 1D array or real scalar
            Corrected roughness mean slope angle (degrees)

        Outputs
        -------

        @param chi: 1D array or real scalar
            Parameter chi in the Hapke formalism

        """

        chi = 1./np.sqrt(1.0 + np.pi * np.tan(theta_bar/180.*np.pi)**2.)

        return chi


    ##################################################################################################################

    def calc_hgphase(self,Theta):
        """
        Calculate the phase function at Theta angles given the double Henyey-Greenstein parameters
        @param Theta: 1D array or real scalar
            Scattering angle (degrees)
        """

        if np.isscalar(Theta)==True:
            ntheta = 1
            Thetax = np.array([Theta])
        else:
            Thetax = Theta

        #Re-arranging the size of Thetax to be (NTHETA,NWAVE)
        Thetax = np.repeat(Thetax[:,np.newaxis],self.NEM,axis=1)

        t1 = (1.-self.G1**2.)/(1. - 2.*self.G1*np.cos(Thetax/180.*np.pi) + self.G1**2.)**1.5
        t2 = (1.-self.G2**2.)/(1. - 2.*self.G2*np.cos(Thetax/180.*np.pi) + self.G2**2.)**1.5
        
        phase = self.F * t1 + (1.0 - self.F) * t2

        #Re-arranging the size of phase to be (NWAVE,NTHETA)
        phase = np.transpose(phase,axes=[1,0])

        return phase

