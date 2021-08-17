from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Aug 09 17:27:12 2021

@author: juanalday

Collision-Induced Absorption Class.
"""

class CIA_0:

    def __init__(self, INORMAL=0, NPAIR=9, NT=25, NWAVE=1501):

        """
        Inputs
        ------
        @param INORMAL: int,
            Flag indicating whether the ortho/para-H2 ratio is in equilibrium (0 for 1:1) or normal (1 for 3:1)
        @param NPAIR: int,
            Number of gaseous pairs listed 
            (Default = 9 : H2-H2 (eqm), H2-He (eqm), H2-H2 (normal), H2-He (normal), H2-N2, H2-CH4, N2-N2, CH4-CH4, H2-CH4)
        @param NT: int,
            Number of temperature levels over which the CIA data is defined 
        @param NWAVE: int,
            Number of spectral points over which the CIA data is defined

        Attributes
        ----------
        @attribute WAVEN: 1D array
            Wavenumber array (NOTE: ALWAYS IN WAVENUMBER, NOT WAVELENGTH)
        @attribute TEMP: 1D array
            Temperature levels at which the CIA data is defined (K)
        @attribute K_CIA: 1D array
            CIA cross sections for each pair at each wavenumber and temperature level

        Methods
        ----------
        CIA_0.read_cia(runname)
        """

        #Input parameters
        self.INORMAL = INORMAL
        self.NPAIR = NPAIR
        self.NT = NT
        self.NWAVE = NWAVE

        # Input the following profiles using the edit_ methods.
        self.WAVEN = None # np.zeros(NWAVE)
        self.TEMP = None # np.zeros(NT)
        self.K_CIA = None #np.zeros(NPAIR,NT,NWAVE)


    def read_cia(self,runname,raddata='/Users/aldayparejo/Documents/Projects/PlanetaryScience/NemesisPy-dist/NemesisPy/Data/cia/'):
        """
        Read the .cia file
        @param runname: str
            Name of the NEMESIS run
        """

        from scipy.io import FortranFile
        
        #Reading .cia file
        f = open(runname+'.cia','r')
        s = f.readline().split()
        cianame = s[0]
        s = f.readline().split()
        dnu = float(s[0])
        s = f.readline().split()
        npara = int(s[0])
        f.close()

        if npara!=0:
            sys.exit('error in read_cia :: routines have not been adapted yet for npara!=0')

        #Reading the actual CIA file
        if npara==0:
            NPAIR = 9

        f = FortranFile(raddata+cianame, 'r' )
        TEMPS = f.read_reals( dtype='float64' )
        KCIA_list = f.read_reals( dtype='float32' )
        NT = len(TEMPS)
        NWAVE = int(len(KCIA_list)/NT/NPAIR)

        NU_GRID = np.linspace(0,dnu*(NWAVE-1),NWAVE)
        K_CIA = np.zeros([NPAIR, NT, NWAVE])
    
        index = 0
        for iwn in range(NWAVE):
            for itemp in range(NT):
                for ipair in range(NPAIR):
                    K_CIA[ipair,itemp,iwn] = KCIA_list[index]
                    index += 1

        self.NWAVE = NWAVE
        self.NT = NT
        self.NPAIR = NPAIR
        self.WAVEN = NU_GRID
        self.TEMP = TEMPS
        self.K_CIA = K_CIA

    def calc_tau_cia(self,ISPACE,WAVEC,Atmosphere,Layer,MakePlot=False):
        """
        Calculate the CIA opacity in each atmospheric layer
        @param ISPACE: int
            Flag indicating whether the calculation must be performed in wavenumbers (0) or wavelength (1)
        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Atmosphere: class
            Python class defining the reference atmosphere
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations
        """

        from scipy import interpolate
        from NemesisPy import find_nearest

#       the mixing ratios of the species contributing to CIA
        qh2=np.zeros(Layer.NLAY)
        qhe=np.zeros(Layer.NLAY)
        qn2=np.zeros(Layer.NLAY)
        qch4=np.zeros(Layer.NLAY)
        qco2=np.zeros(Layer.NLAY)

        for i in range(Atmosphere.NVMR):

            if Atmosphere.ID[i]==39:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    qh2[:] = Layer.PP[:,i] / Layer.PRESS[:]

            if Atmosphere.ID[i]==40:
                qhe[:] = Layer.PP[:,i] / Layer.PRESS[:]

            if Atmosphere.ID[i]==22:
                qn2[:] = Layer.PP[:,i] / Layer.PRESS[:]

            if Atmosphere.ID[i]==6:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    qch4[:] = Layer.PP[:,i] / Layer.PRESS[:]   

            if Atmosphere.ID[i]==2:
                qco2[:] = Layer.PP[:,i] / Layer.PRESS[:]

#       calculating the opacity
        XLEN = Layer.DELH * 1.0e2  #cm
        TOTAM = Layer.TOTAM * 1.0e-4 #cm-2
        AMAGAT = 2.68675E19 #mol cm-3

        amag1 = (Layer.TOTAM*1.0e-4/XLEN)/AMAGAT  #Number density in AMAGAT units
        tau = XLEN*amag1**2

        #Defining the calculation wavenumbers
        if ISPACE==0:
            WAVEN = WAVEC
        elif ISPACE==1:
            WAVEN = 1.e4/WAVEC
            isort = np.argsort(WAVEN)
            WAVEN = WAVEN[isort]

        if((WAVEN.min()<self.WAVEN.min()) or (WAVEN.max()>self.WAVEN.max())):
            print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

#       calculating the CIA opacity at the correct temperature and wavenumber
        NWAVEC = len(WAVEC)   #Number of calculation wavelengths
        tau_cia_layer = np.zeros([NWAVEC,Layer.NLAY])
        for ilay in range(Layer.NLAY):

            #Interpolating to the correct temperature
            temp1 = Layer.TEMP[ilay]
            temp0,it = find_nearest(self.TEMP,temp1)

            if self.TEMP[it]>=temp1:
                ithi = it
                if it==0:
                    itl = 0
                else:
                    itl = it - 1

            elif self.TEMP[it]<temp1:
                itl = it
                if it==self.NT-1:
                    ithi = self.NT - 1
                else:
                    ithi = it + 1

            ktlo = self.K_CIA[:,itl,:]
            kthi = self.K_CIA[:,ithi,:]

            if itl==ithi:
                fhl = 0.0
                fhh = 1.0
            else:
                fhl = (temp1 - self.TEMP[itl])/(self.TEMP[ithi] - self.TEMP[itl])
                fhh = (self.TEMP[ithi] - temp1)/(self.TEMP[ithi] - self.TEMP[itl])

            kt = ktlo*(1.-fhl) + kthi * (1.-fhh)

            #Cheking that interpolation can be performed to the calculation wavenumbers
            inwave = np.where( (self.WAVEN>=WAVEN.min()) & (self.WAVEN<=WAVEN.max()) )
            inwave = inwave[0]
            if len(inwave)>0: 

                k_cia = np.zeros([NWAVEC,self.NPAIR])
                inwave1 = np.where( (WAVEN>=self.WAVEN.min()) & (WAVEN<=self.WAVEN.max()) )
                inwave1 = inwave1[0]

                #fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6))
                #labels = ['H2-H2 (eqm)','H2-He (eqm)','H2-H2 (normal)','H2-He (normal)','H2-N2','H2-CH4','N2-N2','CH4-CH4','H2-CH4)']
                for ipair in range(self.NPAIR):
                    #ax1.plot(self.WAVEN,kt[ipair,:],label=labels[ipair])
                    f = interpolate.interp1d(self.WAVEN,kt[ipair,:])
                    k_cia[inwave1,ipair] = f(WAVEN[inwave1])
                    #ax2.plot(WAVEN,k_cia[:,ipair])
                #plt.tight_layout()
                #plt.show()

                #Combining the CIA absorption of the different pairs (included in .cia file)
                sum1 = np.zeros(NWAVEC)
                if self.INORMAL==0:   #equilibrium hydrogen
                    sum1[:] = sum1[:] + k_cia[:,0] * qh2[ilay] * qh2[ilay] + k_cia[:,1] * qhe[ilay] * qh2[ilay]
                elif self.INORMAL==1: #'normal' hydrogen
                    sum1[:] = sum1[:] + k_cia[:,2] * qh2[ilay] * qh2[ilay] + k_cia[:,3] * qhe[ilay] * qh2[ilay]

                sum1[:] = sum1[:] + k_cia[:,4] * qh2[ilay] * qn2[ilay]
                sum1[:] = sum1[:] + k_cia[:,5] * qn2[ilay] * qch4[ilay]
                sum1[:] = sum1[:] + k_cia[:,6] * qn2[ilay] * qn2[ilay]
                sum1[:] = sum1[:] + k_cia[:,8] * qh2[ilay] * qch4[ilay]
                sum1[:] = sum1[:] + k_cia[:,7] * qch4[ilay] * qch4[ilay]

                #Look up CO2-CO2 CIA coefficients (external)
                k_co2 = co2cia(WAVEN)
                sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]

                #Look up N2-N2 NIR CIA coefficients


                #Look up N2-H2 NIR CIA coefficients



                tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]


        if ISPACE==1:
            tau_cia_layer[:,:] = tau_cia_layer[isort,:]

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            for ilay in range(Layer.NLAY):
                ax1.plot(WAVEC,tau_cia_layer[:,ilay])
            ax1.grid()
            plt.tight_layout()
            plt.show()

        return tau_cia_layer


###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

Other functions interacting with the CIA class
"""


def co2cia(WAVEN):
    """
    Subroutine to return CIA absorption coefficients for CO2-CO2

    @param WAVEN: 1D array
        Wavenumber array (cm-1)
    """

    WAVEL = 1.0e4/WAVEN
    CO2CIA = np.zeros(len(WAVEN))

    #2.3 micron window. Assume de Bergh 1995 a = 4e-8 cm-1/amagat^2
    iin = np.where((WAVEL>=2.15) & (WAVEL<=2.55))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 4.0e-8

    #1.73 micron window. Assume mean a = 6e-9 cm-1/amagat^2
    iin = np.where((WAVEL>=1.7) & (WAVEL<=1.76))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 6.0e-9

    #1.28 micron window. Update from Federova et al. (2014) to
    #aco2 = 1.5e-9 cm-1/amagat^2
    iin = np.where((WAVEL>=1.25) & (WAVEL<=1.35))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 1.5e-9

    #1.18 micron window. Assume a mean a = 1.5e-9 cm-1/amagat^2
    #if(xl.ge.1.05.and.xl.le.1.35)aco2 = 1.5e-9
    #Update from Federova et al. (2014)
    iin = np.where((WAVEL>=1.125) & (WAVEL<=1.225))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 0.5*(0.31+0.79)*1e-9

    #1.10 micron window. Update from Federova et al. (2014)
    iin = np.where((WAVEL>=1.06) & (WAVEL<=1.125))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 0.5*(0.29+0.67)*1e-9

    return CO2CIA