#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:56:22 2021

@author: jingxuanyang

Cloudy Atmosphere Class.
"""
import numpy as np
import matplotlib.pyplot as plt
from .Atmosphere_0 import Atmosphere_0

class Atmosphere_1(Atmosphere_0):
    """
    Atmosphere with aerosols.
    """
    def __init__(self, runname='wasp43b', NP=10, NVMR=6, ID=[0,0,0,0,0,0],
                ISO=[0,0,0,0,0,0], LATITUDE=0.0, IPLANET=1, AMFORM=1,
                NDUST=1, FLAGC=False, NLOCATIONS=1):
        Atmosphere_0.__init__(self, runname=runname, NP=NP, NVMR=NVMR, ID=ID,
                ISO=ISO, LATITUDE=LATITUDE, IPLANET=IPLANET, AMFORM=AMFORM, NLOCATIONS=NLOCATIONS)
        """
        See superclass Atmosphere_0 for base class properties.

        Set up an atmosphere with NDUST types of aerosols.
        When aerosols are defined, need to know if the cloud is in the form of
        a uniform thin haze or in thicker clouds covering a certain fraction
        of the mean area.
        If FLAGC == True, then we have fractional cloud coverage.

        Inputs
        ------
        @param NDUST: int
            Number of types of aerosols to be defined.
        @param FLAGC: bool
            FLAGC=1 if variable fractional cloud cover profile is defined

        Attributes
        ----------
        @param H: 1D array
            Height in m of each points above reference planetary radius.
        @param DUST: 2D array
            DUST[i,j] is the concentration of aerosol type j at
            vertical point i.
        @param FRAC: 1D array
            FRAC[I] is the fractional coverage at level I.
        @param ICLOUD: 2D array
            Used when FLAGC==True.
            If ICLOUD[I,J] is set to 1, then aerosol type J contributes to the
            broken cloud at level I, which has fractional cloud cover of FRAC[I].
            If ICLOUD[I,J] is set to 0, then aerosol J is treated as being part
            of a uniform haze.

        Methods
        -------
        Atmosphere_1.edit_DUST
        Atmosphere_1.edit_FRAC
        Atmosphere_1.edit_ICLOUD
        Atmosphere_1.write_to_file
        Atmosphere_1.check
        """
        assert type(NDUST) == int and NDUST>=0
        self.NDUST = NDUST
        self.FLAGC = FLAGC

        self.DUST = None
        self.FRAC = None
        self.ICLOUD = None

    def edit_DUST(self, DUST_array):
        """
        Edit the Aerosol profile.
        @param DUST_array: 1D array
            Aerosol content of the vertical points.
        """
        DUST_array = np.array(DUST_array)
        try:
            assert DUST_array.shape == (self.NP, self.NDUST),\
                'DUST should be NP by NDUST.'
        except:
            assert DUST_array.shape == (self.NP,) and self.NDUST==1,\
                'DUST should be NP by NDUST.'
        assert (DUST_array>=0).all() == True,\
            'DUST should be non-negative'
        self.DUST = DUST_array

    def edit_FRAC(self, FRAC_array):
        assert self.FLAGC,\
            "Not a fractional cloud coverage profile"
        FRAC_array = np.array(FRAC_array)
        assert len(FRAC_array) == self.NP,\
            'FRAC should have NP elements'
        assert (FRAC_array>=0).all() == True,\
            'FRAC should be non-negative'
        self.FRAC = FRAC_array

    def edit_ICLOUD(self, ICLOUD_array):
        assert self.FLAGC,\
            "Not a fractional cloud coverage profile"
        ICLOUD_array = np.array(ICLOUD_array)
        try:
            assert ICLOUD_array.shape == (self.NP, self.NDUST),\
                'ICLOUD should be NP by NDUST.'
        except:
            assert ICLOUD_array.shape == (self.NP,) and self.NDUST==1,\
                'ICLOUD should be NP by NDUST.'
        if self.NDUST == 1:
            for i in range(self.NP):
                assert ICLOUD_array[i]==1 or ICLOUD_array[i]==0,\
                    'ICLOUD should have either 1 or 0 as elements'
        else:
            for i in range(self.NP):
                for j in range(self.NDUST):
                    assert ICLOUD_array[i,j]==1 or ICLOUD_array[i,j]==0,\
                        'ICLOUD should have either 1 or 0 as elements'
        self.ICLOUD = ICLOUD_array

    def write_to_file(self):
        self.check()
        Atmosphere_0.write_to_file(self)
        """
        Write current aerosol profile to a aerosol.prf file in Nemesis format.
        """
        assert self.NDUST > 0, "No aerosol profile defined."
        f = open('aerosol.prf','w')
        f.write('#aerosol.prf\n')
        f.write('{:<15} {:<15}'.format(self.NP, self.NDUST))
        for i in range(self.NP):
            f.write('\n{:<15.3f} '.format(self.H[i]*1e-3))
            if self.NDUST >= 1:
                for j in range(self.NDUST):
                    f.write('{:<15.3E} '.format(self.DUST[i][j]))    #particles per g of atm
            else:
                f.write('{:<15.3E}'.format(self.DUST[i]))
        f.close()
        # write floud.prf here

    def read_aerosol(self):
        """
        Read the aerosol profiles from an aerosol.ref file
        """

        #Opening file
        f = open('aerosol.ref','r')

        #Reading header
        s = f.readline().split()

        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
        npro = tmp[0]
        naero = tmp[1]

        #Reading data
        height = np.zeros([npro])
        aerodens = np.zeros([npro,naero])
        for i in range(npro):
            tmp = np.fromfile(f,sep=' ',count=naero+1,dtype='float')
            height[i] = tmp[0]
            for j in range(naero):
                aerodens[i,j] = tmp[j+1]

        #Storing the results into the atmospheric class
        if self.NP==None:
            self.NP = npro
        else:
            if self.NP!=npro:
                sys.exit('Number of altitude points in aerosol.ref must be equal to NP')
    
        self.NP = npro
        self.NDUST = naero
        self.edit_H(height*1.0e3)   #m
        self.edit_DUST(aerodens)    #particles m-3

    def write_aerosol(self):
        """
        Write current aerosol profile to a aerosol.ref file in Nemesis format.
        """

        f = open('aerosol.ref','w')
        f.write('#aerosol.ref\n')
        f.write('{:<15} {:<15}'.format(self.NP, self.NDUST))
        for i in range(self.NP):
            f.write('\n{:<15.3f} '.format(self.H[i]*1e-3))
            if self.NDUST >= 1:
                for j in range(self.NDUST):
                    f.write('{:<15.3E} '.format(self.DUST[i][j]))    #particles per cm-3
            else:
                f.write('{:<15.3E}'.format(self.DUST[i]))
        f.close()

    def plot_Dust(self):
        """
        Make a summary plot of the current dust profiles
        """

        fig,ax1 = plt.subplots(1,1,figsize=(3,4))

        for i in range(self.NDUST):
            ax1.plot(self.DUST[:,i],self.H/1.0e3)
        ax1.grid()
        ax1.set_xlabel('Aerosol density (particles m$^{-3}$)')
        ax1.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()


atm1 = Atmosphere_1()
NP = 10
for i in range(1):
    #atm1 = Atmosphere_0()
    #atm1.write_to_file()

    # create profiles from external models
    H = np.linspace(0,9000,NP)
    P = np.logspace(6,1,NP)
    T = np.linspace(40,20,NP)**2
    VMR = np.array([np.ones(NP)*1.6e-6,
                          np.ones(NP)*1.6e-6,
                          np.ones(NP)*1.6e-6,
                          np.ones(NP)*1.6e-6,
                          np.ones(NP)*1.6e-6,
                          np.ones(NP)*1.6e-6,]).T
    DUST = np.linspace(1e10,1e8,NP)
    # add profiles to atmosphere
    atm1.edit_H(H)
    atm1.edit_P(P)
    atm1.edit_T(T)
    atm1.edit_VMR(VMR)
    atm1.edit_DUST(DUST)
    atm1.check()
    # atm1.write_to_file()
