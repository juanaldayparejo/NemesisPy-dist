from NemesisPy.Profile import *
from NemesisPy.Models import *
from NemesisPy.Data import *
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

    def __init__(self, GASGIANT=False, LOWBC=1, GALB=1.0, NEM=2, VEM=[0.,1000.]):

        """
        Inputs
        ------
        @param GASGIANT: log,
            Flag indicating whether the planet has surface (GASGIANT=False) or not (GASGIANT=True)
        @param LOWBC: int,
            Flag indicating the lower boundary condition, which can be Thermal (0) or Lambertian (1)    
        @param GALB: int,
            Ground albedo
        @param NEM: int,
            Number of spectral points defining the emissivity of the surface   
        @param VEM: 1D array
            Wavelengths at which the surface emissivity is defined 
        
        Attributes
        ----------
        @attribute TSURF: real
            Surface temperature (K)
        @attribute EMISSIVITY: 1D array
            Surface emissitivity 

        Methods
        -------
        Surface_0.edit_EMISSIVITY
        Surface_0.read_sur
        """

        #Input parameters
        self.GASGIANT = GASGIANT
        self.LOWBC = LOWBC
        self.GALB = GALB
        self.NEM = NEM
        self.VEM = VEM

        # Input the following profiles using the edit_ methods.
        self.TSURF = None 
        self.EMISSIVITY = None 

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

    def read_sur(self, runname,MakePlot=False):
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

        if MakePlot==True:
            fig,ax1=plt.subplots(1,1,figsize=(8,3))
            ax1.plot(self.VEM,self.EMISSIVITY)
            plt.tight_layout()
            plt.show()
