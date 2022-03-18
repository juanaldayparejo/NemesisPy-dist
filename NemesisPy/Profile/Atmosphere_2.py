#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:57:22 2021

@author: jingxuanyang

Atmosphere Class with para-H2 profile defined.
"""
import numpy as np
from .Atmosphere_0 import Atmosphere_0

class Atmosphere_2(Atmosphere_0):

    def __init__(self, runname='wasp43b', NP=10, NVMR=6, ID=[0,0,0,0,0,0],
                ISO=[0,0,0,0,0,0], LATITUDE=0.0, IPLANET=1, AMFORM=1,
                NDUST=1, FLAGC=False):
        Atmosphere_0.__init__(self, runname=runname, NP=NP, NVMR=NVMR, ID=ID,
                ISO=ISO, LATITUDE=LATITUDE, IPLANET=IPLANET, AMFORM=AMFORM, NLOCATIONS=NLOCATIONS)
        """
        See superclass Atmosphere_0 for base class properties.

        Set up an atmosphere with a para-H2 profile.

        Inputs
        ------

        Attributes
        ----------
        @param PARAH2: 1D array
            Para-H2 profile

        Methods
        -------
        Atmosphere_2.edit_PARAH2
        Atmosphere_2.write_to_file
        Atmosphere_2.check
        """

        self.PARAH2 = None

    def edit_PARAH2(self, PARAH2_array):
        PARAH2_array = np.array(PARAH2_array)
        self.PARAH2 = PARAH2_array

    def write_to_file(self):
        Atmosphere_0.write_to_file(self)
        """
        Write current aerosol profile to a aerosol.prf file in Nemesis format.
        """
        f = open('parah2.prf','w')
        f.write('{:<15}\n'.format(self.NP))
        f.f.write('{:<15}  {:<15}\n'.format(self.H, self.PARAH2))
        f.close()

    def check(self):
        Atmosphere_0.check(self)