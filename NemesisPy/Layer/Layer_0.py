#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from .layer_split import layer_split
from .layer_average import layer_average
"""
Object to store layering scheme settings and averaged properties of each layer.
"""
class Layer_0:
    def __init__(self, RADIUS=None, LAYHT=0, LAYTYP=1, LAYINT=1, NLAY=20, NINT=101,
                AMFORM=1, INTERTYP=1, H_base=None, P_base=None):
        """
        After creating a Layer object, call the method
            integrate(self, H, P, T, LAYANG, ID, VMR) to calculate
            averaged layer properties.

        Inputs
        ------
        @param RADIUS: real
            Reference planetary radius where H=0.  Usually at surface for
            terrestrial planets, or at 1 bar pressure level for gas giants.
        @param LAYHT: real
            Height of the base of the lowest layer. Default 0.0.
        @param LAYTYP: int
            Integer specifying how to split up the layers. Default 1.
            0 = by equal changes in pressure
            1 = by equal changes in log pressure
            2 = by equal changes in height
            3 = by equal changes in path length at LAYANG
            4 = layer base pressure levels specified by P_base
            5 = layer base height levels specified by H_base
            Note 4 and 5 force NLAY = len(P_base) or len(H_base).
        @param LAYINT: int
            Layer integration scheme
            0 = use properties at mid path
            1 = use absorber amount weighted average values
        @param NLAY: int
            Number of layers to split the atmosphere into. Default 20.
        @param NINT: int
            Number of integration points to be used if LAYINT=1.
        @param AMFORM: int
            Currently not used.
        @param INTERTYP: int
            Interger specifying interpolation scheme.  Default 1.
            1=linear, 2=quadratic spline, 3=cubic spline
        @param H_base: 1D array
            Heights of the layer bases defined by user. Default None.
        @param P_base: 1D array
            Pressures of the layer bases defined by user. Default None.

        Methods
        -------

        """
        self.RADIUS = RADIUS
        self.LAYTYP = LAYTYP
        self.NLAY = NLAY
        self.LAYINT = LAYINT
        self.NINT = NINT
        self.LAYHT = LAYHT
        self.AMFORM = AMFORM
        self.INTERTYP = INTERTYP
        self.H_base = H_base
        self.P_base = P_base

        # additional input for layer_split
        self.H = None
        self.P = None
        self.T = None
        self.LAYANG = None
        # output for layer_split
        self.LAYANG = None
        self.BASEH = None
        self.BASEP = None
        self.DELH = None
        # additional input for layer_average
        self.ID = None
        self.VMR = None
        self.DUST = None
        # output for layer_average
        self.HEIGHT = None
        self.PRESS = None
        self.TEMP = None
        self.TOTAM = None
        self.AMOUNT = None
        self.PP = None
        
        # cloud
        self.CONT = None


    def integrate(self, H, P, T, LAYANG, ID, VMR, DUST):
        self.layer_split(H=H, P=P, T=T, LAYANG=LAYANG)
        self.layer_average(ID=ID, VMR=VMR, DUST=DUST)
        """
        @param H: 1D array
            Input profile heights
        @param P: 1D array
            Input profile pressures
        @param T: 1D array
            Input profile temperatures
        @param LAYANG: real
            Zenith angle in degrees defined at LAYHT.
        @param ID: 1D array
            Gas identifiers.
        @param VMR: 2D array
            VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
            the column j corresponds to the gas with RADTRANS ID ID[j].
        """
        return (self.BASEH, self.BASEP, self.BASET, self.HEIGHT, self.PRESS,
                self.TEMP, self.TOTAM, self.AMOUNT, self.PP, self.CONT,
                self.LAYSF, self.DELH)

    def layer_split(self, H, P, T, LAYANG):
        self.H = H
        self.P = P
        self.T = T
        self.LAYANG = LAYANG
        # get layer base height and base pressure
        BASEH, BASEP = layer_split(RADIUS=self.RADIUS, H=H, P=P, LAYANG=LAYANG,
            LAYHT=self.LAYHT, NLAY=self.NLAY, LAYTYP=self.LAYTYP,
            INTERTYP=self.INTERTYP, H_base=self.H_base, P_base=self.P_base)
        self.BASEH = BASEH
        self.BASEP = BASEP

    def layer_average(self, ID, VMR, DUST):
        # get averaged layer properties
        HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,DELH,BASET,LAYSF\
            = layer_average(RADIUS=self.RADIUS, H=self.H, P=self.P, T=self.T,
                ID=ID, VMR=VMR, DUST=DUST, BASEH=self.BASEH, BASEP=self.BASEP,
                LAYANG=self.LAYANG, LAYINT=self.LAYINT, LAYHT=self.LAYHT,
                NINT=self.NINT)
        self.HEIGHT = HEIGHT
        self.PRESS = PRESS
        self.TEMP = TEMP
        self.TOTAM = TOTAM
        self.AMOUNT = AMOUNT
        self.PP = PP
        self.CONT = CONT
        self.DELH = DELH
        self.BASET = BASET
        self.LAYSF = LAYSF
