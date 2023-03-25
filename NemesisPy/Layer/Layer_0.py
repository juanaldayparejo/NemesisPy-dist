#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from .layer_split import layer_split
from .layer_average import layer_average
from .layer_averageg import layer_averageg
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


        @param TAURAY: 2D array (NWAVE,NLAY)
            Rayleigh scattering optical depth
        @param TAUSCAT: 2D array (NWAVE,NLAY)
            Aerosol scattering optical depth
        @param TAUCLSCAT: 3D array (NWAVE,NLAY,NDUST)
            Aerosol scattering optical depth by each aerosol type
        @param TAUDUST: 2D array (NWAVE,NLAY)
            Aerosol extinction optical depth (absorption + scattering)
        @param TAUCIA: 2D array (NWAVE,NLAY)
            CIA optical depth
        @param TAUGAS: 3D array (NWAVE,NG,NLAY)
            Gas optical depth
        @param TAUTOT: 3D array (NWAVE,NG,NLAY)
            Total optical depth



        Methods
        -------

        Layer_0.assess()
        Layer_0.write_hdf5()
        Layer_0.read_hdf5()

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

        #output for the gradients
        self.DTE = None
        self.DAM = None
        self.DCO = None

        #optical depths in each layer
        self.NWAVE = None  #Number of calculation wavelengths
        self.TAURAY = None  #(NWAVE,NLAY) Rayleigh scattering optical depth
        self.TAUSCAT = None #(NWAVE,NLAY) Aerosol scattering optical depth
        self.TAUDUST = None #(NWAVE,NLAY) Aerosol absorption + scattering optical depth
        self.TAUCIA = None  #(NWAVE,NLAY) CIA optical depth
        self.TAUGAS = None  #(NWAVE,NG,NLAY) Gas optical depth
        self.TAUTOT = None  #(NWAVE,NG,NLAY) Total optical depth


    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NLAY), np.integer) == True , \
            'NLAY must be int'
        assert self.NLAY > 0 , \
            'NLAY must be >0'
        
        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.LAYTYP), np.integer) == True , \
            'LAYTYP must be int'
        assert self.LAYTYP >= 0 , \
            'LAYTYP must be >=0'
        assert self.LAYTYP <= 5 , \
            'LAYTYP must be <=5'
        
        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.LAYINT), np.integer) == True , \
            'LAYINT must be int'
        assert self.LAYINT >= 0 , \
            'LAYINT must be >=0 and <=1'
        assert self.LAYINT <= 1 , \
            'LAYINT must be <=1 and <=1'
        
        if self.LAYTYP==4:
            assert self.P_base is not None == True , \
                'P_base must be defined if LAYTYP=4'
        
        if self.LAYTYP==5:
            assert len(self.H_base) == self.NLAY , \
                'H_base must have size (NLAY) if LAYTYP=5'


    def write_hdf5(self,runname):
        """
        Write the information about the layering of the atmosphere in the HDF5 file
        """

        import h5py

        self.assess()

        #Writing the information into the HDF5 file
        f = h5py.File(runname+'.h5','a')
        #Checking if Layer already exists
        if ('/Layer' in f)==True:
            del f['Layer']   #Deleting the Layer information that was previously written in the file

        grp = f.create_group("Layer")

        #Writing the layering type
        dset = grp.create_dataset('LAYTYP',data=self.LAYTYP)
        dset.attrs['title'] = "Layer splitting calculation type"
        if self.LAYTYP==0:
            dset.attrs['type'] = 'Split layers by equal changes in pressure'
        elif self.LAYTYP==1:
            dset.attrs['type'] = 'Split layers by equal changes in log-pressure'
        elif self.LAYTYP==2:
            dset.attrs['type'] = 'Split layers by equal changes in height'
        elif self.LAYTYP==3: 
            dset.attrs['type'] = 'Split layers by equal changes in path length (at LAYANG)'
        elif self.LAYTYP==4:
            dset.attrs['type'] = 'Split layers by defining base pressure of each layer'
        elif self.LAYTYP==5:
            dset.attrs['type'] = 'Split layers by defining base altitude of each layer'

        #Writing the layering integration type
        dset = grp.create_dataset('LAYINT',data=self.LAYINT)
        dset.attrs['title'] = "Layer integration calculation type"
        if self.LAYINT==0:
            dset.attrs['type'] = 'Layer properties calculated at mid-point between layer boundaries'
        if self.LAYINT==1:
            dset.attrs['type'] = 'Layer properties calculated by weighting in terms of atmospheric mass (i.e. Curtis-Godson path)'
    
        #Writing the number of layers
        dset = grp.create_dataset('NLAY',data=self.NLAY)
        dset.attrs['title'] = "Number of layers"

        #Writing the altitude of bottom layer
        dset = grp.create_dataset('LAYHT',data=self.LAYHT)
        dset.attrs['title'] = "Altitude at base of bottom layer"
        dset.attrs['units'] = "m"

        #Writing the base properties of the layers in special cases
        if self.LAYTYP==4:
            dset = grp.create_dataset('P_base',data=self.P_base)
            dset.attrs['title'] = "Pressure at the base of each layer"
            dset.attrs['units'] = "Pressure / Pa"
        
        if self.LAYTYP==5:
            dset = grp.create_dataset('H_base',data=self.H_base)
            dset.attrs['title'] = "Altitude at the base of each layer"
            dset.attrs['units'] = "Altitude / m"


    def read_hdf5(self,runname):
        """
        Read the Layer properties from an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','r')

        #Checking if Surface exists
        e = "/Layer" in f
        if e==False:
            sys.exit('error :: Layer is not defined in HDF5 file')
        else:

            self.NLAY = np.int32(f.get('Layer/NLAY'))
            self.LAYTYP = np.int32(f.get('Layer/LAYTYP'))
            self.LAYINT = np.int32(f.get('Layer/LAYINT'))
            self.LAYHT = np.float64(f.get('Layer/LAYHT'))
            if self.LAYTYP==4:
                self.P_base = np.array(f.get('Layer/P_base'))
            if self.LAYTYP==5:
                self.H_base = np.array(f.get('Layer/H_base'))

        f.close()

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


    def integrateg(self, H, P, T, LAYANG, ID, VMR, DUST):
        self.layer_split(H=H, P=P, T=T, LAYANG=LAYANG)
        self.layer_averageg(ID=ID, VMR=VMR, DUST=DUST)
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
                self.LAYSF, self.DELH, self.DTE, self.DAM, self.DCO)

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

    def layer_averageg(self, ID, VMR, DUST):
        # get averaged layer properties
        HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,DELH,BASET,LAYSF,DTE,DAM,DCO\
            = layer_averageg(RADIUS=self.RADIUS, H=self.H, P=self.P, T=self.T,
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
        self.DTE = DTE
        self.DAM = DAM
        self.DCO = DCO
