#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
#from .layer_split import layer_split
#from .layer_average import layer_average
#from .layer_averageg import layer_averageg
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

#########################################################################################
# END OF LAYER CLASS

# USEFUL FUNCTIONS FOR THE LAYERING CLASS
#########################################################################################

def interp(X_data, Y_data, X, ITYPE=1):
    """
    Routine for 1D interpolation using the SciPy library.

    Inputs
    ------
    @param X_data: 1D array

    @param Y_data: 1D array

    @param X: real

    @param ITYPE: int
        1=linear interpolation
        2=quadratic spline interpolation
        3=cubic spline interpolation
    """
    
    from scipy.interpolate import interp1d
    
    if ITYPE == 1:
        interp = interp1d
        f = interp1d(X_data, Y_data, kind='linear', fill_value='extrapolate')
        Y = f(X)

    elif ITYPE == 2:
        interp = interp1d
        f = interp(X_data, Y_data, kind='quadratic', fill_value='extrapolate')
        Y = f(X)

    elif ITYPE == 3:
        interp = interp1d
        f = interp(X_data, Y_data, kind='cubic', fill_value='extrapolate')
        Y = f(X)

    return Y

def interpg(X_data, Y_data, X):
    """
    Routine for 1D interpolation

    Inputs
    ------
    @param X_data: 1D array

    @param Y_data: 1D array

    @param X: real

    @param ITYPE: int
        1=linear interpolation
    """

    from NemesisPy import find_nearest

    NX = len(X)
    Y = np.zeros(NX)
    J = np.zeros(NX,dtype='int32')
    F = np.zeros(NX)
    for IX in range(NX):

        j = 0
        while X_data[j]<=X[IX]:
            j = j + 1
            if j==len(X_data):
                j = len(X_data) - 1
                break
        
        if j==0:
            j = 1
        J[IX] = j - 1
        F[IX] = (X[IX]-X_data[j-1])/(X_data[j]-X_data[j-1])
        Y[IX] = (1.0-F[IX])*Y_data[j-1] + F[IX]*Y_data[j]

    return Y,J,F

def layer_average(RADIUS, H, P, T, ID, VMR, DUST, BASEH, BASEP,
                  LAYANG=0.0, LAYINT=0, LAYHT=0.0, NINT=101, AMFORM=1):
    """
    Calculates average layer properties.
    Takes an atmosphere profile and a layering shceme specified by
    layer base altitudes and pressures and returns averaged height,
    pressure, temperature, VMR for each layer.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Input profile heights
    @param P: 1D array
        Input profile pressures
    @param T: 1D array
        Input profile temperatures
    @param ID: 1D array
        Gas identifiers.
    @param VMR: 2D array
        VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
        the column j corresponds to the gas with RADTRANS ID ID[j].
    @param DUST: 2D array
        DUST[i,j] is dust density of dust popoulation j at vertical point i  (particles/m3)
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
    @param LAYINT: int
        Layer integration scheme
        0 = use properties at mid path
        1 = use absorber amount weighted average values
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NINT: int
        Number of integration points to be used if LAYINT=1.
    @param AMFORM: int,
        Flag indicating how the molecular weight must be calculated:
        0 - The mean molecular weight of the atmosphere is passed in XMOLWT
        1 - The mean molecular weight of each layer is calculated (atmosphere previously adjusted to have sum(VMR)=1.0)
        2 - The mean molecular weight of each layer is calculated (VMRs do not necessarily add up to 1.0)

    Returns
    -------
    @param HEIGHT: 1D array
        Representative height for each layer
    @param PRESS: 1D array
        Representative pressure for each layer
    @param TEMP: 1D array
        Representative pressure for each layer
    @param TOTAM: 1D array
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        number of molecules per area.
    @param AMOUNT: 1D array
        Representative absorber amounts of each gas at each layer.
        AMOUNT[I,J] is the representative number of gas J molecules
        in layer I in the form of number of molecules per area.
    @param PP: 1D array
        Representative partial pressure for each gas at each layer.
        PP[I,J] is the representative partial pressure of gas J in layer I.
    @param DELH: 1D array
        Layer thicnkness.
    @param BASET: 1D array
        Layer base temperature.
    @param LAYSF: 1D array
        Layer scaling factor.
    """

    from scipy.integrate import simps
    from NemesisPy.Data.ref_data import Calc_mmw

    k_B = 1.38065e-23

    # Calculate layer geometric properties
    NLAY = len(BASEH)
    NPRO = len(H)
    DELH = np.zeros(NLAY)           # layer thickness
    DELH[0:-1] = (BASEH[1:]-BASEH[:-1])
    DELH[-1] = (H[-1]-BASEH[-1])

    sin = np.sin(LAYANG*np.pi/180)  # sin(viewing angle)
    cos = np.cos(LAYANG*np.pi/180)  # cos(viewing angle)

    z0 = RADIUS + LAYHT    # distance from planet centre to lowest layer base
    zmax = RADIUS+H[-1]    # maximum height defined in the profile

    SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
    BASES = np.sqrt((RADIUS+BASEH)**2-(z0*sin)**2)-z0*cos # Path lengths at base of layer
    DELS = np.zeros(NLAY)           # path length in each layer
    DELS[0:-1] = (BASES[1:]-BASES[:-1])
    DELS[-1] = (SMAX-BASES[-1])
    LAYSF = DELS/DELH               # Layer Scaling Factor
    BASET = interp(H,T,BASEH)       # layer base temperature

    # Note number of: profile points, layers, VMRs, aerosol types, flags
    if VMR.ndim == 1:
        NVMR = 1
    else:
        NVMR = len(VMR[0])

    if DUST.ndim == 1:
        NDUST = 1
    else:
        NDUST = len(DUST[0])

    # HEIGHT = average layer height
    HEIGHT = np.zeros(NLAY)
    # PRESS = average layer pressure
    PRESS  = np.zeros(NLAY)
    # TEMP = average layer temperature
    TEMP   = np.zeros(NLAY)
    # TOTAM = total no. of molecules/aera
    TOTAM  = np.zeros(NLAY)
    # DUDS = no. of molecules per area per distance
    DUDS   = np.zeros(NLAY)
    # AMOUNT = no. of molecules/aera for each gas (molecules/m2)
    AMOUNT = np.zeros((NLAY, NVMR))
    # PP = gas partial pressures
    PP     = np.zeros((NLAY, NVMR))
    # MOLWT = mean molecular weight
    MOLWT  = np.zeros(NLAY)
    # CONT = no. of particles/area for each dust population (particles/m2)
    CONT = np.zeros((NLAY,NDUST))

    # Calculate average properties depending on intergration type
    if LAYINT == 0:
        # use layer properties at half path length in each layer
        S = np.zeros(NLAY)
        S[:-1] = (BASES[1:] + BASES[:-1])/2
        S[-1] = (SMAX+BASES[-1])/2
        # Derive other properties from S
        HEIGHT = np.sqrt(S**2+z0**2+2*S*z0*cos) - RADIUS
        PRESS = interp(H,P,HEIGHT)
        TEMP = interp(H,T,HEIGHT)

        # Ideal gas law: N/(Area*Path_length) = P/(k_B*T)
        DUDS = PRESS/(k_B*TEMP)
        TOTAM = DUDS*DELS
        # Use the volume mixing ratio information
        if VMR.ndim > 1:
            AMOUNT = np.zeros((NLAY, NVMR))
            for J in range(NVMR):
                AMOUNT[:,J] = interp(H, VMR[:,J], HEIGHT)
            PP = (AMOUNT.T * PRESS).T
            AMOUNT = (AMOUNT.T * TOTAM).T
        else:
            AMOUNT = interp(H, VMR, HEIGHT)
            PP = AMOUNT * PRESS
            AMOUNT = AMOUNT * TOTAM
            if AMFORM==0:
                sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
            else:
                for I in range(NLAY):
                    MOLWT[I] = Calc_mmw(VMR[I], ID)

        #Use the dust density information
        if DUST.ndim > 1:
            for J in range(NDUST):
                DD = interp(H, DUST[:,J],HEIGHT)  
                CONT[:,J] = DD * DELS

        else:
            DD = interp(H, DUST,HEIGHT)  
            CONT = DD * DELS

    elif LAYINT == 1:
        # Curtis-Godson equivalent path for a gas with constant mixing ratio
        for I in range(NLAY):
            S0 = BASES[I]
            if I < NLAY-1:
                S1 = BASES[I+1]
            else:
                S1 = SMAX
            # sub-divide each layer into NINT layers
            S = np.linspace(S0, S1, NINT)
            h = np.sqrt(S**2+z0**2+2*S*z0*cos)-RADIUS
            p = interp(H,P,h)
            temp = interp(H,T,h)
            duds = p/(k_B*temp)

            amount = np.zeros((NINT, NVMR))
            molwt = np.zeros(NINT)

            TOTAM[I] = simps(duds,S)
            HEIGHT[I]  = simps(h*duds,S)/TOTAM[I]
            PRESS[I] = simps(p*duds,S)/TOTAM[I]
            TEMP[I]  = simps(temp*duds,S)/TOTAM[I]

            if VMR.ndim > 1:
                amount = np.zeros((NINT, NVMR))
                for J in range(NVMR):
                    amount[:,J] = interp(H, VMR[:,J], h)
                    AMOUNT[I,J] = simps(amount[:,J]*duds,S)
                pp = (amount.T * p).T     # gas partial pressures
                for J in range(NVMR):
                    PP[I, J] = simps(pp[:,J]*duds,S)/TOTAM[I]
                
                if AMFORM==0:
                    sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
                else:
                    for K in range(NINT):
                        molwt[K] = Calc_mmw(amount[K,:], ID)
                    MOLWT[I] = simps(molwt*duds,S)/TOTAM[I]
            else:
                amount = interp(H, VMR, h)
                pp = amount * p
                AMOUNT[I] = simps(amount*duds,S)
                PP[I] = simps(pp*duds,S)/TOTAM[I]

                if AMFORM==0:
                    sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
                else:
                    for K in range(NINT):
                        molwt[K] = Calc_mmw(amount[K], ID)
                    MOLWT[I] = simps(molwt*duds,S)/TOTAM[I]

            if DUST.ndim > 1:
                dd = np.zeros((NINT,NDUST))
                for J in range(NDUST):
                    dd[:,J] = interp(H, DUST[:,J], h)
                    CONT[I,J] = simps(dd[:,J],S)
            else:
                dd = interp(H, DUST, h) 
                CONT[I] = simps(dd,S)

    # Scale back to vertical layers
    TOTAM = TOTAM / LAYSF
    if VMR.ndim > 1:
        AMOUNT = (AMOUNT.T * LAYSF**-1 ).T
    else:
        AMOUNT = AMOUNT/LAYSF

    if DUST.ndim > 1:
        CONT = (CONT.T * LAYSF**-1 ).T
    else:
        CONT = CONT/LAYSF

    return HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,DELH,BASET,LAYSF

def layer_averageg(RADIUS, H, P, T, ID, VMR, DUST, BASEH, BASEP,
                  LAYANG=0.0, LAYINT=0, LAYHT=0.0, NINT=101, AMFORM=1):
    """
    Calculates average layer properties.
    Takes an atmosphere profile and a layering shceme specified by
    layer base altitudes and pressures and returns averaged height,
    pressure, temperature, VMR for each layer.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Input profile heights
    @param P: 1D array
        Input profile pressures
    @param T: 1D array
        Input profile temperatures
    @param ID: 1D array
        Gas identifiers.
    @param VMR: 2D array
        VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
        the column j corresponds to the gas with RADTRANS ID ID[j].
    @param DUST: 2D array
        DUST[i,j] is dust density of dust popoulation j at vertical point i  (particles/m3)
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
    @param LAYINT: int
        Layer integration scheme
        0 = use properties at mid path
        1 = use absorber amount weighted average values
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NINT: int
        Number of integration points to be used if LAYINT=1.
    @param AMFORM: int,
        Flag indicating how the molecular weight must be calculated:
        0 - The mean molecular weight of the atmosphere is passed in XMOLWT
        1 - The mean molecular weight of each layer is calculated (atmosphere previously adjusted to have sum(VMR)=1.0)
        2 - The mean molecular weight of each layer is calculated (VMRs do not necessarily add up to 1.0)

    Returns
    -------
    @param HEIGHT: 1D array
        Representative height for each layer
    @param PRESS: 1D array
        Representative pressure for each layer
    @param TEMP: 1D array
        Representative pressure for each layer
    @param TOTAM: 1D array
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        number of molecules per area.
    @param AMOUNT: 1D array
        Representative absorber amounts of each gas at each layer.
        AMOUNT[I,J] is the representative number of gas J molecules
        in layer I in the form of number of molecules per area.
    @param PP: 1D array
        Representative partial pressure for each gas at each layer.
        PP[I,J] is the representative partial pressure of gas J in layer I.
    @param DELH: 1D array
        Layer thicnkness.
    @param BASET: 1D array
        Layer base temperature.
    @param LAYSF: 1D array
        Layer scaling factor.
    @param DTE: 2D array
        Matrix relating the temperature in each layer (TEMP) to the input temperature profile (T)
    @param DAM: 2D array
        Matrix relating the absorber amounts in each layer (AMOUNT) to the input VMR profile (VMR)
    @param DCO: 2D array
        Matrix relating the dust amounts in each layer (CONT) to the input dust profile (DUST)
    """

    from scipy.integrate import simps
    from NemesisPy.Data.ref_data import Calc_mmw

    k_B = 1.38065e-23

    # Calculate layer geometric properties
    NLAY = len(BASEH)
    NPRO = len(H)
    DELH = np.zeros(NLAY)           # layer thickness
    DELH[0:-1] = (BASEH[1:]-BASEH[:-1])
    DELH[-1] = (H[-1]-BASEH[-1])

    sin = np.sin(LAYANG*np.pi/180)  # sin(viewing angle)
    cos = np.cos(LAYANG*np.pi/180)  # cos(viewing angle)

    z0 = RADIUS + LAYHT    # distance from planet centre to lowest layer base
    zmax = RADIUS+H[-1]    # maximum height defined in the profile

    SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
    BASES = np.sqrt((RADIUS+BASEH)**2-(z0*sin)**2)-z0*cos # Path lengths at base of layer
    DELS = np.zeros(NLAY)           # path length in each layer
    DELS[0:-1] = (BASES[1:]-BASES[:-1])
    DELS[-1] = (SMAX-BASES[-1])
    LAYSF = DELS/DELH               # Layer Scaling Factor
    BASET = interp(H,T,BASEH)       # layer base temperature

    # Note number of: profile points, layers, VMRs, aerosol types, flags
    if VMR.ndim == 1:
        NVMR = 1
    else:
        NVMR = len(VMR[0])

    if DUST.ndim == 1:
        NDUST = 1
    else:
        NDUST = len(DUST[0])

    # HEIGHT = average layer height
    HEIGHT = np.zeros(NLAY)
    # PRESS = average layer pressure
    PRESS  = np.zeros(NLAY)
    # TEMP = average layer temperature
    TEMP   = np.zeros(NLAY)
    # TOTAM = total no. of molecules/aera
    TOTAM  = np.zeros(NLAY)
    # DUDS = no. of molecules per area per distance
    DUDS   = np.zeros(NLAY)
    # AMOUNT = no. of molecules/aera for each gas (molecules/m2)
    AMOUNT = np.zeros((NLAY, NVMR))
    # PP = gas partial pressures
    PP     = np.zeros((NLAY, NVMR))
    # MOLWT = mean molecular weight
    MOLWT  = np.zeros(NLAY)
    # CONT = no. of particles/area for each dust population (particles/m2)
    CONT = np.zeros((NLAY,NDUST))
    #DTE = matrix to relate the temperature in each layer (TEMP) to the temperature in the input profiles (T)
    DTE = np.zeros((NLAY, NPRO))
    #DCO = matrix to relate the dust abundance in each layer (CONT) to the dust abundance in the input profiles (DUST)
    DCO = np.zeros((NLAY, NPRO))
    #DAM = matrix to relate the gaseous abundance in each layer (AMOUNT) to the gas VMR in the input profiles (VMR)
    DAM = np.zeros((NLAY, NPRO))

    #Defining the weights for the integration with the Simpson's rule
    w = np.ones(NINT) * 4.
    w[::2] = 2.
    w[0] = 1.0
    w[NINT-1] = 1.0

    # Calculate average properties depending on intergration type
    if LAYINT == 0:
        # use layer properties at half path length in each layer
        S = np.zeros(NLAY)
        S[:-1] = (BASES[1:] + BASES[:-1])/2
        S[-1] = (SMAX+BASES[-1])/2
        # Derive other properties from S
        HEIGHT = np.sqrt(S**2+z0**2+2*S*z0*cos) - RADIUS
        PRESS = interp(H,P,HEIGHT)
        #TEMP = interp(H,T,HEIGHT)
        TEMP,J,F = interpg(H,T,HEIGHT)
        for ilay in range(NLAY):
            DTE[ilay,J[ilay]] = DTE[ilay,J[ilay]] + (1.0-F[ilay])
            DTE[ilay,J[ilay]+1] = DTE[ilay,J[ilay]+1] + (F[ilay])

        # Ideal gas law: N/(Area*Path_length) = P/(k_B*T)
        DUDS = PRESS/(k_B*TEMP)
        TOTAM = DUDS*DELS
        # Use the volume mixing ratio information
        if VMR.ndim > 1:
            AMOUNT = np.zeros((NLAY, NVMR))
            for J in range(NVMR):
                AMOUNT[:,J],JJ,F = interpg(H, VMR[:,J], HEIGHT)
            PP = (AMOUNT.T * PRESS).T
            AMOUNT = (AMOUNT.T * TOTAM).T
        else:
            AMOUNT,JJ,F = interpg(H, VMR, HEIGHT)
            PP = AMOUNT * PRESS
            AMOUNT = AMOUNT * TOTAM
            if AMFORM==0:
                sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
            else:
                for I in range(NLAY):
                    MOLWT[I] = Calc_mmw(VMR[I], ID)

        for ilay in range(NLAY):
            DAM[ilay,JJ[ilay]] = DAM[ilay,JJ[ilay]] + (1.0-F[ilay])*TOTAM[ilay]
            DAM[ilay,JJ[ilay]+1] = DAM[ilay,JJ[ilay]+1] + (F[ilay])*TOTAM[ilay]

        #Use the dust density information
        if DUST.ndim > 1:
            for J in range(NDUST):
                DD,JJ,F = interpg(H, DUST[:,J],HEIGHT)  
                CONT[:,J] = DD * DELS

        else:
            #DD = interp(H, DUST,HEIGHT)  
            DD,JJ,F = interpg(H, DUST,HEIGHT)
            CONT = DD * DELS

        for ilay in range(NLAY):
            DCO[ilay,JJ[ilay]] = DCO[ilay,JJ[ilay]] + (1.0-F[ilay])
            DCO[ilay,JJ[ilay]+1] = DCO[ilay,JJ[ilay]+1] + (F[ilay])

    elif LAYINT == 1:
        # Curtis-Godson equivalent path for a gas with constant mixing ratio
        for I in range(NLAY):
            S0 = BASES[I]
            if I < NLAY-1:
                S1 = BASES[I+1]
            else:
                S1 = SMAX
            # sub-divide each layer into NINT layers
            S = np.linspace(S0, S1, NINT)
            h = np.sqrt(S**2+z0**2+2*S*z0*cos)-RADIUS
            p = interp(H,P,h)
            temp,JJ,F = interpg(H,T,h)
            duds = p/(k_B*temp)

            amount = np.zeros((NINT, NVMR))
            molwt = np.zeros(NINT)

            TOTAM[I] = simps(duds,S)
            HEIGHT[I]  = simps(h*duds,S)/TOTAM[I]
            PRESS[I] = simps(p*duds,S)/TOTAM[I]
            TEMP[I]  = simps(temp*duds,S)/TOTAM[I]

            for iint in range(NINT):
                DTE[I,JJ[iint]] = DTE[I,JJ[iint]] + (1.-F[iint])*w[iint]*duds[iint]
                DTE[I,JJ[iint]+1] = DTE[I,JJ[iint]+1] + (F[iint])*w[iint]*duds[iint]

            if VMR.ndim > 1:
                amount = np.zeros((NINT, NVMR))
                for J in range(NVMR):
                    amount[:,J],JJ,F = interpg(H, VMR[:,J], h)
                    AMOUNT[I,J] = simps(amount[:,J]*duds,S)
                pp = (amount.T * p).T     # gas partial pressures
                for J in range(NVMR):
                    PP[I, J] = simps(pp[:,J]*duds,S)/TOTAM[I]
                
                if AMFORM==0:
                    sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
                else:
                    for K in range(NINT):
                        molwt[K] = Calc_mmw(amount[K,:], ID)
                    MOLWT[I] = simps(molwt*duds,S)/TOTAM[I]
            else:
                amount,JJ,F = interpg(H, VMR, h)
                pp = amount * p
                AMOUNT[I] = simps(amount*duds,S)
                PP[I] = simps(pp*duds,S)/TOTAM[I]

                if AMFORM==0:
                    sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
                else:
                    for K in range(NINT):
                        molwt[K] = Calc_mmw(amount[K], ID)
                    MOLWT[I] = simps(molwt*duds,S)/TOTAM[I]

            for iint in range(NINT):
                DAM[I,JJ[iint]] = DAM[I,JJ[iint]] + (1.-F[iint])*duds[iint]*w[iint]
                DAM[I,JJ[iint]+1] = DAM[I,JJ[iint]+1] + (F[iint])*duds[iint]*w[iint]

            if DUST.ndim > 1:
                dd = np.zeros((NINT,NDUST))
                for J in range(NDUST):
                    dd[:,J],JJ,F = interpg(H, DUST[:,J], h)
                    CONT[I,J] = simps(dd[:,J],S)
            else:
                dd,JJ,F = interpg(H, DUST, h) 
                CONT[I] = simps(dd,S)

            for iint in range(NINT):
                DCO[I,JJ[iint]] = DCO[I,JJ[iint]] + (1.-F[iint])*w[iint]
                DCO[I,JJ[iint]+1] = DCO[I,JJ[iint]+1] + (F[iint])*w[iint]

    #Finishing the integration for the matrices
    for IPRO in range(NPRO):
        DTE[:,IPRO] = DTE[:,IPRO] * DELS / 100. / 3. / TOTAM
        DAM[:,IPRO] = DAM[:,IPRO] * DELS / 100. / 3.
        DCO[:,IPRO] = DCO[:,IPRO] * DELS / 100. / 3.

    # Scale back to vertical layers
    TOTAM = TOTAM / LAYSF
    if VMR.ndim > 1:
        AMOUNT = (AMOUNT.T * LAYSF**-1 ).T
    else:
        AMOUNT = AMOUNT/LAYSF

    if DUST.ndim > 1:
        CONT = (CONT.T * LAYSF**-1 ).T
    else:
        CONT = CONT/LAYSF

    for IPRO in range(NPRO):
        DAM[:,IPRO] = DAM[:,IPRO] / LAYSF
        DCO[:,IPRO] = DCO[:,IPRO] / LAYSF

    return HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,DELH,BASET,LAYSF,DTE,DAM,DCO

def layer_split(RADIUS, H, P, LAYANG=0.0, LAYHT=0.0, NLAY=20,
        LAYTYP=1, INTERTYP=1, H_base=None, P_base=None):
    """
    Splits an atmosphere into NLAY layers.
    Takes a set of altitudes H with corresponding pressures P and returns
    the altitudes and pressures of the base of the layers.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Heights at which the atmosphere profile is specified.
        (At altitude H[i] the pressure is P[i].)
    @param P: 1D array
        Pressures at which the atmosphere profile is specified.
        (At pressure P[i] the altitude is H[i].)
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
        Default 0.0 (nadir geometry). Only needed for layer type 3.
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NLAY: int
        Number of layers to split the atmosphere into. Default 20.
    @param LAYTYP: int
        Integer specifying how to split up the layers. Default 1.
        0 = by equal changes in pressure
        1 = by equal changes in log pressure
        2 = by equal changes in height
        3 = by equal changes in path length at LAYANG
        4 = layer base pressure levels specified by P_base
        5 = layer base height levels specified by H_base
        Note 4 and 5 force NLAY = len(P_base) or len(H_base).
    @param H_base: 1D array
        Heights of the layer bases defined by user. Default None.
    @param P_base: 1D array
        Pressures of the layer bases defined by user. Default None.
    @param INTERTYP: int
        Interger specifying interpolation scheme.  Default 1.
        1=linear, 2=quadratic spline, 3=cubic spline

    Returns
    -------
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    """

    if LAYHT<H[0]:
        print('Warning from layer_split() :: LAYHT < H(0). Resetting LAYHT')
        LAYHT = H[0]

    #assert (LAYHT>=H[0]) and (LAYHT<H[-1]) , \
    #    'Lowest layer base height LAYHT not contained in atmospheric profile'
    #assert not (H_base and P_base), \
    #    'Cannot input both layer base heights and base pressures'

    if LAYTYP == 0: # split by equal pressure intervals
        PBOT = interp(H,P,LAYHT,INTERTYP)  # pressure at base of lowest layer
        BASEP = np.linspace(PBOT,P[-1],NLAY+1)[:-1]
        BASEH = interp(P,H,BASEP,INTERTYP)

    elif LAYTYP == 1: # split by equal log pressure intervals
        PBOT = interp(H,P,LAYHT,INTERTYP)  # pressure at base of lowest layer
        BASEP = np.logspace(np.log10(PBOT),np.log10(P[-1]),NLAY+1)[:-1]
        BASEH = interp(P,H,BASEP,INTERTYP)

    elif LAYTYP == 2: # split by equal height intervals
        BASEH = np.linspace(H[0]+LAYHT, H[-1], NLAY+1)[:-1]
        BASEP = interp(H,P,BASEH,INTERTYP)

    elif LAYTYP == 3: # split by equal line-of-sight path intervals
        assert LAYANG<=90 and LAYANG>=0,\
            'Zennith angle should be in [0,90]'
        sin = np.sin(LAYANG*np.pi/180)      # sin(zenith angle)
        cos = np.cos(LAYANG*np.pi/180)      # cos(zenith angle)
        z0 = RADIUS + LAYHT                 # distance from centre to lowest layer's base
        zmax = RADIUS+H[-1]                 # maximum height
        SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
        BASES = np.linspace(0, SMAX, NLAY+1)[:-1]
        BASEH = np.sqrt(BASES**2+z0**2+2*BASES*z0*cos) - RADIUS
        logBASEP = interp(H,np.log(P),BASEH,INTERTYP)
        BASEP = np.exp(logBASEP)

    elif LAYTYP == 4: # split by specifying input base pressures
        assert P_base, 'Need input layer base pressures'
        assert  (P_base[-1] >= P[-1]) and (P_base[0] <= P[0]), \
            'Input layer base pressures out of range of atmosphere profile'
        BASEP = P_base
        NLAY = len(BASEP)
        BASEH = interp(P,H,BASEP,INTERTYP)

    elif LAYTYP == 5: # split by specifying input base heights
        #NLAY,H_base = read_hlay()
        BASEH = H_base
        #NLAY = len(H_base)
        
        logBASEP = interp(H,np.log(P),BASEH,INTERTYP)
        BASEP = np.exp(logBASEP)


    else:
        raise('Layering scheme not defined')

    return BASEH, BASEP

def read_hlay():

    """
        FUNCTION NAME : read_hlay()
        
        DESCRIPTION : Read the height.lay file used to set the altitude of the base of the layers
                      in a Nemesis run
        
        INPUTS : None
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            nlay :: Number of layers in atmospheric model

        
        CALLING SEQUENCE:
        
            nlay,hbase = read_hlay()
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open('height.lay','r')

    header = f.readline().split()

    s = f.readline().split()
    nlay = int(s[0])
    hbase = np.zeros(nlay)
    for i in range(nlay):
        s = f.readline().split()
        hbase[i] = float(s[0])

    return nlay,hbase