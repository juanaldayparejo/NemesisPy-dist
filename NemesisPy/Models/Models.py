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
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class StateVector:

    def __init__(self, NX=10, JPRE=-1, JTAN=-1, JSURF=-1, JALB=-1, JXSC=-1, JRAD=-1, JLOGG=-1, JFRAC=-1):

        """
        Inputs
        ------
        @param NX: int,
            Number of points in the state vector
        @param JPRE: int,
            Position of ref. tangent pressure in state vector (if included)
        @param JTAN: int,
            Position of tangent altitude correction in state vector (if included)
        @param JSURF: int,
            Position of surface temperature in state vector (if included)
        @param JALB: int,
            Position of start of surface albedo spectrum in state vector (if included)
        @param JXSC: int,
            Position of start of x-section spectrum in state vector (if included)
        @param JRAD: int,
            Position of radius of the planet in state vector (if included)
        @param JLOGG: int,
            Position of surface log_10(g) of planet in state vector (if included)     
         @param JFRAC: int,
            Position of fractional coverage in state vector (if included)     
        
        Attributes
        ----------
        @attribute XN: 1D array
            State vector
        @attribute SX: 2D array
            Covariance matrix of the state vector
        @attribute LX: 1D array
            Flag indicating whether the elements of the state vector are carried in log-scale
        @attribute FIX: 1D array
            Flag indicating whether the elements of the state vector must be fixed

        Methods
        -------
        StateVector.edit_XN
        StateVector.edit_LX
        StateVector.edit_FIX
        StateVector.edit_SX
        """

        #Input parameters
        self.NX = NX
        self.JPRE = JPRE
        self.JTAN = JTAN
        self.JSURF = JSURF
        self.JALB = JALB
        self.JXSC = JXSC
        self.JRAD = JRAD
        self.JLOGG = JLOGG
        self.JFRAC = JFRAC

        # Input the following profiles using the edit_ methods.
        self.XN = None # np.zeros(NX)
        self.LX = None # np.zeros(NX)
        self.FIX =  None # np.zeros(NX)
        self.SX = None # np.zeros((NX, NX))

    def edit_XN(self, XN_array):
        """
        Edit the State Vector.
        @param XN_array: 1D array
            Parameters defining the state vector
        """
        XN_array = np.array(XN_array)
        assert len(XN_array) == self.NX, 'XN should have NX elements'
        self.XN = XN_array

    def edit_LX(self, LX_array):
        """
        Edit the the flag indicating if the elements are in log-scale
        @param LX_array: 1D array
            Flag indicating whether a particular element of the state 
            vector is in log-scale (1) or not (0)
        """
        LX_array = np.array(LX_array,dtype='int32')
        assert len(LX_array) == self.NX, 'LX should have NX elements'
        self.LX = LX_array  

    def edit_FIX(self, FIX_array):
        """
        Edit the the flag indicating if the elements are to be fixed
        @param FIX_array: 1D array
            Flag indicating whether a particular element of the state 
            vector is fixed (1) or not (0)
        """
        FIX_array = np.array(FIX_array,dtype='int32')
        assert len(FIX_array) == self.NX, 'FIX should have NX elements'
        self.FIX = FIX_array 

    def edit_SX(self, SX_array):
        """
        Edit the state vector covariance matrix
        @param SX_array: 2D array
            State vector covariance matrix
        """
        SX_array = np.array(SX_array,dtype='int32')
        assert len(SX_array[:,0]) == self.NX, 'SX should have (NX,NX) elements'
        assert len(SX_array[0,:]) == self.NX, 'SX should have (NX,NX) elements'
        self.SX = SX_array 

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

Model variables Class.
"""

class Variables:

    def __init__(self, NVAR=2, NPARAM=10):

        """
        Inputs
        ------
        @param NVAR: int,
            Number of model variables to be included
        @param NPARAM: int,
            Number of extra parameters needed to implement the different models       
        
        Attributes
        ----------
        @attribute VARIDENT: 2D array
            Variable ID
        @attribute VARPARAM: 2D array
            Extra parameters needed to implement the parameterisation
        @attribute NXVAR: 1D array
            Number of points in state vector associated with each variable

        Methods
        -------
        StateVector.edit_VARIDENT
        StateVector.edit_VARPARAM
        StateVector.edit_NXVAR
        """

        #Input parameters
        self.NVAR = NVAR
        self.NPARAM = NPARAM

        # Input the following profiles using the edit_ methods.
        self.VARIDENT = None # np.zeros(NVAR,3)
        self.VARPARAM = None # np.zeros(NVAR,NPARAM)
        self.NXVAR =  None # np.zeros(NX)

    def edit_VARIDENT(self, VARIDENT_array):
        """
        Edit the Variable IDs
        @param VARIDENT_array: 2D array
            Parameter IDs defining the parameterisation
        """
        VARIDENT_array = np.array(VARIDENT_array)
        #assert len(VARIDENT_array[:,0]) == self.NVAR, 'VARIDENT should have (NVAR,3) elements'
        #assert len(VARIDENT_array[0,:]) == 3, 'VARIDENT should have (NVAR,3) elements'
        self.VARIDENT = VARIDENT_array

    def edit_VARPARAM(self, VARPARAM_array):
        """
        Edit the extra parameters needed to implement the parameterisations
        @param VARPARAM_array: 2D array
            Extra parameters defining the model
        """
        VARPARAM_array = np.array(VARPARAM_array)
        #assert len(VARPARAM_array[:,0]) == self.NVAR, 'VARPARAM should have (NVAR,NPARAM) elements'
        #assert len(VARPARAM_array[0,:]) == self.NPARAM, 'VARPARAM should have (NVAR,NPARAM) elements'
        self.VARPARAM = VARPARAM_array

    def calc_NXVAR(self, NPRO):
        """
        Calculate the array defining the number of parameters in the state 
        vector associated with each model
        @param NXVAR_array: 1D array
            Number of parameters in the state vector associated with each model
        """

        nxvar = np.zeros(self.NVAR,dtype='int32')
        for i in range(self.NVAR):

            if self.NVAR==1:
                imod = self.VARIDENT[2]
                ipar = self.VARPARAM[0]
            else:
                imod = self.VARIDENT[i,2]
                ipar = self.VARPARAM[i,0]

            if imod == -1:
                nxvar[i] = NPRO
            elif imod == 0:
                nxvar[i] = NPRO
            elif imod == 1:
                nxvar[i] = 2
            elif imod == 2:
                nxvar[i] = 1
            elif imod == 3:
                nxvar[i] = 1
            elif imod == 4:
                nxvar[i] = 3
            elif imod == 5:
                nxvar[i] = 1
            elif imod == 6:
                nxvar[i] = 2
            elif imod == 7:
                nxvar[i] = 2
            elif imod == 8:
                nxvar[i] = 3
            elif imod == 9:
                nxvar[i] = 3
            elif imod == 10:
                nxvar[i] = 4
            elif imod == 11:
                nxvar[i] = 2
            elif imod == 12:
                nxvar[i] = 3
            elif imod == 13:
                nxvar[i] = 3
            elif imod == 14:
                nxvar[i] = 3
            elif imod == 15:
                nxvar[i] = 3
            elif imod == 16:
                nxvar[i] = 4
            elif imod == 17:
                nxvar[i] = 2
            elif imod == 18:
                nxvar[i] = 2
            elif imod == 19:
                nxvar[i] = 4
            elif imod == 20:
                nxvar[i] = 2
            elif imod == 21:
                nxvar[i] = 2
            elif imod == 22:
                nxvar[i] = 5
            elif imod == 23:
                nxvar[i] = 4
            elif imod == 24:
                nxvar[i] = 3
            elif imod == 25:
                nxvar[i] = int(ipar)
            elif imod == 26:
                nxvar[i] = 4
            elif imod == 27:
                nxvar[i] = 3
            elif imod == 28:
                nxvar[i] = 1
            elif imod == 228:
                nxvar[i] = 7
            elif imod == 229:
                nxvar[i] = 7
            elif imod == 230:
                nxvar[i] = 2*int(ipar)
            elif imod == 444:
                nxvar[i] = 1 + 1 + int(ipar)
            elif imod == 666:
                nxvar[i] = 1
            elif imod == 998:
                nxvar[i] = int(ipar)
            elif imod == 999:
                nxvar[i] = 1
            else:
                sys.exit('error :: varID not included in calc_NXVAR()')

        self.NXVAR = nxvar

###############################################################################################

def modelm1(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : model0()
        
        DESCRIPTION :
        
            Function defining the model parameterisation -1 in NEMESIS.
            In this model, the aerosol profiles is modelled as a continuous profile in units
            of particles per cm3. Note that typical units of aerosol profiles in NEMESIS
            are in particles per gram of atmosphere
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xprof(npro) :: Atmospheric aerosol profile in particles/cm3
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = modelm1(atm,ipar,xprof)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npro = len(xprof)
    if npro!=atm.NP:
        sys.exit('error in model 0 :: Number of levels in atmosphere does not match and profile')

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([npro,npar,npro])

    if ipar<atm.NVMR:  #Gas VMR
        sys.exit('error :: Model -1 is just compatible with aerosol populations')
    elif ipar==atm.NVMR: #Temperature
        sys.exit('error :: Model -1 is just compatible with aerosol populations')
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        x1 = np.exp(xprof)
        if jtmp<atm.NDUST:
            rho = atm.calc_rho(molwt)  #kg/m3
            rho = rho / 1.0e3 #g/cm3
            atm.DUST[:,jtmp] = x1 / rho
        elif jtmp==atm.NDUST:
            sys.exit('error :: Model -1 is just compatible with aerosol populations')
        elif jtmp==atm.NDUST+1:
            sys.exit('error :: Model -1 is just compatible with aerosol populations')
    
    for j in range(npro):
        xmap[0:npro,ipar,j] = x1[:] / rho[:]
        

    if MakePlot==True:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

        for i in range(atm.NDUST):
            ax1.semilogx(atm.DUST[:,i]*rho,atm.H/1000.)
            ax2.semilogx(atm.DUST[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Aerosol density (particles per cm$^{-3}$)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Aerosol density (particles per gram of atm)')
        ax2.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model0(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : model0()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 0 in NEMESIS.
            In this model, the atmospheric parameters are modelled as continuous profiles
            in which each element of the state vector corresponds to the atmospheric profile 
            at each altitude level
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xprof(npro) :: Atmospheric profile
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model0(atm,ipar,xprof)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npro = len(xprof)
    if npro!=atm.NP:
        sys.exit('error in model 0 :: Number of levels in atmosphere does not match and profile')

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([npro,npar,npro])

    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        x1 = np.exp(xprof)
        atm.VMR[:,jvmr] = x1
    elif ipar==atm.NVMR: #Temperature
        x1 = xprof
        atm.T[:] = x1
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        x1 = np.exp(xprof)
        if jtmp<atm.NDUST:
            atm.DUST[:,jtmp] = x1
        elif jtmp==atm.NDUST:
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            atm.FRAC = x1
    
    for j in range(npro):
        xmap[0:npro,ipar,j] = x1[:]
        

    if MakePlot==True:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

        ax1.semilogx(atm.P/101325.,atm.H/1000.)
        ax2.plot(atm.T,atm.H/1000.)
        for i in range(atm.NVMR):
            ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Altitude (km)')
        ax3.set_xlabel('Volume mixing ratio')
        ax3.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model2(atm,ipar,scf,MakePlot=False):
    
    """
        FUNCTION NAME : model2()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 2 in NEMESIS.
            In this model, the atmospheric parameters are scaled using a single factor with 
            respect to the vertical profiles in the reference atmosphere
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            scf :: Scaling factor
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model2(atm,ipar,scf)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([1,npar,atm.NP])

    x1 = np.zeros(atm.NP)
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        x1[:] = atm.VMR[:,jvmr] * scf
        atm.VMR[:,jvmr] =  x1
    elif ipar==atm.NVMR: #Temperature
        x1[:] = atm.T[:] * scf
        atm.T[:] = x1 
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST:
            x1[:] = atm.DUST[:,jtmp] * scf
            atm.DUST[:,jtmp] = x1
        elif jtmp==atm.NDUST:
            x1[:] = atm.PARAH2 * scf
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            x1[:] = atm.FRAC * scf
            atm.FRAC = x1

    xmap[0,ipar,:] = x1[:]
    
    if MakePlot==True:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

        ax1.semilogx(atm.P/101325.,atm.H/1000.)
        ax2.plot(atm.T,atm.H/1000.)
        for i in range(atm.NVMR):
            ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Altitude (km)')
        ax3.set_xlabel('Volume mixing ratio')
        ax3.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model3(atm,ipar,scf,MakePlot=False):
    
    """
        FUNCTION NAME : model2()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 2 in NEMESIS.
            In this model, the atmospheric parameters are scaled using a single factor 
            in logscale with respect to the vertical profiles in the reference atmosphere
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            scf :: Log scaling factor
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model2(atm,ipar,scf)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([1,npar,atm.NP])

    x1 = np.zeros(atm.NP)
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        x1[:] = atm.VMR[:,jvmr] * np.exp(scf)
        atm.VMR[:,jvmr] =  x1 
    elif ipar==atm.NVMR: #Temperature
        x1[:] = atm.T[:] * np.exp(scf)
        atm.T[:] = x1 
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST:
            x1[:] = atm.DUST[:,jtmp] * np.exp(scf)
            atm.DUST[:,jtmp] = x1
        elif jtmp==atm.NDUST:
            x1[:] = atm.PARAH2 * np.exp(scf)
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            x1[:] = atm.FRAC * np.exp(scf)
            atm.FRAC = x1

    xmap[0,ipar,:] = x1[:]
    
    if MakePlot==True:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

        ax1.semilogx(atm.P/101325.,atm.H/1000.)
        ax2.plot(atm.T,atm.H/1000.)
        for i in range(atm.NVMR):
            ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Altitude (km)')
        ax3.set_xlabel('Volume mixing ratio')
        ax3.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap