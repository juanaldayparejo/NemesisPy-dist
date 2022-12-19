from NemesisPy import *
from NemesisPy.Models.Models import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from numba import jit

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 15 2022

@author: juanalday

Forward Model Class.
"""

class ForwardModel_0:

    def __init__(self, runname='wasp121', Atmosphere=None, Surface=None,
        Measurement=None, Spectroscopy=None, Stellar=None, Scatter=None,
        CIA=None, Layer=None, Variables=None, adjust_hydrostat=True):

        """Forward Model class

        The Forward Model class compiles different function required to perform the radiative transfer calculations.
        It takes as arguments the different Reference Classes, which provide all the required information to compute
        the forward model. The Forward Model class then modifies and uses this information to perform the calculations.


        Inputs (Reference classes)
        ------
        @class Atmosphere:,
            Class defining the Atmosphere
        @class Surface:,
            Class defining the Surface
        @class Measurement:,
            Class defining the Measurement
        @class Scatter:,
            Class defining the Scatter
        @class Spectroscopy:,
            Class defining the Spectroscopy
        @class Stellar:,
            Class defining the Stellar
        @class CIA:,
            Class defining the CIA
        @class Layer:,
            Class defining the Layer
        @class Variables:,
            Class defining the Variables
        @log adjust_hydrostat:,
            Flag indicating whether the re-adjustment of the pressure or altitude levels
            based on the hydrostatic equilibrium equation must be performed or not.

        Attributes (Attribute classes)
        ----------

        Note: The classes appear to be duplicated, some of them having an X in the end.
        The main difference between these two types is that ones are the reference classes
        given as an input, and which are not varied through the Forward Model. The other ones
        are the reference classes modified by the model paramterisations and varied through
        the calculations to calculate a specific forward model

        @attribute AtmosphereX:
            Class defining the Atmospheric profile for a particular forward model
        @attribute SurfaceX:
            Class defining the Surface for a particular forward model
        @attribute MeasurementX:
            Class defining the Measurement for a particular forward model
        @attribute ScatterX:
            Class defining the Scatter for a particular forward model
        @attribute CIAX:
            Class defining the CIA for a particular forward model
        @attribute SpectroscopyX:
            Class defining the Spectroscopy for a particular forward model
        @attribute StellarX:
            Class defining the Stellar for a particular forward model
        @attribute LayerX:
            Class defining the Layer for a particular forward model
        @attribute PathX:
            Class defining the Path for a particular forward model

        Methods
        -------

        Forward Models and Jacobeans
        ##########################################

            ForwardModel_0.nemesisfm()
            ForwardModel_0.nemesisfmg()
            ForwardModel_0.nemesisSOfm()
            ForwardModel_0.nemesisSOfmg()
            ForwardModel_0.jacobian_nemesis(nemesisSO=False)

        Mapping models into reference classes
        ##########################################

            ForwardModel_0.subprofretg()
            ForwardModel_0.subspecret()

        Path calculation and geometry
        ##########################################

            ForwardModel_0.select_Measurement()
            ForwardModel_0.calc_path()
            ForwardModel_0.calc_pathg()
            ForwardModel_0.calc_path_SO()
            ForwardModel_0.calc_pathg_SO()

        Radiative transfer calculations
        ##########################################

            ForwardModel_0.CIRSrad()
            ForwardModel_0.CIRSradg()
            ForwardModel_0.calc_tau_cia()
            ForwardModel_0.calc_tau_dust()

        Multiple scattering routines
        ###########################################

            ForwardModel_0.scloud11wave()

        """

        from copy import copy

        self.runname = runname

        #Building the reference classes into the Forward Model class
        self.Atmosphere = Atmosphere
        self.Surface = Surface
        self.Measurement = Measurement
        self.Scatter = Scatter
        self.Spectroscopy = Spectroscopy
        self.CIA = CIA
        self.Stellar = Stellar
        self.Variables = Variables
        self.Layer = Layer
        self.adjust_hydrostat=adjust_hydrostat

        #Creating extra class to hold the variables class in each permutation of the Jacobian Matrix
        self.Variables1 = copy(Variables)

        #Creating extra classes to store the parameters for a particular forward model
        self.AtmosphereX = copy(Atmosphere)
        self.SurfaceX = copy(Surface)
        self.MeasurementX = copy(Measurement)
        self.ScatterX = copy(Scatter)
        self.SpectroscopyX = copy(Spectroscopy)
        self.CIAX = copy(CIA)
        self.StellarX = copy(Stellar)
        self.LayerX = copy(Layer)
        self.PathX = None


    ###############################################################################################
    ###############################################################################################
    # CALCULATIONS OF FORWARD MODELS AND JACOBEANS
    ###############################################################################################
    ###############################################################################################

    ###############################################################################################

    def nemesisfm(self):

        """
            FUNCTION NAME : nemesisfm()

            DESCRIPTION : This function computes a forward model

            INPUTS : none

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra

            CALLING SEQUENCE:

                ForwardModel.nemesisfm()

            MODIFICATION HISTORY : Juan Alday (14/03/2022)

        """

        from NemesisPy.Models import subprofretg
        from NemesisPy.Path import AtmCalc_0,Path_0,calc_path
        from NemesisPy import find_nearest,subspecret
        from scipy import interpolate
        from copy import copy

        #Estimating the number of calculations that will need to be computed to model the spectra
        #included in the Measurement class (taking into account al geometries and averaging points)
        NCALC = np.sum(self.Measurement.NAV)
        SPECONV = np.zeros(self.Measurement.MEAS.shape) #Initalise the array where the spectra will be stored (NWAVE,NGEOM)
        for IGEOM in range(self.Measurement.NGEOM):

            #Calculating new wave array
            if self.Spectroscopy.ILBL==0:
                self.Measurement.wavesetb(self.Spectroscopy,IGEOM=IGEOM)
            if self.Spectroscopy.ILBL==2:
                self.Measurement.wavesetc(self.Spectroscopy,IGEOM=IGEOM)

            #Initialise array for averaging spectra (if required by NAV>1)
            SPEC = np.zeros(self.Measurement.NWAVE)
            dSPEC = np.zeros((self.Measurement.NWAVE,self.Variables.NX))
            WGEOMTOT = 0.0
            for IAV in range(self.Measurement.NAV[IGEOM]):

                #Selecting the relevant Measurement
                self.select_Measurement(IGEOM,IAV)

                #Making copy of classes to avoid overwriting them
                self.AtmosphereX = copy(self.Atmosphere)
                self.ScatterX = copy(self.Scatter)
                self.StellarX = copy(self.Stellar)
                self.SurfaceX = copy(self.Surface)
                self.SpectroscopyX = copy(self.Spectroscopy)
                self.LayerX = copy(self.Layer)
                self.CIAX = copy(self.CIA)
                flagh2p = False

                #Updating the required parameters based on the current geometry
                self.ScatterX.SOL_ANG = self.MeasurementX.SOL_ANG[0,0]
                self.ScatterX.EMISS_ANG = self.MeasurementX.EMISS_ANG[0,0]
                self.ScatterX.AZI_ANG = self.MeasurementX.AZI_ANG[0,0]

                if self.SpectroscopyX.ILBL==0:
                    self.MeasurementX.wavesetb(self.SpectroscopyX,IGEOM=IGEOM)
                if self.SpectroscopyX.ILBL==2:
                    self.MeasurementX.wavesetc(self.SpectroscopyX,IGEOM=IGEOM)

                #Changing the different classes taking into account the parameterisations in the state vector
                xmap = self.subprofretg()

                #Calling gsetpat to split the new reference atmosphere and calculate the path
                self.calc_path()

                #Calling CIRSrad to perform the radiative transfer calculations
                #SPEC1 = CIRSrad(self.runname,self.Variables.self.MeasurementX,self.AtmosphereX,self.SpectroscopyX,self.ScatterX,self.StellarX,self.SurfaceX,self.CIAX,self.LayerX,self.PathX)
                SPEC1 = self.CIRSrad()

                #Averaging the spectra in case NAV>1
                if self.MeasurementX.NAV[IGEOM]>1:
                    SPEC[:] = SPEC[:] + self.Measurement.WGEOM[IGEOM,IAV] * SPEC1[:,0]
                    WGEOMTOT = WGEOMTOT + self.Measurement.WGEOM[IGEOM,IAV]
                else:
                    SPEC[:] = SPEC1[:,0]

            if self.Measurement.NAV[IGEOM]>1:
                SPEC[:] = SPEC[:] / WGEOMTOT

            #Applying any changes to the spectra required by the state vector
            SPEC,dSPEC = self.subspecret(SPEC,dSPEC)

            #Convolving the spectra with the Instrument line shape
            if self.SpectroscopyX.ILBL==0: #k-tables

                if os.path.exists(self.runname+'.fwh')==True:
                    FWHMEXIST=self.runname
                else:
                    FWHMEXIST=''

                SPECONV1 = self.Measurement.conv(SPEC,IGEOM=IGEOM,FWHMEXIST='')

            elif Spectroscopy.ILBL==2: #LBL-tables

                SPECONV1 = self.Measurement.lblconv(SPEC,IGEOM=IGEOM)

            SPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM] = SPECONV1[0:self.Measurement.NCONV[IGEOM]]

        return SPECONV


    ###############################################################################################

    def nemesisfmg(self):

        """
            FUNCTION NAME : nemesisfmg()

            DESCRIPTION : This function computes a forward model and the analytical gradients

            INPUTS :

                runname :: Name of the Nemesis run
                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                Layer :: Python class defining the layering scheme to be applied in the calculations

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra
                dSPECMOD(NCONV,NGEOM,NX) :: Gradients of the spectra in each geometry with respect to the elements
                                            in the state vector

            CALLING SEQUENCE:

                nemesisfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from NemesisPy.Path import AtmCalc_0,Path_0,calc_pathg
        from NemesisPy.Radtrans import lblconv
        from NemesisPy import find_nearest,subspecret
        from scipy import interpolate
        from copy import copy
        from NemesisPy.Radtrans.Radtrans import CIRSradg
        from NemesisPy.Radtrans.Radtrans import map2pro
        from NemesisPy.Radtrans.Radtrans import map2xvec

        #Estimating the number of calculations that will need to be computed to model the spectra
        #included in the Measurement class (taking into account al geometries and averaging points)
        NCALC = np.sum(self.Measurement.NAV)
        SPECONV = np.zeros(self.Measurement.MEAS.shape) #Initalise the array where the spectra will be stored (NWAVE,NGEOM)
        dSPECONV = np.zeros((self.Measurement.NCONV.max(),self.Measurement.NGEOM,self.Variables.NX)) #Initalise the array where the gradients will be stored (NWAVE,NGEOM,NX)
        for IGEOM in range(self.Measurement.NGEOM):

            #Calculating new wave array
            if self.Spectroscopy.ILBL==0:
                self.Measurement.wavesetb(self.Spectroscopy,IGEOM=IGEOM)
            if self.Spectroscopy.ILBL==2:
                self.Measurement.wavesetc(self.Spectroscopy,IGEOM=IGEOM)

            #Initialise array for averaging spectra (if required by NAV>1)
            SPEC = np.zeros(self.Measurement.NWAVE)
            dSPEC = np.zeros((self.Measurement.NWAVE,self.Variables.NX))
            WGEOMTOT = 0.0
            for IAV in range(self.Measurement.NAV[IGEOM]):

                #Selecting the relevant Measurement
                self.select_Measurement(IGEOM,IAV)

                #Making copy of classes to avoid overwriting them
                self.AtmosphereX = copy(self.Atmosphere)
                self.ScatterX = copy(self.Scatter)
                self.StellarX = copy(self.Stellar)
                self.SurfaceX = copy(self.Surface)
                self.SpectroscopyX = copy(self.Spectroscopy)
                self.LayerX = copy(self.Layer)
                self.CIAX = copy(self.CIA)
                flagh2p = False

                #Updating the required parameters based on the current geometry
                self.ScatterX.SOL_ANG = self.MeasurementX.SOL_ANG[0,0]
                self.ScatterX.EMISS_ANG = self.MeasurementX.EMISS_ANG[0,0]
                self.ScatterX.AZI_ANG = self.MeasurementX.AZI_ANG[0,0]

                if self.SpectroscopyX.ILBL==0:
                    self.MeasurementX.wavesetb(self.SpectroscopyX,IGEOM=IGEOM)
                if self.SpectroscopyX.ILBL==2:
                    self.MeasurementX.wavesetc(self.SpectroscopyX,IGEOM=IGEOM)

                #Changing the different classes taking into account the parameterisations in the state vector
                xmap = self.subprofretg()

                #Calling gsetpat to split the new reference atmosphere and calculate the path
                self.calc_pathg()

                #Calling CIRSrad to perform the radiative transfer calculations
                #SPEC1,dSPEC3,dTSURF = CIRSradg(self.runname,self.Variables,self.MeasurementX,self.AtmosphereX,self.SpectroscopyX,self.ScatterX,self.StellarX,self.SurfaceX,self.CIAX,self.LayerX,self.PathX)
                SPEC1,dSPEC3,dTSURF = self.CIRSradg()

                #Mapping the gradients from Layer properties to Profile properties
                print('Mapping gradients from Layer to Profile')
                #Calculating the elements from NVMR+2+NDUST that need to be mapped
                incpar = []
                for i in range(self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST):
                    if np.mean(xmap[:,i,:])!=0.0:
                        incpar.append(i)

                dSPEC2 = map2pro(dSPEC3,self.MeasurementX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.PathX.NLAYIN,self.PathX.LAYINC,self.LayerX.DTE,self.LayerX.DAM,self.LayerX.DCO,INCPAR=incpar)
                #(NWAVE,NVMR+2+NDUST,NPRO,NPATH)
                del dSPEC3

                #Mapping the gradients from Profile properties to elements in state vector
                print('Mapping gradients from Profile to State Vector')
                dSPEC1 = map2xvec(dSPEC2,self.MeasurementX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.Variables.NX,xmap)
                #(NWAVE,NPATH,NX)
                del dSPEC2

                #Adding the temperature surface gradient if required
                if self.Variables.JSURF>=0:
                    dSPEC1[:,0,self.Variables.JSURF] = dTSURF[:,0]

                #Averaging the spectra in case NAV>1
                if self.Measurement.NAV[IGEOM]>1:
                    SPEC[:] = SPEC[:] + self.Measurement.WGEOM[IGEOM,IAV] * SPEC1[:,0]
                    dSPEC[:,:] = dSPEC[:,:] + self.Measurement.WGEOM[IGEOM,IAV] * dSPEC1[:,0,:]
                    WGEOMTOT = WGEOMTOT + self.Measurement.WGEOM[IGEOM,IAV]
                else:
                    SPEC[:] = SPEC1[:,0]
                    dSPEC[:,:] = dSPEC1[:,0,:]

            if self.Measurement.NAV[IGEOM]>1:
                SPEC[:] = SPEC[:] / WGEOMTOT
                dSPEC[:,:] = dSPEC[:,:] / WGEOMTOT

            #Applying any changes to the spectra required by the state vector
            SPEC,dSPEC = self.subspecret(SPEC,dSPEC)

            #Convolving the spectra with the Instrument line shape
            if self.Spectroscopy.ILBL==0: #k-tables

                if os.path.exists(self.runname+'.fwh')==True:
                    FWHMEXIST=self.runname
                else:
                    FWHMEXIST=''

                SPECONV1,dSPECONV1 = self.Measurement.convg(SPEC,dSPEC,IGEOM=IGEOM,FWHMEXIST='')

            elif self.Spectroscopy.ILBL==2: #LBL-tables

                SPECONV1,dSPECONV1 = self.Measurement.lblconvg(SPEC,dSPEC,IGEOM=IGEOM)

            SPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM] = SPECONV1[0:self.Measurement.NCONV[IGEOM]]
            dSPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM,:] = dSPECONV1[0:self.Measurement.NCONV[IGEOM],:]

        return SPECONV,dSPECONV

    ###############################################################################################

    def nemesisSOfm(self):

        """
            FUNCTION NAME : nemesisSOfm()

            DESCRIPTION : This function computes a forward model for a solar occultation observation

            INPUTS : none

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra

            CALLING SEQUENCE:

                ForwardModel.nemesisSOfm()

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from NemesisPy.Models import subprofretg,subspecret
        from NemesisPy.Path import AtmCalc_0,Path_0,calc_path_SO
        from NemesisPy import find_nearest
        from scipy import interpolate
        from copy import copy
        from NemesisPy.Radtrans.Radtrans import CIRSrad

        #First we change the reference atmosphere taking into account the parameterisations in the state vector
        self.Variables1 = copy(self.Variables)
        self.MeasurementX = copy(self.Measurement)
        self.AtmosphereX = copy(self.Atmosphere)
        self.ScatterX = copy(self.Scatter)
        self.StellarX = copy(self.Stellar)
        self.SurfaceX = copy(self.Surface)
        self.LayerX = copy(self.Layer)
        self.SpectroscopyX = copy(self.Spectroscopy)
        self.CIAX = copy(self.CIA)
        flagh2p = False

        #Setting up flag not to re-compute levels based on hydrostatic equilibrium (unless pressure or tangent altitude are retrieved)
        self.adjust_hydrostat = False

        #Mapping variables into different classes
        xmap = self.subprofretg()

        #Calculating the atmospheric paths
        self.calc_path_SO()
        BASEH_TANHE = np.zeros(self.PathX.NPATH)
        for i in range(self.PathX.NPATH):
            BASEH_TANHE[i] = self.LayerX.BASEH[self.PathX.LAYINC[int(self.PathX.NLAYIN[i]/2),i]]/1.0e3

        #Calling CIRSrad to calculate the spectra
        #SPECOUT = CIRSrad(self.MeasurementX,self.AtmosphereX,self.SpectroscopyX,self.ScatterX,self.StellarX,self.SurfaceX,self.CIAX,self.LayerX,self.PathX)
        SPECOUT = self.CIRSrad()

        #Interpolating the spectra to the correct altitudes defined in Measurement
        SPECMOD = np.zeros([self.MeasurementX.NWAVE,self.MeasurementX.NGEOM])
        dSPECMOD = np.zeros([self.MeasurementX.NWAVE,self.MeasurementX.NGEOM,self.Variables.NX])
        for i in range(self.MeasurementX.NGEOM):

            #Find altitudes above and below the actual tangent height
            base0,ibase = find_nearest(BASEH_TANHE,self.MeasurementX.TANHE[i])
            if base0<=self.MeasurementX.TANHE[i]:
                ibasel = ibase
                ibaseh = ibase + 1
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            if ibaseh>self.PathX.NPATH-1:
                SPECMOD[:,i] = SPECOUT[:,ibasel]
            else:
                fhl = (self.MeasurementX.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
                fhh = (BASEH_TANHE[ibaseh]-self.MeasurementX.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

                SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)

        #Applying any changes to the spectra required by the state vector
        SPECMOD,dSPECMOD = self.subspecret(SPECMOD,dSPECMOD)

        #Convolving the spectrum with the instrument line shape
        print('Convolving spectra and gradients with instrument line shape')
        if self.SpectroscopyX.ILBL==0:
            SPECONV,dSPECONV = self.MeasurementX.convg(SPECMOD,dSPECMOD,IGEOM='All')
        elif self.SpectroscopyX.ILBL==2:
            SPECONV = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')

        return SPECONV


    ###############################################################################################

    def nemesisSOfmg(self):

        """
            FUNCTION NAME : nemesisSOfmg()

            DESCRIPTION : This function computes a forward model for a solar occultation observation and the gradients
                       of the transmission spectrum with respect to the elements in the state vector

            INPUTS :

                runname :: Name of the Nemesis run
                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra
                dSPECMOD(NCONV,NGEOM,NX) :: Derivatives of each spectrum in each geometry with
                                        respect to the elements of the state vector

            CALLING SEQUENCE:

                nemesisSOfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from NemesisPy.Path import AtmCalc_0,Path_0,calc_pathg_SO
        from NemesisPy import find_nearest
        from scipy import interpolate
        from copy import copy
        from NemesisPy.Radtrans.Radtrans import CIRSradg
        from NemesisPy.Radtrans.Radtrans import map2pro
        from NemesisPy.Radtrans.Radtrans import map2xvec

        #First we change the reference atmosphere taking into account the parameterisations in the state vector
        self.MeasurementX = copy(self.Measurement)
        self.AtmosphereX = copy(self.Atmosphere)
        self.ScatterX = copy(self.Scatter)
        self.StellarX = copy(self.Stellar)
        self.SurfaceX = copy(self.Surface)
        self.LayerX = copy(self.Layer)
        self.SpectroscopyX = copy(self.Spectroscopy)
        self.CIAX = copy(self.CIA)
        flagh2p = False

        #Setting up flag not to re-compute levels based on hydrostatic equilibrium (unless pressure or tangent altitude are retrieved)
        self.adjust_hydrostat = False

        #Mapping variables into different classes
        xmap = self.subprofretg()

        #Calculating the atmospheric paths
        self.calc_pathg_SO()
        BASEH_TANHE = np.zeros(self.PathX.NPATH)
        for i in range(self.PathX.NPATH):
            BASEH_TANHE[i] = self.LayerX.BASEH[self.PathX.LAYINC[int(self.PathX.NLAYIN[i]/2),i]]/1.0e3


        #Calling CIRSrad to calculate the spectra
        print('Running CIRSradg')
        #SPECOUT,dSPECOUT2,dTSURF = CIRSradg(self.runname,self.Variables,self.MeasurementX,self.AtmosphereX,self.SpectroscopyX,self.ScatterX,self.StellarX,self.SurfaceX,self.CIAX,self.LayerX,self.PathX)
        SPECOUT,dSPECOUT2,dTSURF = self.CIRSradg()

        #Mapping the gradients from Layer properties to Profile properties
        print('Mapping gradients from Layer to Profile')
        #Calculating the elements from NVMR+2+NDUST that need to be mapped
        incpar = []
        for i in range(self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST):
            if np.mean(xmap[:,i,:])!=0.0:
                incpar.append(i)

        dSPECOUT1 = map2pro(dSPECOUT2,self.MeasurementX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.PathX.NLAYIN,self.PathX.LAYINC,self.LayerX.DTE,self.LayerX.DAM,self.LayerX.DCO,INCPAR=incpar)
        #(NWAVE,NVMR+2+NDUST,NPRO,NPATH)
        del dSPECOUT2

        #Mapping the gradients from Profile properties to elements in state vector
        print('Mapping gradients from Profile to State Vector')
        dSPECOUT = map2xvec(dSPECOUT1,self.MeasurementX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.Variables.NX,xmap)
        #(NWAVE,NPATH,NX)
        del dSPECOUT1

        #Interpolating the spectra to the correct altitudes defined in Measurement
        SPECMOD = np.zeros([self.MeasurementX.NWAVE,self.MeasurementX.NGEOM])
        dSPECMOD = np.zeros([self.MeasurementX.NWAVE,self.MeasurementX.NGEOM,self.Variables.NX])
        for i in range(self.MeasurementX.NGEOM):

            #Find altitudes above and below the actual tangent height
            base0,ibase = find_nearest(BASEH_TANHE,self.MeasurementX.TANHE[i])
            if base0<=self.MeasurementX.TANHE[i]:
                ibasel = ibase
                ibaseh = ibase + 1
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            if ibaseh>self.PathX.NPATH-1:
                SPECMOD[:,i] = SPECOUT[:,ibasel]
                dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]
            else:
                fhl = (self.MeasurementX.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
                fhh = (BASEH_TANHE[ibaseh]-self.MeasurementX.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

                SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)
                dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]*(1.-fhl) + dSPECOUT[:,ibaseh,:]*(1.-fhh)


        #Applying any changes to the spectra required by the state vector
        SPECMOD,dSPECMOD = self.subspecret(SPECMOD,dSPECMOD)


        #Convolving the spectrum with the instrument line shape
        print('Convolving spectra and gradients with instrument line shape')
        if self.SpectroscopyX.ILBL==0:
            SPECONV,dSPECONV = self.MeasurementX.convg(SPECMOD,dSPECMOD,IGEOM='All')
        elif self.SpectroscopyX.ILBL==2:
            SPECONV,dSPECONV = self.MeasurementX.lblconvg(SPECMOD,dSPECMOD,IGEOM='All')

        #Calculating the gradients of any parameterisations involving the convolution
        dSPECONV = self.subspeconv(SPECMOD,dSPECONV)

        return SPECONV,dSPECONV

    ###############################################################################################

    def jacobian_nemesis(self,NCores=1,nemesisSO=False):

        """

            FUNCTION NAME : jacobian_nemesisSO()

            DESCRIPTION :

                This function calculates the Jacobian matrix by calling nx+1 times nemesisSOfm().
                This routine is set up so that each forward model is calculated in parallel,
                increasing the computational speed of the code

            INPUTS :

                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated
                NCores :: Number of cores that can be used to parallelise the calculation of the jacobian matrix

            OUTPUTS :

                YN(NY) :: New measurement vector
                KK(NY,NX) :: Jacobian matrix

            CALLING SEQUENCE:

                YN,KK = jacobian_nemesisSO(Variables,Measurement,Atmosphere,Scatter,Stellar,Surface,CIA,Layer)

            MODIFICATION HISTORY : Juan Alday (29/07/2021)

        """

        from NemesisPy import nemesisSOfm
        from NemesisPy import nemesisSOfm_parallel
        from copy import copy


        #################################################################################
        # Making some calculations for storing all the arrays
        #################################################################################

        nproc = self.Variables.NX+1 #Number of times we need to run the forward model

        #Constructing state vector after perturbation of each of the elements and storing in matrix

        self.Variables.calc_DSTEP() #Calculating the step size for the perturbation of each element
        nxn = self.Variables.NX+1
        xnx = np.zeros([self.Variables.NX,nxn])
        for i in range(self.Variables.NX+1):
            if i==0:   #First element is the normal state vector
                xnx[0:self.Variables.NX,i] = self.Variables.XN[0:self.Variables.NX]
            else:      #Perturbation of each element
                xnx[0:self.Variables.NX,i] = self.Variables.XN[0:self.Variables.NX]
                #xnx[i-1,i] = Variables.XN[i-1]*1.05
                xnx[i-1,i] = self.Variables.XN[i-1] + self.Variables.DSTEP[i-1]
                if self.Variables.XN[i-1]==0.0:
                    xnx[i-1,i] = 0.05


        """
        #Because of the parallelisation, the parameters that are kept fixed need to be located at the end of the
        #state vector, otherwise the code crashes
        ic = 0
        for i in range(self.Variables.NX):

            if ic==1:
                if self.Variables.FIX[i]==0:
                    sys.exit('error :: Fixed parameters in the state vector must be located at the end of the array')

            if self.Variables.FIX[i]==1:
                ic = 1
        """

        #################################################################################
        # Calculating the first forward model and the analytical part of Jacobian
        #################################################################################

        #self.Variables.NUM[:] = 1     #Uncomment for trying numerical differentiation

        ian1 = np.where(self.Variables.NUM==0)  #Gradients calculated using CIRSradg
        ian1 = ian1[0]

        iYN = 0
        KK = np.zeros([self.Measurement.NY,self.Variables.NX])

        if len(ian1)>0:

            print('Calculating analytical part of the Jacobian :: Calling nemesisSOfmg ')

            if nemesisSO==True:
                SPECMOD,dSPECMOD = self.nemesisSOfmg()
            else:
                SPECMOD,dSPECMOD = self.nemesisfmg()

            YN = np.resize(np.transpose(SPECMOD),[self.Measurement.NY])
            for ix in range(self.Variables.NX):
                KK[:,ix] = np.resize(np.transpose(dSPECMOD[:,:,ix]),[self.Measurement.NY])

            iYN = 1 #Indicates that some of the gradients and the measurement vector have already been caculated


        #################################################################################
        # Calculating all the required forward models for numerical differentiation
        #################################################################################

        inum1 = np.where( (self.Variables.NUM==1) & (self.Variables.FIX==0) )
        inum = inum1[0]

        if iYN==0:
            nfm = len(inum) + 1  #Number of forward models to run to calculate the Jacobian and measurement vector
            ixrun = np.zeros(nfm,dtype='int32')
            ixrun[0] = 0
            ixrun[1:nfm] = inum[:] + 1
        else:
            nfm = len(inum)  #Number of forward models to run to calculate the Jacobian
            ixrun = np.zeros(nfm,dtype='int32')
            ixrun[0:nfm] = inum[:] + 1


        #Calling the forward model nfm times to calculate the measurement vector for each case
        YNtot = np.zeros([self.Measurement.NY,nfm])

        print('Calculating numerical part of the Jacobian :: running '+str(nfm)+' forward models ')
        if NCores>1:
            #ray.init(num_cpus=NCores)
            #YNtot_ids = []
            #SpectroscopyP = ray.put(self.Spectroscopy)
            #for ix in range(nfm):
            #    YNtot_ids.append(nemesisSOfm_parallel.remote(ixrun[ix],runname,Variables,Measurement,Atmosphere,SpectroscopyP,Scatter,Stellar,Surface,CIA,Layer,xnx))
            sys.exit('error :: Need to implement NCores>1 this in the ForwardModel class')


            #Block until the results have finished and get the results.
            YNtot1 = ray.get(YNtot_ids)
            for ix in range(nfm):
                YNtot[0:self.Measurement.NY,ix] = YNtot1[ix]
            ray.shutdown()

        else:

            for ifm in range(nfm):
                print('Calculating forward model '+str(ifm)+'/'+str(nfm))
                #self.Variables1 = copy(self.Variables)
                #self.Variables1.XN = xnx[0:self.Variables1.NX,ixrun[ifm]]
                self.Variables.XN = xnx[0:self.Variables.NX,ixrun[ifm]]

                if nemesisSO==True:
                    SPECMOD = self.nemesisSOfm()
                else:
                    SPECMOD = self.nemesisfm()
                YNtot[0:self.Measurement.NY,ifm] = np.resize(np.transpose(SPECMOD),[self.Measurement.NY])

        if iYN==0:
            YN = np.zeros(self.Measurement.NY)
            YN[:] = YNtot[0:self.Measurement.NY,0]

        #################################################################################
        # Calculating the Jacobian matrix
        #################################################################################

        for i in range(len(inum)):

            if iYN==0:
                ifm = i + 1
            else:
                ifm = i

            xn1 = self.Variables.XN[inum[i]] * 1.05
            if xn1==0.0:
                xn1=0.05
            if self.Variables.FIX[inum[i]] == 0:
                KK[:,inum[i]] = (YNtot[:,ifm]-YN)/(xn1-self.Variables.XN[inum[i]])

        return YN,KK



    ###############################################################################################
    ###############################################################################################
    # MAPPING THE STATE VECTOR (MODEL PARAMETERISATIONS) INTO THE REFERENCE CLASSES
    ###############################################################################################
    ###############################################################################################


    ###############################################################################################

    def subprofretg(self):

        """
        FUNCTION NAME : subprogretg()

        DESCRIPTION : Updates the reference classes based on the variables and parameterisations in the
                      state vector. Changes to other parameters in the model based on the variables
                      and parameterisations in the state vector are also performed here. However,
                      the functional derivatives to these other parameters are not included since
                      they cannot be determined analytically.

        INPUTS : none

        OPTIONAL INPUTS: none

        OUTPUTS :

            xmap(maxv,ngas+2+ncont,npro) :: Matrix relating functional derivatives calculated
                                             by CIRSRADG to the elements of the state vector.
                                             Elements of XMAP are the rate of change of
                                             the profile vectors (i.e. temperature, vmr prf
                                             files) with respect to the change in the state
                                             vector elements. So if X1(J) is the modified
                                             temperature,vmr,clouds at level J to be
                                             written out to runname.prf or aerosol.prf then
                                            XMAP(K,L,J) is d(X1(J))/d(XN(K)) and where
                                            L is the identifier (1 to NGAS+1+2*NCONT)

        CALLING SEQUENCE:

            xmap = ForwardModel.subprofretg()

        MODIFICATION HISTORY : Juan Alday (15/03/2022)

        """

        #First determining whether there are multiple latitudes/longitudes in the Atmosphere or Surface
        #If so, calculating the relevant surface and atmospheric properties at the latitude and longitude of the observation

        if self.Atmosphere.NLOCATIONS>1:

            #Selecting the relevant atmospheric profiles
            self.AtmosphereX.NLOCATIONS = 1
            self.select_Profile(self.MeasurementX.LATITUDE,self.MeasurementX.LONGITUDE)

        if self.Surface.NLOCATIONS>1:

            #Selecting the relevant surface properties
            self.SurfaceX.NLOCATIONS = 1
            self.select_Surface(self.MeasurementX.LATITUDE,self.MeasurementX.LONGITUDE)


        #Checking if hydrostatic equilibrium needs to be used for any parameterisation
        if self.Variables.JPRE!=-1:
            self.adjust_hydrostat = True
        if self.Variables.JTAN!=-1:
            self.adjust_hydrostat = True

        #Modify profile via hydrostatic equation to make sure the atm is in hydrostatic equilibrium
        if self.adjust_hydrostat==True:
            if self.Variables.JPRE==-1:
                jhydro = 0
                #Then we modify the altitude levels and keep the pressures fixed
                self.AtmosphereX.adjust_hydrostatH()
                self.AtmosphereX.calc_grav()   #Updating the gravity values at the new heights
            else:
                #Then we modifify the pressure levels and keep the altitudes fixed
                jhydro = 1
                for i in range(self.Variables.NVAR):
                    if self.Variables.VARIDENT[i,0]==666:
                        htan = self.Variables.VARPARAM[i,0] * 1000.
                ptan = np.exp(self.Variables.XN[self.Variables.JPRE]) * 101325.
                self.AtmosphereX.adjust_hydrostatP(htan,ptan)

        #Adjust VMRs to add up to 1 if AMFORM=1 and re-calculate molecular weight in atmosphere
        if self.AtmosphereX.AMFORM==1:
            self.AtmosphereX.adjust_VMR()
            self.AtmosphereX.calc_molwt()
        elif self.AtmosphereX.AMFORM==2:
            self.AtmosphereX.calc_molwt()

        #Calculate atmospheric density
        rho = self.AtmosphereX.calc_rho() #kg/m3

        #Initialising xmap
        xmap = np.zeros((self.Variables.NX,self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST,self.AtmosphereX.NP))

        #Going through the different variables an updating the atmosphere accordingly
        ix = 0
        for ivar in range(self.Variables.NVAR):

            if self.Variables.VARIDENT[ivar,2]<=100:

                #Reading the atmospheric profile which is going to be changed by the current variable
                xref = np.zeros([self.AtmosphereX.NP])

                if self.Variables.VARIDENT[ivar,0]==0:     #Temperature is to be retrieved
                    xref[:] = self.AtmosphereX.T
                    ipar = self.AtmosphereX.NVMR
                elif self.Variables.VARIDENT[ivar,0]>0:    #Gas VMR is to be retrieved
                    jvmr = np.where( (np.array(self.AtmosphereX.ID)==self.Variables.VARIDENT[ivar,0]) & (np.array(self.AtmosphereX.ISO)==self.Variables.VARIDENT[ivar,1]) )
                    jvmr = int(jvmr[0])
                    xref[:] = self.AtmosphereX.VMR[:,jvmr]
                    ipar = jvmr
                elif self.Variables.VARIDENT[ivar,0]<0:
                    jcont = -int(self.Variables.VARIDENT[ivar,0])
                    if jcont>self.AtmosphereX.NDUST+2:
                        sys.exit('error :: Variable outside limits',self.Variables.VARIDENT[ivar,0],self.Variables.VARIDENT[ivar,1],self.Variables.VARIDENT[ivar,2])
                    elif jcont==self.AtmosphereX.NDUST+1:   #Para-H2
                        if flagh2p==True:
                            xref[:] = self.AtmosphereX.PARAH2
                        else:
                            sys.exit('error :: Para-H2 is declared as variable but atmosphere is not from Giant Planet')
                    elif abs(jcont)==self.AtmosphereX.NDUST+2: #Fractional cloud cover
                        xref[:] = self.AtmosphereX.FRAC
                    else:
                        xref[:] = self.AtmosphereX.DUST[:,jcont-1]

                    ipar = self.AtmosphereX.NVMR + jcont

            x1 = np.zeros(self.AtmosphereX.NP)

            if self.Variables.VARIDENT[ivar,2]==-1:
#           Model -1. Continuous aerosol profile in particles cm-3
#           ***************************************************************

                xprof = np.zeros(self.Variables.NXVAR[ivar])
                xprof[:] = self.Variables.XN[ix:ix+self.Variables.NXVAR[ivar]]
                self.AtmosphereX,xmap1 = modelm1(self.AtmosphereX,ipar,xprof)
                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,2]==0:
#           Model 0. Continuous profile
#           ***************************************************************

                xprof = np.zeros(self.Variables.NXVAR[ivar])
                xprof[:] = self.Variables.XN[ix:ix+self.Variables.NXVAR[ivar]]
                self.AtmosphereX,xmap1 = model0(self.AtmosphereX,ipar,xprof)
                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,2]==2:
#           Model 2. Scaling factor
#           ***************************************************************

                self.AtmosphereX,xmap1 = model2(self.AtmosphereX,ipar,self.Variables.XN[ix])
                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,2]==3:
#           Model 3. Log scaling factor
#           ***************************************************************

                self.AtmosphereX,xmap1 = model3(self.AtmosphereX,ipar,self.Variables.XN[ix])
                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,2]==9:
#           Model 9. Simple cloud represented by base height, fractional scale height
#                    and the total integrated cloud density
#           ***************************************************************

                tau = np.exp(self.Variables.XN[ix])  #Integrated dust column-density
                fsh = np.exp(self.Variables.XN[ix+1]) #Fractional scale height
                href = self.Variables.XN[ix+2] #Base height (km)

                self.AtmosphereX,xmap1 = model9(self.AtmosphereX,ipar,href,fsh,tau)
                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,2]==49:
#           Model 50. Continuous profile in linear scale
#           ***************************************************************

                xprof = np.zeros(self.Variables.NXVAR[ivar])
                xprof[:] = self.Variables.XN[ix:ix+self.Variables.NXVAR[ivar]]
                self.AtmosphereX,xmap1 = model49(self.AtmosphereX,ipar,xprof)
                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,2]==50:
#           Model 50. Continuous profile of scaling factors
#           ***************************************************************

                xprof = np.zeros(self.Variables.NXVAR[ivar])
                xprof[:] = self.Variables.XN[ix:ix+self.Variables.NXVAR[ivar]]
                self.AtmosphereX,xmap1 = model50(self.AtmosphereX,ipar,xprof)
                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==228:
#           Model 228. Retrieval of instrument line shape for ACS-MIR and wavelength calibration
#           **************************************************************************************

                V0 = self.Variables.XN[ix]
                C0 = self.Variables.XN[ix+1]
                C1 = self.Variables.XN[ix+2]
                C2 = self.Variables.XN[ix+3]
                P0 = self.Variables.XN[ix+4]
                P1 = self.Variables.XN[ix+5]
                P2 = self.Variables.XN[ix+6]
                P3 = self.Variables.XN[ix+7]

                self.MeasurementX,self.SpectroscopyX = model228(self.MeasurementX,self.SpectroscopyX,V0,C0,C1,C2,P0,P1,P2,P3)

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==229:
#           Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
#           ***************************************************************

                par1 = self.Variables.XN[ix]
                par2 = self.Variables.XN[ix+1]
                par3 = self.Variables.XN[ix+2]
                par4 = self.Variables.XN[ix+3]
                par5 = self.Variables.XN[ix+4]
                par6 = self.Variables.XN[ix+5]
                par7 = self.Variables.XN[ix+6]

                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par5,par6,par7)

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==230:
#           Model 230. Retrieval of multiple instrument line shapes for ACS-MIR
#           ***************************************************************

                nwindows = int(self.Variables.VARPARAM[ivar,0])
                liml = np.zeros(nwindows)
                limh = np.zeros(nwindows)
                i0 = 1
                for iwin in range(nwindows):
                    liml[iwin] = self.Variables.VARPARAM[ivar,i0]
                    limh[iwin] = self.Variables.VARPARAM[ivar,i0+1]
                    i0 = i0 + 2

                par1 = np.zeros((7,nwindows))
                for iwin in range(nwindows):
                    for jwin in range(7):
                        par1[jwin,iwin] = self.Variables.XN[ix]
                        ix = ix + 1

                self.MeasurementX = model230(self.MeasurementX,nwindows,liml,limh,par1)

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==231:
#           Model 231. Continuum addition to transmission spectra using a varying scaling factor (given a polynomial of degree N)
#           ***************************************************************

                #The computed transmission spectra is multiplied by R = R0 * POL
                #Where POL is given by POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

                #The effect of this model takes place after the computation of the spectra in CIRSrad!
                if int(self.Variables.VARPARAM[ivar,0])!=self.MeasurementX.NGEOM:
                    sys.exit('error using Model 231 :: The number of levels for the addition of continuum must be the same as NGEOM')

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==2310:
#           Model 2310. Continuum addition to transmission spectra using a varying scaling factor (given a polynomial of degree N)
#                       in several spectral windows
#           ***************************************************************

                #The computed transmission spectra is multiplied by R = R0 * POL
                #Where POL is given by POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

                #The effect of this model takes place after the computation of the spectra in CIRSrad!
                if int(self.Variables.VARPARAM[ivar,0])!=self.MeasurementX.NGEOM:
                    sys.exit('error using Model 2310 :: The number of levels for the addition of continuum must be the same as NGEOM')

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==232:
#           Model 232. Continuum addition to transmission spectra using the angstrom coefficient
#           ***************************************************************

                #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
                #Where the parameters to fit are TAU0 and ALPHA

                #The effect of this model takes place after the computation of the spectra in CIRSrad!
                if int(self.Variables.NXVAR[ivar]/2)!=self.MeasurementX.NGEOM:
                    sys.exit('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==233:
#           Model 233. Continuum addition to transmission spectra using a variable angstrom coefficient
#           ***************************************************************

                #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
                #Where the aerosol opacity is modelled following

                # np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

                #The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
                #233 converges to model 232 when a2=0.

                #The effect of this model takes place after the computation of the spectra in CIRSrad!
                if int(self.Variables.NXVAR[ivar]/3)!=self.MeasurementX.NGEOM:
                    sys.exit('error using Model 233 :: The number of levels for the addition of continuum must be the same as NGEOM')

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==446:
#           Model 446. model for retrieving an aerosol density profile + aerosol particle size (log-normal distribution)
#           ***************************************************************

                #This model fits a continuous vertical profile for the aerosol density and the particle size, which
                #is assumed to follow a log-normal distribution

                if int(self.Variables.NXVAR[ivar]/2)!=self.AtmosphereX.NP:
                    sys.exit('error using Model 446 :: The number of levels for the addition of continuum must be the same as NPRO')

                aero_dens = np.zeros(self.AtmosphereX.NP)
                aero_rsize = np.zeros(self.AtmosphereX.NP)
                aero_dens[:] = np.exp(self.Variables.XN[ix:ix+self.AtmosphereX.NP])
                aero_rsize[:] = np.exp(self.Variables.XN[ix+self.AtmosphereX.NP:ix+self.AtmosphereX.NP+self.AtmosphereX.NP])
                #aero_rsize[:] = Variables.XN[ix+Atmosphere.NP:ix+Atmosphere.NP+Atmosphere.NP]

                nlevel = int(self.Variables.VARPARAM[ivar,0])
                aero_id = int(self.Variables.VARPARAM[ivar,1])
                sigma_rsize = self.Variables.VARPARAM[ivar,2]
                idust0 = int(self.Variables.VARPARAM[ivar,3])
                WaveNorm = self.Variables.VARPARAM[ivar,4]

                #Reading the refractive index from the dictionary
                self.ScatterX.WAVER = aerosol_info[str(aero_id)]["wave"]
                self.ScatterX.REFIND_REAL = aerosol_info[str(aero_id)]["refind_real"]
                self.ScatterX.REFIND_IM = aerosol_info[str(aero_id)]["refind_im"]

                if self.AtmosphereX.NDUST<nlevel:
                    sys.exit('error in Model 446 :: The number of aerosol populations must at least be equal to the number of altitude levels')

                self.AtmosphereX,self.ScatterX,xmap1 = model446(self.AtmosphereX,self.ScatterX,idust0,aero_id,aero_dens,aero_rsize,sigma_rsize,WaveNorm)

                xmap[ix:ix+self.Variables.NXVAR[ivar],:,0:self.AtmosphereX.NP] = xmap1[:,:,:]

                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==666:
#           Model 666. Retrieval of tangent pressure at given tangent height
#           ***************************************************************
                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==667:
#           Model 667. Retrieval of dilution factor to account for thermal gradients in planets
#           ***************************************************************
                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif Variables.VARIDENT[ivar,0]==999:
#           Model 999. Retrieval of surface temperature
#           ***************************************************************

                tsurf = self.Variables.XN[ix]
                self.SurfaceX.TSURF = tsurf

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            else:
                print('error in Variable ',self.Variables.VARIDENT[ivar,0],self.Variables.VARIDENT[ivar,1],self.Variables.VARIDENT[ivar,2])
                sys.exit('error :: Model parameterisation has not yet been included')


        #Now check if any gas in the retrieval saturates

        #Adjust VMRs to add up to 1 if AMFORM=1
        if self.AtmosphereX.AMFORM==1:
            self.AtmosphereX.adjust_VMR()
            self.AtmosphereX.calc_molwt()

        #Re-scale H/P based on the hydrostatic equilibrium equation
        if self.adjust_hydrostat==True:
            if jhydro==0:
                #Then we modify the altitude levels and keep the pressures fixed
                self.AtmosphereX.adjust_hydrostatH()
                self.AtmosphereX.calc_grav()   #Updating the gravity values at the new heights
            else:
                #Then we modifify the pressure levels and keep the altitudes fixed
                self.AtmosphereX.adjust_hydrostatP(htan,ptan)

        #Write out modified profiles to .prf file
        #Atmosphere.write_to_file()

        return xmap


    ###############################################################################################

    def subspecret(self,SPECMOD,dSPECMOD):

        """
        FUNCTION NAME : subspecret()

        DESCRIPTION : Performs any required changes to the modelled spectra based on the parameterisations
                      included in the state vector. These changes can include for example the superposition
                      of diffraction orders in an AOTF spectrometer or the scaling of the spectra to account
                      for hemispheric assymmetries in exoplanet retrievals.

        INPUTS :

            Measurement :: Python class defining the observation
            Variables :: Python class defining the parameterisations and state vector
            SPECMOD(NWAVE,NGEOM) :: Modelled spectrum in each geometry (not yet convolved with ILS)
            dSPECMOD(NWAVE,NGEOM,NX) :: Modelled gradients in each geometry (not yet convolved with ILS)

        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is made

        OUTPUTS :

            SPECMOD :: Updated modelled spectrum
            dSPECMOD :: Updated gradients

        CALLING SEQUENCE:

            SPECMOD = subspecret(Measurement,Variables,SPECMOD)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)

        """

        #Going through the different variables an updating the spectra and gradients accordingly
        ix = 0
        for ivar in range(self.Variables.NVAR):

            if self.Variables.VARIDENT[ivar,0]==231:
#           Model 231. Scaling of spectra using a varying scaling factor (following a polynomial of degree N)
#           ****************************************************************************************************

                NGEOM = int(self.Variables.VARPARAM[ivar,0])
                NDEGREE = int(self.Variables.VARPARAM[ivar,1])

                if self.MeasurementX.NGEOM>1:

                    for i in range(self.MeasurementX.NGEOM):

                        #Getting the coefficients
                        T = np.zeros(NDEGREE+1)
                        for j in range(NDEGREE+1):
                            T[j] = self.Variables.XN[ix+j]

                        #WAVE0 = self.MeasurementX.WAVE.min()
                        WAVE0 = self.MeasurementX.VCONV[0,0]
                        spec = np.zeros(self.MeasurementX.NWAVE)
                        spec[:] = SPECMOD[:,i]

                        #Changing the state vector based on this parameterisation
                        POL = np.zeros(self.MeasurementX.NWAVE)
                        for j in range(NDEGREE+1):
                            POL[:] = POL[:] + T[j]*(self.MeasurementX.WAVE[:]-WAVE0)**j

                        SPECMOD[:,i] = SPECMOD[:,i] * POL[:]

                        #Changing the rest of the gradients based on the impact of this parameterisation
                        for ixn in range(self.Variables.NX):
                            dSPECMOD[:,i,ixn] = dSPECMOD[:,i,ixn] * POL[:]

                        #Defining the analytical gradients for this parameterisation
                        for j in range(NDEGREE+1):
                            dSPECMOD[:,i,ix+j] = spec[:] * (self.Measurement.WAVE[:]-WAVE0)**j

                        ix = ix + (NDEGREE+1)

            elif self.Variables.VARIDENT[ivar,0]==2310:
#           Model 2310. Scaling of spectra using a varying scaling factor (following a polynomial of degree N)
#                       in multiple spectral windows
#           ****************************************************************************************************

                NGEOM = int(self.Variables.VARPARAM[ivar,0])
                NDEGREE = int(self.Variables.VARPARAM[ivar,1])
                NWINDOWS = int(self.Variables.VARPARAM[ivar,2])

                lowin = np.zeros(NWINDOWS)
                hiwin = np.zeros(NWINDOWS)
                i0 = 0
                for IWIN in range(NWINDOWS):
                    lowin[IWIN] = float(self.Variables.VARPARAM[ivar,3+i0])
                    i0 = i0 + 1
                    hiwin[IWIN] = float(self.Variables.VARPARAM[ivar,3+i0])
                    i0 = i0 + 1

                for IWIN in range(NWINDOWS):

                    ivin = np.where( (self.MeasurementX.WAVE>=lowin[IWIN]) & (self.MeasurementX.WAVE<hiwin[IWIN]) )[0]
                    nvin = len(ivin)

                    for i in range(self.MeasurementX.NGEOM):

                        #Getting the coefficients
                        T = np.zeros(NDEGREE+1)
                        for j in range(NDEGREE+1):
                            T[j] = self.Variables.XN[ix+j]

                        WAVE0 = self.MeasurementX.WAVE[ivin].min()
                        spec = np.zeros(nvin)
                        spec[:] = SPECMOD[ivin,i]

                        #Changing the state vector based on this parameterisation
                        POL = np.zeros(nvin)
                        for j in range(NDEGREE+1):
                            POL[:] = POL[:] + T[j]*(self.MeasurementX.WAVE[ivin]-WAVE0)**j

                        SPECMOD[ivin,i] = SPECMOD[ivin,i] * POL[:]

                        #Changing the rest of the gradients based on the impact of this parameterisation
                        for ixn in range(self.Variables.NX):
                            dSPECMOD[ivin,i,ixn] = dSPECMOD[ivin,i,ixn] * POL[:]

                        #Defining the analytical gradients for this parameterisation
                        for j in range(NDEGREE+1):
                            dSPECMOD[ivin,i,ix+j] = spec[:] * (self.Measurement.WAVE[ivin]-WAVE0)**j

                        ix = ix + (NDEGREE+1)

            elif self.Variables.VARIDENT[ivar,0]==232:
#           Model 232. Continuum addition to transmission spectra using the angstrom coefficient
#           ***************************************************************

                #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
                #Where the parameters to fit are TAU0 and ALPHA

                #The effect of this model takes place after the computation of the spectra in CIRSrad!
                if int(self.Variables.NXVAR[ivar]/2)!=self.MeasurementX.NGEOM:
                    sys.exit('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

                if self.MeasurementX.NGEOM>1:

                    for i in range(self.MeasurementX.NGEOM):
                        TAU0 = self.Variables.XN[ix]
                        ALPHA = self.Variables.XN[ix+1]
                        WAVE0 = self.Variables.VARPARAM[ivar,1]

                        spec = np.zeros(self.MeasurementX.NWAVE)
                        spec[:] = SPECMOD[:,i]

                        #Changing the state vector based on this parameterisation
                        SPECMOD[:,i] = SPECMOD[:,i] * np.exp ( -TAU0 * (self.MeasurementX.WAVE/WAVE0)**(-ALPHA) )

                        #Changing the rest of the gradients based on the impact of this parameterisation
                        for ixn in range(self.Variables.NX):
                            dSPECMOD[:,i,ixn] = dSPECMOD[:,i,ixn] * np.exp ( -TAU0 * (self.MeasurementX.WAVE/WAVE0)**(-ALPHA) )

                        #Defining the analytical gradients for this parameterisation
                        dSPECMOD[:,i,ix] = spec[:] * ( -((self.MeasurementX.WAVE/WAVE0)**(-ALPHA)) * np.exp ( -TAU0 * (self.MeasurementX.WAVE/WAVE0)**(-ALPHA) ) )
                        dSPECMOD[:,i,ix+1] = spec[:] * TAU0 * np.exp ( -TAU0 * (self.MeasurementX.WAVE/WAVE0)**(-ALPHA) ) * np.log(self.MeasurementX.WAVE/WAVE0) * (self.MeasurementX.WAVE/WAVE0)**(-ALPHA)

                        ix = ix + 2

                else:


                    T0 = self.Variables.XN[ix]
                    ALPHA = self.Variables.XN[ix+1]
                    WAVE0 = self.Variables.VARPARAM[ivar,1]

                    """
                    spec = np.zeros(Measurement.NWAVE)
                    spec[:] = SPECMOD
                    SPECMOD[:] = SPECMOD[:] * ( T0*(Measurement.WAVE/WAVE0)**(-ALPHA) )
                    for ixn in range(Variables.NX):
                        dSPECMOD[:,ixn] = dSPECMOD[:,ixn] * ( T0*(Measurement.WAVE/WAVE0)**(-ALPHA) )

                    #Defining the analytical gradients for this parameterisation
                    dSPECMOD[:,ix] = spec * ((Measurement.WAVE/WAVE0)**(-ALPHA))
                    dSPECMOD[:,ix+1] = -spec * T0 * np.log(Measurement.WAVE/WAVE0) * (Measurement.WAVE/WAVE0)**(-ALPHA)
                    """

                    ix = ix + 2

            elif self.Variables.VARIDENT[ivar,0]==233:
#           Model 232. Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)
#           ***************************************************************

                #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
                #Where the aerosol opacity is modelled following

                # np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

                #The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
                #233 converges to model 232 when a2=0.

                #The effect of this model takes place after the computation of the spectra in CIRSrad!
                if int(self.Variables.NXVAR[ivar]/3)!=self.MeasurementX.NGEOM:
                    sys.exit('error using Model 233 :: The number of levels for the addition of continuum must be the same as NGEOM')

                if self.MeasurementX.NGEOM>1:

                    for i in range(self.MeasurementX.NGEOM):

                        A0 = self.Variables.XN[ix]
                        A1 = self.Variables.XN[ix+1]
                        A2 = self.Variables.XN[ix+2]

                        spec = np.zeros(self.MeasurementX.NWAVE)
                        spec[:] = SPECMOD[:,i]

                        #Calculating the aerosol opacity at each wavelength
                        TAU = np.exp(A0 + A1 * np.log(self.MeasurementX.WAVE) + A2 * np.log(self.MeasurementX.WAVE)**2.)

                        #Changing the state vector based on this parameterisation
                        SPECMOD[:,i] = SPECMOD[:,i] * np.exp ( -TAU )

                        #Changing the rest of the gradients based on the impact of this parameterisation
                        for ixn in range(self.Variables.NX):
                            dSPECMOD[:,i,ixn] = dSPECMOD[:,i,ixn] * np.exp ( -TAU )

                        #Defining the analytical gradients for this parameterisation
                        dSPECMOD[:,i,ix] = spec[:] * (-TAU) * np.exp(-TAU)
                        dSPECMOD[:,i,ix+1] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(self.MeasurementX.WAVE)
                        dSPECMOD[:,i,ix+2] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(self.MeasurementX.WAVE)**2.

                        ix = ix + 3

                else:

                    A0 = self.Variables.XN[ix]
                    A1 = self.Variables.XN[ix+1]
                    A2 = self.Variables.XN[ix+2]

                    #Getting spectrum
                    spec = np.zeros(self.MeasurementX.NWAVE)
                    spec[:] = SPECMOD

                    #Calculating aerosol opacity
                    TAU = np.exp(A0 + A1 * np.log(self.MeasurementX.WAVE) + A2 * np.log(self.MeasurementX.WAVE)**2.)

                    SPECMOD[:] = SPECMOD[:] * np.exp(-TAU)
                    for ixn in range(self.Variables.NX):
                        dSPECMOD[:,ixn] = dSPECMOD[:,ixn] * np.exp(-TAU)

                    #Defining the analytical gradients for this parameterisation
                    dSPECMOD[:,ix] = spec[:] * (-TAU) * np.exp(-TAU)
                    dSPECMOD[:,ix+1] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(self.MeasurementX.WAVE)
                    dSPECMOD[:,ix+2] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(self.MeasurementX.WAVE)**2.

                    ix = ix + 3

            elif self.Variables.VARIDENT[ivar,0]==667:
#           Model 667. Spectrum scaled by dilution factor to account for thermal gradients in planets
#           **********************************************************************************************

                xfactor = self.Variables.XN[ix]
                spec = np.zeros(self.MeasurementX.NWAVE)
                spec[:] = SPECMOD
                SPECMOD = model667(SPECMOD,xfactor)
                dSPECMOD = dSPECMOD * xfactor
                dSPECMOD[:,ix] = spec[:]
                ix = ix + 1

            else:
                ix = ix + self.Variables.NXVAR[ivar]

        return SPECMOD,dSPECMOD


    ###############################################################################################

    def subspeconv(self,SPECMOD,dSPECONV):

        """
        FUNCTION NAME : subspeconv()

        DESCRIPTION : Calculate the gradients for any model parameterisation that involves the convolution
                       of the modelled spectrum with the instrument lineshape. These parameterisations can 
                       include for example the retrieval of the spectral resolution of the instrument function,
                       in case it is not well characterised. 

        INPUTS :

            Measurement :: Python class defining the observation
            Variables :: Python class defining the parameterisations and state vector
            SPECMOD(NWAVE,NGEOM) :: Modelled spectrum in each geometry (not yet convolved with ILS)
            dSPECONV(NCONV,NGEOM,NX) :: Modelled gradients in each geometry (previously convolved with ILS)

        OPTIONAL INPUTS: none

        OUTPUTS :

            dSPECONV :: Updated gradients in each geometry

        CALLING SEQUENCE:

            SPECONV = subspecret(Measurement,Variables,SPECMOD,dSPECONV)

        MODIFICATION HISTORY : Juan Alday (15/07/2022)

        """

        #Going through the different variables an updating the spectra and gradients accordingly
        ix = 0
        for ivar in range(self.Variables.NVAR):

            if self.Variables.VARIDENT[ivar,0]==229:
#           Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
#           ***************************************************************

                #Getting the reference values for the ILS parameterisation
                par1 = self.Variables.XN[ix]
                par2 = self.Variables.XN[ix+1]
                par3 = self.Variables.XN[ix+2]
                par4 = self.Variables.XN[ix+3]
                par5 = self.Variables.XN[ix+4]
                par6 = self.Variables.XN[ix+5]
                par7 = self.Variables.XN[ix+6]

                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par5,par6,par7)

                #Performing first convolution of the spectra
                SPECONV_ref = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')

                #Going through each of the parameters to calculate the gradients

                par11 = self.Variables.XN[ix]*1.05
                self.MeasurementX = model229(self.MeasurementX,par11,par2,par3,par4,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix] = (SPECONV1-SPECONV_ref)/(par11-par1)

                par21 = self.Variables.XN[ix+1]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par21,par3,par4,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+1] = (SPECONV1-SPECONV_ref)/(par21-par2)

                par31 = self.Variables.XN[ix+2]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par31,par4,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+2] = (SPECONV1-SPECONV_ref)/(par31-par3)

                par41 = self.Variables.XN[ix+3]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par41,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+3] = (SPECONV1-SPECONV_ref)/(par41-par4)

                par51 = self.Variables.XN[ix+4]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par51,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+4] = (SPECONV1-SPECONV_ref)/(par51-par5)

                par61 = self.Variables.XN[ix+5]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par5,par61,par7)
                SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+5] = (SPECONV1-SPECONV_ref)/(par61-par6)

                par71 = self.Variables.XN[ix+6]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par5,par6,par71)
                SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+6] = (SPECONV1-SPECONV_ref)/(par71-par7)

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==230:
#           Model 230. Retrieval of multiple instrument line shapes for ACS-MIR (multiple spectral windows)
#           ***************************************************************

                #Getting reference values and calculating the reference convolved spectrum
                nwindows = int(self.Variables.VARPARAM[ivar,0])
                liml = np.zeros(nwindows)
                limh = np.zeros(nwindows)
                i0 = 1
                for iwin in range(nwindows):
                    liml[iwin] = self.Variables.VARPARAM[ivar,i0]
                    limh[iwin] = self.Variables.VARPARAM[ivar,i0+1]
                    i0 = i0 + 2

                par1 = np.zeros((7,nwindows))
                il = 0
                for iwin in range(nwindows):
                    for jwin in range(7):
                        par1[jwin,iwin] = self.Variables.XN[ix+il]
                        il = il + 1

                self.MeasurementX = model230(self.MeasurementX,nwindows,liml,limh,par1)

                #Performing first convolution of the spectra
                SPECONV_ref = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')

                il = 0
                for iwin in range(nwindows):
                    for jwin in range(7):
                        par2 = np.zeros(par1.shape)
                        par2[:,:] = par1[:,:]
                        par2[jwin,iwin] = par1[jwin,iwin] * 1.05

                        self.MeasurementX = model230(self.MeasurementX,nwindows,liml,limh,par2)

                        SPECONV1 = self.MeasurementX.lblconv(SPECMOD,IGEOM='All')
                        dSPECONV[:,:,ix+il] = (SPECONV1-SPECONV_ref)/(par2[jwin,iwin]-par1[jwin,iwin])

                        il = il + 1

                ix = ix + self.Variables.NXVAR[ivar]

            else:
                ix = ix + self.Variables.NXVAR[ivar]

        return dSPECONV

    ###############################################################################################
    ###############################################################################################
    # PATH CALCULATION AND DEFINITION OF GEOMETRY
    ###############################################################################################
    ###############################################################################################


    ###############################################################################################

    def select_Measurement(self,IGEOM,IAV):

        """
            FUNCTION NAME : select_Measurement()

            DESCRIPTION : This function fills the Measurement1 class with the information about
                          a specific measurement that wants to be modelled

            INPUTS :

                IGEOM :: Integer defining the number of the geometry (from 0 to NGEOM - 1)
                IAV :: Integer defining the number of the averaging point for the geometry (from 0 to NAV(IGEOM))

            OPTIONAL INPUTS: none

            OUTPUTS :

                Updated Measurement1 class

            CALLING SEQUENCE:

                ForwardModel.select_Measurement(IGEOM,IAV)

            MODIFICATION HISTORY : Juan Alday (25/08/2022)

        """

        self.MeasurementX.NGEOM = 1
        self.MeasurementX.NAV = np.ones(self.MeasurementX.NGEOM,dtype='int32')
        self.MeasurementX.FWHM = self.Measurement.FWHM
        self.MeasurementX.IFORM = self.Measurement.IFORM
        self.MeasurementX.ISPACE = self.Measurement.ISPACE

        self.MeasurementX.NCONV = np.zeros(self.MeasurementX.NGEOM,dtype='int32')
        self.MeasurementX.NCONV[0] = self.Measurement.NCONV[IGEOM]

        self.MeasurementX.VCONV = np.zeros((self.MeasurementX.NCONV[0],self.MeasurementX.NGEOM))
        self.MeasurementX.VCONV[:,0] = self.Measurement.VCONV[0:self.MeasurementX.NCONV[0],IGEOM]

        self.MeasurementX.FLAT = np.zeros((self.MeasurementX.NGEOM,self.MeasurementX.NAV[0]))
        self.MeasurementX.FLON = np.zeros((self.MeasurementX.NGEOM,self.MeasurementX.NAV[0]))
        self.MeasurementX.WGEOM = np.zeros((self.MeasurementX.NGEOM,self.MeasurementX.NAV[0]))
        self.MeasurementX.SOL_ANG = np.zeros((self.MeasurementX.NGEOM,self.MeasurementX.NAV[0]))
        self.MeasurementX.EMISS_ANG = np.zeros((self.MeasurementX.NGEOM,self.MeasurementX.NAV[0]))
        self.MeasurementX.AZI_ANG = np.zeros((self.MeasurementX.NGEOM,self.MeasurementX.NAV[0]))
        self.MeasurementX.FLAT[0,0] = self.Measurement.FLAT[IGEOM,IAV]
        self.MeasurementX.FLON[0,0] = self.Measurement.FLON[IGEOM,IAV]
        self.MeasurementX.WGEOM[0,0] = self.Measurement.WGEOM[IGEOM,IAV]
        self.MeasurementX.AZI_ANG[0,0] = self.Measurement.AZI_ANG[IGEOM,IAV]
        self.MeasurementX.SOL_ANG[0,0] = self.Measurement.SOL_ANG[IGEOM,IAV]
        self.MeasurementX.EMISS_ANG[0,0] = self.Measurement.EMISS_ANG[IGEOM,IAV]

        self.MeasurementX.LATITUDE = self.MeasurementX.FLAT[0,0]
        self.MeasurementX.LONGITUDE = self.MeasurementX.FLON[0,0]


    ###############################################################################################

    def calc_path(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_path()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                    different parameters in the Path class are changed to perform correctly
                    the radiative transfer calculations

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_path(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from NemesisPy.Path import AtmCalc_0,Path_0


        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        LAYANG = 0.0
        if Scatter.EMISS_ANG<0.0:
            Layer.LAYHT = Scatter.SOL_ANG * 1.0e3
            LAYANG = 90.0

        BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH\
            = Layer.integrate(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)

        #Setting the flags for the Path and calculation types
        ##############################################################################

        limb = False
        nadir = False
        ipzen = 0
        therm = False
        wf = False
        netflux = False
        outflux = False
        botflux = False
        upflux = False
        cg = False
        hemisphere = False
        nearlimb = False
        single = False
        sphsingle = False
        scatter = False
        broad = False
        absorb = False
        binbb = True

        if Scatter.EMISS_ANG>=0.0:
            limb=False
            nadir=True
            angle=Scatter.EMISS_ANG
            botlay=0
        else:
            nadir=False
            limb=True
            angle=90.0
            botlay=0

        if Scatter.ISCAT==0:   #No scattering
            if Measurement.IFORM==4:  #Atmospheric transmission multiplied by solar flux (no thermal emission then)
                therm=False
            else:
                therm=True
            scatter=False
        elif Scatter.ISCAT==1: #Multiple scattering
            therm=False
            scatter=True
        elif Scatter.ISCAT==2: #Internal scattered radiation field
            therm=False
            scatter=True
            nearlimb=True
        elif Scatter.ISCAT==3: #Single scattering in plane-parallel atmosphere
            therm=False
            single=True
        elif Scatter.ISCAT==4: #Single scattering in spherical atmosphere
            therm=False
            sphsingle=True


        #Performing the calculation of the atmospheric path
        ##############################################################################

        #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
        NCALC = 1    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        iAtmCalc = AtmCalc_0(Layer,LIMB=limb,NADIR=nadir,BOTLAY=botlay,ANGLE=angle,IPZEN=ipzen,\
                         THERM=therm,WF=wf,NETFLUX=netflux,OUTFLUX=outflux,BOTFLUX=botflux,UPFLUX=upflux,\
                         CG=cg,HEMISPHERE=hemisphere,NEARLIMB=nearlimb,SINGLE=single,SPHSINGLE=sphsingle,\
                         SCATTER=scatter,BROAD=broad,ABSORB=absorb,BINBB=binbb)
        AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)

    ###############################################################################################

    def calc_pathg(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_path()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                    different parameters in the Path class are changed to perform correctly
                    the radiative transfer calculations. This version also computes the matrices relating
                    the properties of each layer (Layer) with the properties of the input profiles (Atmosphere)

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_pathg(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from NemesisPy.Path import AtmCalc_0,Path_0
        import numpy as np

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        LAYANG = 0.0
        if Scatter.EMISS_ANG<0.0:
            Layer.LAYHT = Scatter.SOL_ANG * 1.0e3
            LAYANG = 90.0

        BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH, DTE, DAM, DCO\
            = Layer.integrateg(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)

        #Setting the flags for the Path and calculation types
        ##############################################################################

        limb = False
        nadir = False
        ipzen = 0
        therm = False
        wf = False
        netflux = False
        outflux = False
        botflux = False
        upflux = False
        cg = False
        hemisphere = False
        nearlimb = False
        single = False
        sphsingle = False
        scatter = False
        broad = False
        absorb = False
        binbb = True

        if Scatter.EMISS_ANG>=0.0:
            limb=False
            nadir=True
            angle=Scatter.EMISS_ANG
            botlay=0
        else:
            nadir=False
            limb=True
            angle=90.0
            botlay=0

        if Scatter.ISCAT==0:   #No scattering
            if Measurement.IFORM==4:  #Atmospheric transmission multiplied by solar flux (no thermal emission then)
                therm=False
            else:
                therm=True
            scatter=False
        elif Scatter.ISCAT==1: #Multiple scattering
            therm=False
            scatter=True
        elif Scatter.ISCAT==2: #Internal scattered radiation field
            therm=False
            scatter=True
            nearlimb=True
        elif Scatter.ISCAT==3: #Single scattering in plane-parallel atmosphere
            therm=False
            single=True
        elif Scatter.ISCAT==4: #Single scattering in spherical atmosphere
            therm=False
            sphsingle=True


        #Performing the calculation of the atmospheric path
        ##############################################################################

        #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
        NCALC = 1    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        iAtmCalc = AtmCalc_0(Layer,LIMB=limb,NADIR=nadir,BOTLAY=botlay,ANGLE=angle,IPZEN=ipzen,\
                         THERM=therm,WF=wf,NETFLUX=netflux,OUTFLUX=outflux,BOTFLUX=botflux,UPFLUX=upflux,\
                         CG=cg,HEMISPHERE=hemisphere,NEARLIMB=nearlimb,SINGLE=single,SPHSINGLE=sphsingle,\
                         SCATTER=scatter,BROAD=broad,ABSORB=absorb,BINBB=binbb)
        AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)

    ###############################################################################################

    def calc_path_SO(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_path_SO()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                      different parameters in the Path class are changed to perform correctly
                      the radiative transfer calculations

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            ForwardModel.calc_path_SO()

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from NemesisPy.Path import AtmCalc_0,Path_0
        from NemesisPy import find_nearest

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        #Based on the new reference atmosphere, we split the atmosphere into layers
        #In solar occultation LAYANG = 90.0
        LAYANG = 90.0

        BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH\
            = Layer.integrate(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)

        #Based on the atmospheric layerinc, we calculate each required atmospheric path to model the measurements
        #############################################################################################################

        #Calculating the required paths that need to be calculated
        ITANHE = []
        for igeom in range(Measurement.NGEOM):

            base0,ibase = find_nearest(Layer.BASEH/1.0e3,Measurement.TANHE[igeom])
            if base0<=Measurement.TANHE[igeom]:
                ibasel = ibase
                ibaseh = ibase + 1
                if ibaseh==Layer.NLAY:
                    ibaseh = ibase
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            ITANHE.append(ibasel)
            ITANHE.append(ibaseh)

        ITANHE = np.unique(ITANHE)

        NCALC = len(ITANHE)    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        for ICALC in range(NCALC):
            iAtmCalc = AtmCalc_0(Layer,LIMB=True,BOTLAY=ITANHE[ICALC],ANGLE=90.0,IPZEN=0,THERM=False)
            AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)


    ###############################################################################################

    def calc_pathg_SO(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_pathg_SO()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                  different parameters in the Path class are changed to perform correctly
                  the radiative transfer calculations. This version also computes the matrices relating
                  the properties of each layer (Layer) with the properties of the input profiles (Atmosphere)

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_pathg(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from NemesisPy.Path import AtmCalc_0,Path_0
        from NemesisPy import find_nearest

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX


        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        #Based on the new reference atmosphere, we split the atmosphere into layers
        #In solar occultation LAYANG = 90.0
        LAYANG = 90.0

        BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH, DTE, DAM, DCO\
            = Layer.integrateg(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)

        #Based on the atmospheric layerinc, we calculate each required atmospheric path to model the measurements
        #############################################################################################################

        #Calculating the required paths that need to be calculated
        ITANHE = []
        for igeom in range(Measurement.NGEOM):

            base0,ibase = find_nearest(Layer.BASEH/1.0e3,Measurement.TANHE[igeom])
            if base0<=Measurement.TANHE[igeom]:
                ibasel = ibase
                ibaseh = ibase + 1
                if ibaseh==Layer.NLAY:
                    ibaseh = ibase
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            ITANHE.append(ibasel)
            ITANHE.append(ibaseh)

        ITANHE = np.unique(ITANHE)

        NCALC = len(ITANHE)    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        for ICALC in range(NCALC):
            iAtmCalc = AtmCalc_0(Layer,LIMB=True,BOTLAY=ITANHE[ICALC],ANGLE=90.0,IPZEN=0,THERM=False)
            AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)



    ###############################################################################################
    ###############################################################################################
    # RADIATIVE TRANSFER
    ###############################################################################################
    ###############################################################################################


    ###############################################################################################

    def CIRSrad(self):

        """
            FUNCTION NAME : CIRSrad()

            DESCRIPTION : This function computes the spectrum given the calculation type

            INPUTS :

                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations
                Path :: Python class defining the calculation type and the path

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECOUT(Measurement.NWAVE,Path.NPATH) :: Output spectrum (non-convolved) in the units given by IMOD

            CALLING SEQUENCE:

                SPECOUT = CIRSrad(Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        import matplotlib as matplotlib
        from scipy import interpolate
        from NemesisPy import k_overlap, k_overlapg, planck
        from copy import copy

        #Initialise some arrays
        ###################################

        #Initialise the inputs
        Measurement=self.MeasurementX
        Atmosphere=self.AtmosphereX
        Spectroscopy=self.SpectroscopyX
        Scatter=self.ScatterX
        Stellar=self.StellarX
        Surface=self.SurfaceX
        CIA=self.CIAX
        Layer=self.LayerX
        Path=self.PathX

        #Calculating the vertical opacity of each layer
        ######################################################
        ######################################################
        ######################################################
        ######################################################

        #There will be different kinds of opacities:
        #   Continuum opacity due to aerosols coming from the extinction coefficient
        #   Continuum opacity from different gases like H, NH3 (flags in .fla file)
        #   Collision-Induced Absorption
        #   Scattering opacity derived from the particle distribution and the single scattering albedo.
        #        For multiple scattering, this is passed to scattering routines
        #   Line opacity due to gaseous absorption (K-tables or LBL-tables)


        #Calculating the continuum absorption by gaseous species
        #################################################################################################################

        #Computes a polynomial approximation to any known continuum spectra for a particular gas over a defined wavenumber region.

        #To be done

        #Calculating the vertical opacity by CIA
        #################################################################################################################


        if CIA==None:
            TAUCIA = np.zeros((Measurement.NWAVE,Layer.NLAY))
            dTAUCIA = np.zeros((Measurement.NWAVE,Layer.NLAY,7))
            print('CIRSrad :: CIA not included in calculations')
        else:
            TAUCIA,dTAUCIA,IABSORB = self.calc_tau_cia() #(NWAVE,NLAY);(NWAVE,NLAY,7)

        #Calculating the vertical opacity by Rayleigh scattering
        #################################################################################################################

        if Scatter.IRAY==0:
            TAURAY = np.zeros((Measurement.NWAVE,Layer.NLAY))
        elif Scatter.IRAY==1:
            TAURAY,dTAURAY = Scatter.calc_tau_rayleighj(Measurement.ISPACE,
                Measurement.WAVE,Layer) #(NWAVE,NLAY)
        else:
            sys.exit('error in CIRSrad :: IRAY type has not been implemented yet')

        #Calculating the vertical opacity by aerosols from the extinction coefficient and single scattering albedo
        #################################################################################################################

        #Obtaining the phase function of each aerosol at the scattering angle
        if Scatter.ISCAT>0:
            sol_ang = Scatter.SOL_ANG
            emiss_ang = Scatter.EMISS_ANG
            azi_ang = Scatter.AZI_ANG

            #Calculating cos(alpha), where alpha is the scattering angle
            calpha = np.sin(sol_ang / 180. * np.pi) * np.sin(emiss_ang / 180. * np.pi) * np.cos( azi_ang/180.*np.pi - np.pi ) - \
                     np.cos(emiss_ang / 180. * np.pi) * np.cos(sol_ang / 180. * np.pi)

            if Scatter.ISCAT==3: #NEEDS TO BE FINISHED!
                phasef = np.zeros(Scatter.NDUST+1)   #Phase angle for each aerosol type and for Rayleigh scattering
                phasef[Scatter.NDUST] = 0.75 * (1. + calpha**2.)  #Phase function for Rayleigh scattering (Hansen and Travis, 1974)


        TAUDUST1,TAUCLSCAT,dTAUDUST1,dTAUCLSCAT = self.calc_tau_dust() #(NWAVE,NLAYER,NDUST)

        #Adding the opacity by the different dust populations
        TAUDUST = np.sum(TAUDUST1,2)  #(NWAVE,NLAYER) Absorption + Scattering
        TAUSCAT = np.sum(TAUCLSCAT,2)  #(NWAVE,NLAYER) Scattering

        del TAUDUST1

        #Calculating the gaseous line opacity in each layer
        ########################################################################################################

        if Spectroscopy.ILBL==2:  #LBL-table

            TAUGAS = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Layer.NLAY,Spectroscopy.NGAS])  #Vertical opacity of each gas in each layer

            #Calculating the cross sections for each gas in each layer
            k = Spectroscopy.calc_klbl(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE)

            for i in range(Spectroscopy.NGAS):
                IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
                IGAS = IGAS[0]

                #Calculating vertical column density in each layer
                VLOSDENS = Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #cm-2

                #Calculating vertical opacity for each gas in each layer
                TAUGAS[:,0,:,i] = k[:,:,i] * VLOSDENS

            #Combining the gaseous opacity in each layer
            TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)

            del k

        elif Spectroscopy.ILBL==0:    #K-table

            #Calculating the k-coefficients for each gas in each layer
            k_gas,dkgasdT = Spectroscopy.calc_kg(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE) # (NWAVE,NG,NLAY,NGAS)

            f_gas = np.zeros([Spectroscopy.NGAS,Layer.NLAY])
            utotl = np.zeros(Layer.NLAY)
            for i in range(Spectroscopy.NGAS):
                IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
                IGAS = IGAS[0]

                #When using gradients
                f_gas[i,:] = Layer.AMOUNT[:,IGAS[0]] * 1.0e-4 * 1.0e-20  #Vertical column density of the radiatively active gases in cm-2

            #f_gas = np.zeros([Spectroscopy.NGAS,Layer.NLAY])
            #utotl = np.zeros(Layer.NLAY)
            #for i in range(Spectroscopy.NGAS):
            #    IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
            #    IGAS = IGAS[0]

            #    f_gas[i,:] = Layer.PP[:,IGAS].T / Layer.PRESS                     #VMR of each radiatively active gas
            #    utotl[:] = utotl[:] + Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #Vertical column density of the radiatively active gases

            #Combining the k-distributions of the different gases in each layer
            k_layer,dk_layer = k_overlapg(Measurement.NWAVE,Spectroscopy.NG,Spectroscopy.DELG,Spectroscopy.NGAS,Layer.NLAY,k_gas,dkgasdT,f_gas)

            #Calculating the opacity of each layer
            TAUGAS = k_layer #(NWAVE,NG,NLAY)
            #Calculating the opacity of each layer
            #TAUGAS = k_layer * utotl   #(NWAVE,NG,NLAY)

            del k_gas
            del k_layer

        else:
            sys.exit('error in CIRSrad :: ILBL must be either 0 or 2')


        #Combining the different kinds of opacity in each layer
        ########################################################################################################

        TAUTOT = np.zeros(TAUGAS.shape) #(NWAVE,NG,NLAY)
        for ig in range(Spectroscopy.NG):
            TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]

        #Calculating the line-of-sight opacities
        #################################################################################################################

        TAUTOT_LAYINC = TAUTOT[:,:,Path.LAYINC[:,:]] * Path.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)


        #Step through the different number of paths and calculate output spectrum
        ############################################################################

        #Output paths may be:
        #	      Imod
        #		0	(Atm) Pure transmission
        #		1	(Atm) Absorption (useful for small transmissions)
        #		2	(Atm) Emission. Planck function evaluated at each
        #				wavenumber. NOT SUPPORTED HERE.
        #		3	(Atm) Emission. Planck function evaluated at bin
        #				center.
        #		8	(Combined Cell,Atm) The product of two
        #				previous output paths.
        #		11	(Atm) Contribution function.
        #		13	(Atm) SCR Sideband
        #		14	(Atm) SCR Wideband
        #		15	(Atm) Multiple scattering (multiple models)
        #		16	(Atm) Single scattering approximation.
        #		21	(Atm) Net flux calculation (thermal)
        #		22	(Atm) Limb scattering calculation
        #		23	(Atm) Limb scattering calculation using precomputed
        #			      internal radiation field.
        #		24	(Atm) Net flux calculation (scattering)
        #		25	(Atm) Upwards flux (internal) calculation (scattering)
        #		26	(Atm) Upwards flux (top) calculation (scattering)
        #		27	(Atm) Downwards flux (bottom) calculation (scattering)
        #		28	(Atm) Single scattering approximation (spherical)

        IMODM = np.unique(Path.IMOD)

        if IMODM==0:

            #Calculating the total opacity over the path
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

            #Pure transmission spectrum
            SPECOUT = np.exp(-(TAUTOT_PATH))  #(NWAVE,NG,NPATH)

            xfac = 1.0
            if Measurement.IFORM==4:  #If IFORM=4 we should multiply the transmission by solar flux
                Stellar.calc_solar_flux()
                #Interpolating to the calculation wavelengths
                f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLFLUX)
                solflux = f(Measurement.WAVE)
                xfac = solflux
                for ipath in range(Path.NPATH):
                    for ig in range(Spectroscopy.NG):
                        SPECOUT[:,ig,ipath] = SPECOUT[:,ig,ipath] * xfac

        elif IMODM==1:

            #Calculating the total opacity over the path
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

            #Absorption spectrum (useful for small transmissions)
            SPECOUT = 1.0 - np.exp(-(TAUTOT_PATH)) #(NWAVE,NG,NPATH)

        elif IMODM==3: #Thermal emission from planet

            SPECOUT = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Path.NPATH])

            #Defining the units of the output spectrum
            xfac = 1.
            if Measurement.IFORM==1:
                xfac=np.pi*4.*np.pi*((Atmosphere.RADIUS)*1.0e2)**2.
                f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLSPEC)
                solpspec = f(Measurement.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
                xfac = xfac / solpspec

            #Calculating spectrum
            for ipath in range(Path.NPATH):


                #Calculating atmospheric contribution
                taud = np.zeros([Measurement.NWAVE,Spectroscopy.NG])
                trold = np.ones([Measurement.NWAVE,Spectroscopy.NG])
                specg = np.zeros([Measurement.NWAVE,Spectroscopy.NG])

                for j in range(Path.NLAYIN[ipath]):

                    taud[:,:] = taud[:,:] + TAUTOT_LAYINC[:,:,j,ipath]
                    tr = np.exp(-taud)

                    bb = planck(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[j,ipath])
                    for ig in range(Spectroscopy.NG):
                        specg[:,ig] = specg[:,ig] + (trold[:,ig]-tr[:,ig])*bb[:] * xfac

                    trold = copy(tr)

                #Calculating surface contribution

                p1 = Layer.PRESS[Path.LAYINC[int(Path.NLAYIN[ipath]/2)-1,ipath]]
                p2 = Layer.PRESS[Path.LAYINC[int(Path.NLAYIN[ipath]-1),ipath]]

                if p2>p1:  #If not limb path, we add the surface contribution

                    if Surface.TSURF<=0.0:
                        radground = planck(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[Path.NLAYIN[ipath]-1,ipath])
                    else:
                        bbsurf = planck(Measurement.ISPACE,Measurement.WAVE,Surface.TSURF)

                        f = interpolate.interp1d(Surface.VEM,Surface.EMISSIVITY)
                        emissivity = f(Measurement.WAVE)

                        radground = bbsurf * emissivity

                    for ig in range(Spectroscopy.NG):
                        specg[:,ig] = specg[:,ig] + trold[:,ig] * radground[:] * xfac

                SPECOUT[:,:,ipath] = specg[:,:]

        elif IMODM==15: #Multiple scattering calculation

            #Calculating the thermal emission of each atmospheric layer
            bb = planck(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[0:Path.NLAYIN[ipath],ipath])  #(NWAVE,NLAYIN)


            #Calculating the surface properties at each wavelength (emissivity, albedo and thermal emission)
            radground = np.zeros(Measurement.NWAVE)
            galb = np.zeros(Measurement.NWAVE)
            if Surface.TSURF<=0.0:  #No surface
                radground = planck(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[Path.NLAYIN[ipath]-1,ipath])
            else:
                #Calculating the blackbody at given temperature
                bbsurf = planck(Measurement.ISPACE,Measurement.WAVE,Surface.TSURF)

                #Calculating the emissivity
                f = interpolate.interp1d(Surface.VEM,Surface.EMISSIVITY)
                emissivity = f(Measurement.WAVE)

                #Calculating thermal emission from surface
                radground = bbsurf * emissivity

                #Calculating ground albedo
                if Surface.GALB<0.0:
                    galb[:] = 1.0 - emissivity[:]
                else:
                    galb[:] = Surface.GALB
            

            #Calculating the single scattering albedo of each layer
            # and the contribution from each aerosol species to the total 
            omega = np.zeros((Measurement.NWAVE,Path.NLAYIN[ipath]))  #Single scattering albedo
            eps = np.ones((Measurement.NWAVE,Path.NLAYIN[ipath]))
            lfrac = np.zeros((Measurement.NWAVE,Path.NLAYIN[ipath]))  #Fraction of scattering of each aerosol type to the total
            for ilay in range(Path.NLAYIN[ipath]):
                iiscat = np.where( (TAUSCAT[:,Path.LAYINC[ilay,ipath]]+TAURAY[:,Path.LAYINC[ilay,ipath]])>0.0 )[0]
                
                omega[iiscat,ilay] = (TAUSCAT[iiscat,Path.LAYINC[ilay,ipath]]+TAURAY[iiscat,Path.LAYINC[ilay,ipath]])/(TAUTOT[iiscat,Path.LAYINC[ilay,ipath]])
                lfrac[iiscat,ilay] = TAUCLSCAT[iiscat,Path.LAYINC[ilay,ipath],:] / TAUSCAT[iiscat,Path.LAYINC[ilay,ipath]]
                eps[iiscat,ilay] = 1.0 - omega[iiscat,ilay]


            #Calculating the solar flux at the top of the atmosphere
            solar = np.zeros(Measurement.NWAVE)
            if Stellar.SOLEXIST==True:
                Stellar.calc_solar_flux()
                solar[:] = Stellar.SOLFLUX   #W cm-2 (cm-1)-1 or W cm-2 um-1


            #Defining the units of the output spectrum
            xfac = 1.
            if Measurement.IFORM==1:
                xfac=np.pi*4.*np.pi*((Atmosphere.RADIUS)*1.0e2)**2.
                f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLSPEC)
                solpspec = f(Measurement.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
                xfac = xfac / solpspec
            elif Measurement.IFORM==3:
                xfac=np.pi*4.*np.pi*((Atmosphere.RADIUS)*1.0e2)**2.


            #Calculating the radiance
            SPECOUT = scloud11wave(sol_ang,emiss_ang,azi_ang)

            '''
      		  	call scloud11wave(rad1, sol_ang, emiss_ang,
     1                          aphi, radg, solar, lowbc, galb1, iray,
     2				mu1, wt1, nmu,   
     3				nf, Ig, x, vv, eps, omegas,bnu, taus, 
     4				taur,nlays, ncont,lfrac)
            '''

        else:
            sys.exit('error in CIRSrad :: Calculation type not included in CIRSrad')

        #Now integrate over g-ordinates
        SPECOUT = np.tensordot(SPECOUT, Spectroscopy.DELG, axes=([1],[0])) #NWAVE,NPATH

        return SPECOUT


    ###############################################################################################

    def CIRSradg(self):

        """
            FUNCTION NAME : CIRSradg()

            DESCRIPTION : This function computes the spectrum given the calculation type

            INPUTS :

                runname :: Name of the Nemesis run
                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations
                Path :: Python class defining the calculation type and the path

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECOUT(Measurement.NWAVE,Path.NPATH) :: Output spectrum (non-convolved) in the units given by IMOD

            CALLING SEQUENCE:

                SPECOUT = CIRSradg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        import matplotlib as matplotlib
        from scipy import interpolate
        from NemesisPy import k_overlap, k_overlapg, planck, planckg
        from copy import copy

        #Initialise some arrays
        ###################################

        #Initialise the inputs
        Measurement=self.MeasurementX
        Atmosphere=self.AtmosphereX
        Spectroscopy=self.SpectroscopyX
        Scatter=self.ScatterX
        Stellar=self.StellarX
        Surface=self.SurfaceX
        CIA=self.CIAX
        Layer=self.LayerX
        Path=self.PathX

        #Calculating the vertical opacity of each layer
        ######################################################
        ######################################################
        ######################################################
        ######################################################

        #There will be different kinds of opacities:
        #   Continuum opacity due to aerosols coming from the extinction coefficient
        #   Continuum opacity from different gases like H, NH3 (flags in .fla file)
        #   Collision-Induced Absorption
        #   Scattering opacity derived from the particle distribution and the single scattering albedo.
        #        For multiple scattering, this is passed to scattering routines
        #   Line opacity due to gaseous absorption (K-tables or LBL-tables)


        #Defining the matrices where the derivatives will be stored
        dTAUCON = np.zeros((Measurement.NWAVE,Atmosphere.NVMR+2+Scatter.NDUST,Layer.NLAY)) #(NWAVE,NLAY,NGAS+2+NDUST)
        dTAUSCA = np.zeros((Measurement.NWAVE,Atmosphere.NVMR+2+Scatter.NDUST,Layer.NLAY)) #(NWAVE,NLAY,NGAS+2+NDUST)

        #Calculating the continuum absorption by gaseous species
        #################################################################################################################

        #Computes a polynomial approximation to any known continuum spectra for a particular gas over a defined wavenumber region.

        #To be done

        #Calculating the vertical opacity by CIA
        #################################################################################################################

        if CIA==None:
            TAUCIA = np.zeros((Measurement.NWAVE,Layer.NLAY))
            dTAUCIA = np.zeros((Measurement.NWAVE,Layer.NLAY,7))
            print('CIRSrad :: CIA not included in calculations')
        else:
            print('CIRSradg :: Calculating CIA opacity')
            TAUCIA,dTAUCIA,IABSORB = self.calc_tau_cia() #(NWAVE,NLAY);(NWAVE,NLAY,7)

            for i in range(5):
                if IABSORB[i]>=0:
                    dTAUCON[:,IABSORB[i],:] = dTAUCON[:,IABSORB[i],:] + dTAUCIA[:,:,i] / (Layer.TOTAM.T) #dTAUCIA/dAMOUNT (m2)

            dTAUCON[:,Atmosphere.NVMR,:] = dTAUCON[:,Atmosphere.NVMR,:] + dTAUCIA[:,:,5]  #dTAUCIA/dT

            flagh2p = False
            if flagh2p==True:
                dTAUCON[:,Atmosphere.NVMR+1+Scatter.NDUST,:] = dTAUCON[:,Atmosphere.NVMR+1+Scatter.NDUST,:] + dTAUCIA[:,:,6]  #dTAUCIA/dPARA-H2

        #Calculating the vertical opacity by Rayleigh scattering
        #################################################################################################################

        if Scatter.IRAY==0:
            TAURAY = np.zeros([Measurement.NWAVE,Layer.NLAY])
            dTAURAY = np.zeros([Measurement.NWAVE,Layer.NLAY])
        elif Scatter.IRAY==1:
            TAURAY,dTAURAY = Scatter.calc_tau_rayleighj(Measurement.ISPACE,Measurement.WAVE,Layer) #(NWAVE,NLAY)
            for i in range(Atmosphere.NVMR):
                dTAUCON[:,i,:] = dTAUCON[:,i,:] + dTAURAY[:,:] #dTAURAY/dAMOUNT (m2)
        else:
            sys.exit('error in CIRSrad :: IRAY type has not been implemented yet')

        #Calculating the vertical opacity by aerosols from the extinction coefficient and single scattering albedo
        #################################################################################################################

        """
        #Obtaining the phase function of each aerosol at the scattering angle
        if Path.SINGLE==True:
            sol_ang = Scatter.SOL_ANG
            emiss_ang = Scatter.EMISS_ANG
            azi_ang = Scatter.AZI_ANG

            phasef = np.zeros(Scatter.NDUST+1)   #Phase angle for each aerosol type and for Rayleigh scattering

            #Calculating cos(alpha), where alpha is the scattering angle
            calpha = np.sin(sol_ang / 180. * np.pi) * np.sin(emiss_ang / 180. * np.pi) * np.cos( azi_ang/180.*np.pi - np.pi ) - \
                    np.cos(emiss_ang / 180. * np.pi) * np.cos(sol_ang / 180. * np.pi)


            phasef[Scatter.NDUST] = 0.75 * (1. + calpha**2.)  #Phase function for Rayleigh scattering (Hansen and Travis, 1974)
        """

        print('CIRSradg :: Calculating DUST opacity')
        TAUDUST1,TAUCLSCAT,dTAUDUST1,dTAUCLSCAT = self.calc_tau_dust() #(NWAVE,NLAYER,NDUST)

        #Adding the opacity by the different dust populations
        TAUDUST = np.sum(TAUDUST1,2)  #(NWAVE,NLAYER)
        TAUSCAT = np.sum(TAUCLSCAT,2)  #(NWAVE,NLAYER)

        for i in range(Scatter.NDUST):
            dTAUCON[:,Atmosphere.NVMR+1+i,:] = dTAUCON[:,Atmosphere.NVMR+1+i,:] + dTAUDUST1[:,:,i]  #dTAUDUST/dAMOUNT (m2)
            dTAUSCA[:,Atmosphere.NVMR+1+i,:] = dTAUSCA[:,Atmosphere.NVMR+1+i,:] + dTAUCLSCAT[:,:,i]

        #Calculating the gaseous line opacity in each layer
        ########################################################################################################

        print('CIRSradg :: Calculating GAS opacity')
        if Spectroscopy.ILBL==2:  #LBL-table

            TAUGAS = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Layer.NLAY,Spectroscopy.NGAS])  #Vertical opacity of each gas in each layer
            dTAUGAS = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Atmosphere.NVMR+2+Scatter.NDUST,Layer.NLAY])

            #Calculating the cross sections for each gas in each layer
            k,dkdT = Spectroscopy.calc_klblg(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE)

            for i in range(Spectroscopy.NGAS):
                IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
                IGAS = IGAS[0]

                #Calculating vertical column density in each layer
                VLOSDENS = Layer.AMOUNT[:,IGAS].T * 1.0e-20   #m-2

                #Calculating vertical opacity for each gas in each layer
                TAUGAS[:,0,:,i] = k[:,:,i] * 1.0e-4 * VLOSDENS
                dTAUGAS[:,0,IGAS[0],:] = k[:,:,i] * 1.0e-4 * 1.0e-20  #dTAUGAS/dAMOUNT (m2)
                dTAUGAS[:,0,Atmosphere.NVMR,:] = dTAUGAS[:,0,Atmosphere.NVMR,:] + dkdT[:,:,i] * 1.0e-4 * VLOSDENS #dTAUGAS/dT

            #Combining the gaseous opacity in each layer
            TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)

        elif Spectroscopy.ILBL==0:    #K-table

            dTAUGAS = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Atmosphere.NVMR+2+Scatter.NDUST,Layer.NLAY])

            #Calculating the k-coefficients for each gas in each layer
            k_gas,dkgasdT = Spectroscopy.calc_kg(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE) # (NWAVE,NG,NLAY,NGAS)

            f_gas = np.zeros([Spectroscopy.NGAS,Layer.NLAY])
            utotl = np.zeros(Layer.NLAY)
            for i in range(Spectroscopy.NGAS):
                IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
                IGAS = IGAS[0]

                #When using gradients
                f_gas[i,:] = Layer.AMOUNT[:,IGAS[0]] * 1.0e-4 * 1.0e-20  #Vertical column density of the radiatively active gases in cm-2

            #Combining the k-distributions of the different gases in each layer, as well as their gradients
            k_layer,dk_layer = k_overlapg(Measurement.NWAVE,Spectroscopy.NG,Spectroscopy.DELG,Spectroscopy.NGAS,Layer.NLAY,k_gas,dkgasdT,f_gas)

            #Calculating the opacity of each layer
            TAUGAS = k_layer #(NWAVE,NG,NLAY)

            #Calculating the gradients of each layer and for each gas
            for i in range(Spectroscopy.NGAS):
                IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
                IGAS = IGAS[0]
                dTAUGAS[:,:,IGAS[0],:] = dk_layer[:,:,:,i] * 1.0e-4 * 1.0e-20  #dTAU/dAMOUNT (m2)

            dTAUGAS[:,:,Atmosphere.NVMR,:] = dk_layer[:,:,:,Spectroscopy.NGAS] #dTAU/dT


        else:
            sys.exit('error in CIRSrad :: ILBL must be either 0 or 2')

        #Combining the different kinds of opacity in each layer
        ########################################################################################################

        print('CIRSradg :: Calculating TOTAL opacity')
        TAUTOT = np.zeros(TAUGAS.shape) #(NWAVE,NG,NLAY)
        dTAUTOT = np.zeros(dTAUGAS.shape) #(NWAVE,NG,NVMR+2+NDUST,NLAY)
        for ig in range(Spectroscopy.NG):
            TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]
            dTAUTOT[:,ig,:,:] = (dTAUGAS[:,ig,:,:] + dTAUCON[:,:,:]) #dTAU/dAMOUNT (m2) or dTAU/dK (K-1)
        del TAUGAS,TAUCIA,TAUDUST,TAURAY
        del dTAUGAS,dTAUCON

        #Calculating the line-of-sight opacities
        #################################################################################################################

        print('CIRSradg :: Calculating TOTAL line-of-sight opacity')
        TAUTOT_LAYINC = TAUTOT[:,:,Path.LAYINC[:,:]] * Path.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)
        dTAUTOT_LAYINC = dTAUTOT[:,:,:,Path.LAYINC[:,:]] * Path.SCALE[:,:] #(NWAVE,NG,NGAS+2+NDUST,NLAYIN,NPATH)


        #Step through the different number of paths and calculate output spectrum
        ############################################################################

        #Output paths may be:
        #	      Imod
        #		0	(Atm) Pure transmission
        #		1	(Atm) Absorption (useful for small transmissions)
        #		2	(Atm) Emission. Planck function evaluated at each
        #				wavenumber. NOT SUPPORTED HERE.
        #		3	(Atm) Emission. Planck function evaluated at bin
        #				center.
        #		8	(Combined Cell,Atm) The product of two
        #				previous output paths.
        #		11	(Atm) Contribution function.
        #		13	(Atm) SCR Sideband
        #		14	(Atm) SCR Wideband
        #		15	(Atm) Multiple scattering (multiple models)
        #		16	(Atm) Single scattering approximation.
        #		21	(Atm) Net flux calculation (thermal)
        #		22	(Atm) Limb scattering calculation
        #		23	(Atm) Limb scattering calculation using precomputed
        #			      internal radiation field.
        #		24	(Atm) Net flux calculation (scattering)
        #		25	(Atm) Upwards flux (internal) calculation (scattering)
        #		26	(Atm) Upwards flux (top) calculation (scattering)
        #		27	(Atm) Downwards flux (bottom) calculation (scattering)
        #		28	(Atm) Single scattering approximation (spherical)


        IMODM = np.unique(Path.IMOD)

        SPECOUT = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Path.NPATH])
        dSPECOUT = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Atmosphere.NVMR+2+Scatter.NDUST,Path.NLAYIN.max(),Path.NPATH])
        dTSURF = np.zeros((Measurement.NWAVE,Spectroscopy.NG,Path.NPATH))


        if IMODM==0:

            print('CIRSradg :: Calculating TRANSMISSION')
            #Calculating the total opacity over the path
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)
            del TAUTOT_LAYINC

            #Pure transmission spectrum
            SPECOUT = np.exp(-(TAUTOT_PATH))  #(NWAVE,NG,NPATH)
            #del TAUTOT_PATH

            xfac = np.ones(Measurement.NWAVE)
            if Measurement.IFORM==4:  #If IFORM=4 we should multiply the transmission by solar flux
                Stellar.calc_solar_flux()
                #Interpolating to the calculation wavelengths
                f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLFLUX)
                solflux = f(Measurement.WAVE)
                xfac = solflux
                for ipath in range(Path.NPATH):
                    for ig in range(Spectroscopy.NG):
                        SPECOUT[:,ig,ipath] = SPECOUT[:,ig,ipath] * xfac


            print('CIRSradg :: Calculating GRADIENTS')
            for iwave in range(Measurement.NWAVE):
                for ig in range(Spectroscopy.NG):
                    for ipath in range(Path.NPATH):
                        dSPECOUT[iwave,ig,:,:,ipath] = -SPECOUT[iwave,ig,ipath] * dTAUTOT_LAYINC[iwave,ig,:,:,ipath]
            del dTAUTOT_LAYINC
            del TAUTOT_PATH


        elif IMODM==1:

            #Calculating the total opacity over the path
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

            #Absorption spectrum (useful for small transmissions)
            SPECOUT = 1.0 - np.exp(-(TAUTOT_PATH)) #(NWAVE,NG,NPATH)


        elif IMODM==3: #Thermal emission from planet

            #Defining the units of the output spectrum
            xfac = 1.
            if Measurement.IFORM==1:
                xfac=np.pi*4.*np.pi*((Atmosphere.RADIUS)*1.0e2)**2.
                f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLSPEC)
                solpspec = f(Measurement.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
                xfac = xfac / solpspec

            #Calculating spectrum
            for ipath in range(Path.NPATH):


                #Calculating atmospheric contribution
                tlayer = np.zeros([Measurement.NWAVE,Spectroscopy.NG])
                taud = np.zeros([Measurement.NWAVE,Spectroscopy.NG])
                trold = np.ones([Measurement.NWAVE,Spectroscopy.NG])
                specg = np.zeros([Measurement.NWAVE,Spectroscopy.NG])

                dtolddq = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Atmosphere.NVMR+2+Scatter.NDUST,Path.NLAYIN.max()])
                dtrdq = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Atmosphere.NVMR+2+Scatter.NDUST,Path.NLAYIN.max()])
                dspecg = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Atmosphere.NVMR+2+Scatter.NDUST,Path.NLAYIN.max()])


                for j in range(Path.NLAYIN[ipath]):

                    tlayer[:,:] = np.exp(-TAUTOT_LAYINC[:,:,j,ipath])
                    taud[:,:] = taud[:,:] + TAUTOT_LAYINC[:,:,j,ipath]
                    tr = trold[:,:] * tlayer[:,:]

                    #Calculating the spectrum
                    bb,dBdT = planckg(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[j,ipath])
                    for ig in range(Spectroscopy.NG):
                        specg[:,ig] = specg[:,ig] + (trold[:,ig]-tr[:,ig])*bb[:] * xfac

                    #Setting up the gradients
                    for k in range(Atmosphere.NVMR+2+Atmosphere.NDUST):

                        j1 = 0
                        while j1<j:
                            dtrdq[:,:,k,j1] = dtolddq[:,:,k,j1] * tlayer[:,:]
                            for ig in range(Spectroscopy.NG):
                                dspecg[:,ig,k,j1] = dspecg[:,ig,k,j1] + (dtolddq[:,ig,k,j1]-dtrdq[:,ig,k,j1])*bb[:] * xfac
                            j1 = j1 + 1

                        tmp = dTAUTOT_LAYINC[:,:,k,j1]
                        dtrdq[:,:,k,j1] = -tmp[:,:,0] * tlayer[:,:] * trold[:,:]

                        for ig in range(Spectroscopy.NG):
                            dspecg[:,ig,k,j1] = dspecg[:,ig,k,j1] + (dtolddq[:,ig,k,j1]-dtrdq[:,ig,k,j1])*bb[:] * xfac

                        if k==Atmosphere.NVMR:
                            for ig in range(Spectroscopy.NG):
                                dspecg[:,ig,k,j] = dspecg[:,ig,k,j] + (trold[:,ig]-tr[:,ig]) * xfac * dBdT[:]


                    #Saving arrays for next iteration
                    trold = copy(tr)
                    j1 = 0
                    while j1<j:
                        dtolddq[:,:,:,j1] = dtrdq[:,:,:,j1]
                        j1 = j1 + 1
                    dtolddq[:,:,:,j1] = dtrdq[:,:,:,j1]


                #Calculating surface contribution

                p1 = Layer.PRESS[Path.LAYINC[int(Path.NLAYIN[ipath]/2)-1,ipath]]
                p2 = Layer.PRESS[Path.LAYINC[int(Path.NLAYIN[ipath]-1),ipath]]

                tempgtsurf = np.zeros((Measurement.NWAVE,Spectroscopy.NG))
                if p2>p1:  #If not limb path, we add the surface contribution

                    if Surface.TSURF<=0.0:
                        radground,dradgrounddT = planckg(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[Path.NLAYIN[ipath]-1,ipath])
                    else:
                        bbsurf,dbsurfdT = planckg(Measurement.ISPACE,Measurement.WAVE,Surface.TSURF)

                        f = interpolate.interp1d(Surface.VEM,Surface.EMISSIVITY)
                        emissivity = f(Measurement.WAVE)

                        radground = bbsurf * emissivity
                        dradgrounddT = dbsurfdT * emissivity

                    for ig in range(Spectroscopy.NG):
                        specg[:,ig] = specg[:,ig] + trold[:,ig] * radground[:] * xfac
                        tempgtsurf[:,ig] = xfac * trold[:,ig] * dradgrounddT[:]

                    for j in range(Path.NLAYIN[ipath]):
                        for k in range(Atmosphere.NVMR+2+Atmosphere.NDUST):
                            for ig in range(Spectroscopy.NG):
                                dspecg[:,ig,k,j] = dspecg[:,ig,k,j] + xfac * radground[:] * dtolddq[:,ig,k,j]

                SPECOUT[:,:,ipath] = specg[:,:]
                dSPECOUT[:,:,:,:,ipath] = dspecg[:,:,:,:]
                dTSURF[:,:,ipath] = tempgtsurf[:,:]

        #Now integrate over g-ordinates
        print('CIRSradg :: Integrading over g-ordinates')
        SPECOUT = np.tensordot(SPECOUT, Spectroscopy.DELG, axes=([1],[0])) #NWAVE,NPATH
        dSPECOUT = np.tensordot(dSPECOUT, Spectroscopy.DELG, axes=([1],[0])) #(WAVE,NGAS+2+NDUST,NLAYIN,NPATH)
        dTSURF = np.tensordot(dTSURF, Spectroscopy.DELG, axes=([1],[0])) #NWAVE,NPATH

        return SPECOUT,dSPECOUT,dTSURF

###############################################################################################

    def calc_tau_cia(self,ISPACE=None,WAVEC=None,CIA=None,Atmosphere=None,Layer=None,MakePlot=False):
        """
        Calculate the CIA opacity in each atmospheric layer
        @param ISPACE: int
            Flag indicating whether the calculation must be performed in wavenumbers (0) or wavelength (1)
        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param CIA: class
            Python class defining the CIA cross sections
        @param Atmosphere: class
            Python class defining the reference atmosphere
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAUCIA(NWAVE,NLAY) :: CIA optical depth in each atmospheric layer
        dTAUCIA(NWAVE,NLAY,7) :: Rate of change of CIA optical depth with:
                                 (1) H2 vmr
                                 (2) He vmr
                                 (3) N2 vmr
                                 (4) CH4 vmr
                                 (5) CO2 vmr
                                 (6) Temperature
                                 (7) para-H2 fraction
        IABSORB(5) :: Flag set to gas number in reference atmosphere for the species whose gradient is calculated
        """

        from scipy import interpolate
        from NemesisPy import find_nearest
        from NemesisPy.Radtrans.CIA_0 import co2cia,n2h2cia,n2n2cia


#       Initialising variables
        if ISPACE is None:
            ISPACE = self.MeasurementX.ISPACE
        if WAVEC is None:
            WAVEC = self.MeasurementX.WAVE
        if CIA is None:
            CIA = self.CIAX
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Layer is None:
            Layer = self.LayerX

#       the mixing ratios of the species contributing to CIA
        qh2=np.zeros(Layer.NLAY)
        qhe=np.zeros(Layer.NLAY)
        qn2=np.zeros(Layer.NLAY)
        qch4=np.zeros(Layer.NLAY)
        qco2=np.zeros(Layer.NLAY)
        IABSORB = np.ones(5,dtype='int32') * -1
        for i in range(Atmosphere.NVMR):

            if Atmosphere.ID[i]==39:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    qh2[:] = Layer.PP[:,i] / Layer.PRESS[:]
                    IABSORB[0] = i

            if Atmosphere.ID[i]==40:
                qhe[:] = Layer.PP[:,i] / Layer.PRESS[:]
                IABSORB[1] = i

            if Atmosphere.ID[i]==22:
                qn2[:] = Layer.PP[:,i] / Layer.PRESS[:]
                IABSORB[2] = i

            if Atmosphere.ID[i]==6:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    qch4[:] = Layer.PP[:,i] / Layer.PRESS[:]
                    IABSORB[3] = i

            if Atmosphere.ID[i]==2:
                qco2[:] = Layer.PP[:,i] / Layer.PRESS[:]
                IABSORB[4] = i

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

        if((WAVEN.min()<CIA.WAVEN.min()) or (WAVEN.max()>CIA.WAVEN.max())):
            print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

#       calculating the CIA opacity at the correct temperature and wavenumber
        NWAVEC = len(WAVEC)   #Number of calculation wavelengths
        tau_cia_layer = np.zeros((NWAVEC,Layer.NLAY))
        dtau_cia_layer = np.zeros((NWAVEC,Layer.NLAY,7))
        for ilay in range(Layer.NLAY):

            #Interpolating to the correct temperature
            temp1 = Layer.TEMP[ilay]
            temp0,it = find_nearest(CIA.TEMP,temp1)

            if CIA.TEMP[it]>=temp1:
                ithi = it
                if it==0:
                    temp1 = CIA.TEMP[it]
                    itl = 0
                    ithi = 1
                else:
                    itl = it - 1

            elif CIA.TEMP[it]<temp1:
                itl = it
                if it==CIA.NT-1:
                    temp1 = CIA.TEMP[it]
                    ithi = CIA.NT - 1
                    itl = CIA.NT - 2
                else:
                    ithi = it + 1

            ktlo = CIA.K_CIA[:,itl,:]
            kthi = CIA.K_CIA[:,ithi,:]

            fhl = (temp1 - CIA.TEMP[itl])/(CIA.TEMP[ithi] - CIA.TEMP[itl])
            fhh = (CIA.TEMP[ithi] - temp1)/(CIA.TEMP[ithi] - CIA.TEMP[itl])
            dfhldT = 1./(CIA.TEMP[ithi] - CIA.TEMP[itl])
            dfhhdT = -1./(CIA.TEMP[ithi] - CIA.TEMP[itl])

            kt = ktlo*(1.-fhl) + kthi * (1.-fhh)
            dktdT = -ktlo * dfhldT - kthi * dfhhdT

            #Cheking that interpolation can be performed to the calculation wavenumbers
            inwave = np.where( (CIA.WAVEN>=WAVEN.min()) & (CIA.WAVEN<=WAVEN.max()) )
            inwave = inwave[0]
            if len(inwave)>0:

                k_cia = np.zeros((NWAVEC,CIA.NPAIR))
                dkdT_cia = np.zeros((NWAVEC,CIA.NPAIR))
                inwave1 = np.where( (WAVEN>=CIA.WAVEN.min()) & (WAVEN<=CIA.WAVEN.max()) )
                inwave1 = inwave1[0]

                for ipair in range(CIA.NPAIR):
                    f = interpolate.interp1d(CIA.WAVEN,kt[ipair,:])
                    k_cia[inwave1,ipair] = f(WAVEN[inwave1])
                    f = interpolate.interp1d(CIA.WAVEN,dktdT[ipair,:])
                    dkdT_cia[inwave1,ipair] = f(WAVEN[inwave1])
                    #k_cia[inwave1,ipair] = np.interp(WAVEN[inwave1],CIA.WAVEN,kt[ipair,:])
                    #dkdT_cia[inwave1,ipair] = np.interp(WAVEN[inwave1],CIA.WAVEN,dktdT[ipair,:])


                #Combining the CIA absorption of the different pairs (included in .cia file)
                sum1 = np.zeros(NWAVEC)
                if CIA.INORMAL==0:   #equilibrium hydrogen
                    sum1[:] = sum1[:] + k_cia[:,0] * qh2[ilay] * qh2[ilay] + k_cia[:,1] * qhe[ilay] * qh2[ilay]
                    dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + 2.*qh2[ilay]*k_cia[:,0] + qhe[ilay]*k_cia[:,1]
                    dtau_cia_layer[:,ilay,1] = dtau_cia_layer[:,ilay,1] + qh2[ilay]*k_cia[:,1]
                    dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qh2[ilay] * qh2[ilay] * dkdT_cia[:,0] + dkdT_cia[:,1] * qhe[ilay] * qh2[ilay]

                elif CIA.INORMAL==1: #'normal' hydrogen
                    sum1[:] = sum1[:] + k_cia[:,2] * qh2[ilay] * qh2[ilay] + k_cia[:,3] * qhe[ilay] * qh2[ilay]
                    dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + 2.*qh2[ilay]*k_cia[:,2] + qhe[ilay]*k_cia[:,3]
                    dtau_cia_layer[:,ilay,1] = dtau_cia_layer[:,ilay,1] + qh2[ilay]*k_cia[:,3]
                    dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qh2[ilay] * qh2[ilay] * dkdT_cia[:,2] + dkdT_cia[:,3] * qhe[ilay] * qh2[ilay]

                sum1[:] = sum1[:] + k_cia[:,4] * qh2[ilay] * qn2[ilay]
                dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + qn2[ilay] * k_cia[:,4]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + qh2[ilay] * k_cia[:,4]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qn2[ilay]*qh2[ilay] * dkdT_cia[:,4]

                sum1[:] = sum1[:] + k_cia[:,5] * qn2[ilay] * qch4[ilay]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + qch4[ilay] * k_cia[:,5]
                dtau_cia_layer[:,ilay,3] = dtau_cia_layer[:,ilay,3] + qn2[ilay] * k_cia[:,5]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qn2[ilay]*qch4[ilay] * dkdT_cia[:,5]

                sum1[:] = sum1[:] + k_cia[:,6] * qn2[ilay] * qn2[ilay]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + 2.*qn2[ilay] * k_cia[:,6]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qn2[ilay]*qn2[ilay] * dkdT_cia[:,6]

                sum1[:] = sum1[:] + k_cia[:,7] * qch4[ilay] * qch4[ilay]
                dtau_cia_layer[:,ilay,3] = dtau_cia_layer[:,ilay,3] + 2.*qch4[ilay] * k_cia[:,7]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qch4[ilay]*qch4[ilay] * dkdT_cia[:,7]

                sum1[:] = sum1[:] + k_cia[:,8] * qh2[ilay] * qch4[ilay]
                dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + qch4[ilay] * k_cia[:,8]
                dtau_cia_layer[:,ilay,3] = dtau_cia_layer[:,ilay,3] + qh2[ilay] * k_cia[:,8]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qch4[ilay]*qh2[ilay] * dkdT_cia[:,8]

                #Look up CO2-CO2 CIA coefficients (external)
                k_co2 = co2cia(WAVEN)
                sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]
                dtau_cia_layer[:,ilay,4] = dtau_cia_layer[:,ilay,4] + 2.*qco2[ilay]*k_co2[:]

                #Look up N2-N2 NIR CIA coefficients
                k_n2n2 = n2n2cia(WAVEN)
                sum1[:] = sum1[:] + k_n2n2[:] * qn2[ilay] * qn2[ilay]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + 2.*qn2[ilay]*k_n2n2[:]

                #Look up N2-H2 NIR CIA coefficients
                k_n2h2 = n2h2cia(WAVEN)
                sum1[:] = sum1[:] + k_n2h2[:] * qn2[ilay] * qh2[ilay]
                dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + qn2[ilay] * k_n2h2[:]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + qh2[ilay] * k_n2h2[:]

                tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]
                dtau_cia_layer[:,ilay,:] = dtau_cia_layer[:,ilay,:] * tau[ilay]


        if ISPACE==1:
            tau_cia_layer[:,:] = tau_cia_layer[isort,:]
            dtau_cia_layer[:,:,:] = dtau_cia_layer[isort,:,:]

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            for ilay in range(Layer.NLAY):
                ax1.plot(WAVEC,tau_cia_layer[:,ilay])
            ax1.grid()
            plt.tight_layout()
            plt.show()

        return tau_cia_layer,dtau_cia_layer,IABSORB

###############################################################################################

    def calc_tau_dust(self,WAVEC=None,Scatter=None,Layer=None,MakePlot=False):
        """
        Calculate the aerosol opacity in each atmospheric layer

        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Scatter: class
            Scatter:: Python class defining the optical properties of the aerosols in the atmosphere
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAUDUST(NWAVE,NLAY,NDUST) :: Aerosol opacity for each aerosol type and each layer (from extinction coefficients)
        TAUCLSCAT(NWAVE,NLAY,NDUST) :: Aerosol scattering opacity for each aerosol type and each layer
        dTAUDUSTdq(NWAVE,NLAY,NDUST) :: Rate of change of the aerosol opacity with the dust abundance
        dTAUCLSCATdq(NWAVE,NLAY,NDUST) :: Rate of change of the aerosol scattering opacity with dust abundance
        """

#       Initialising variables
        if WAVEC is None:
            WAVEC = self.MeasurementX.WAVE
        if Scatter is None:
            Scatter = self.ScatterX
        if Layer is None:
            Layer = self.LayerX


        from scipy import interpolate
        from NemesisPy import find_nearest

        if((WAVEC.min()<Scatter.WAVE.min()) & (WAVEC.max()>Scatter.WAVE.min())):
            sys.exit('error in Scatter_0() :: Spectral range for calculation is outside of range in which the Aerosol properties are defined')

        #Calculating the opacity at each vertical layer for each dust population
        NWAVEC = len(WAVEC)
        TAUDUST = np.zeros((NWAVEC,Layer.NLAY,Scatter.NDUST))
        TAUCLSCAT = np.zeros((NWAVEC,Layer.NLAY,Scatter.NDUST))
        dTAUDUSTdq = np.zeros((NWAVEC,Layer.NLAY,Scatter.NDUST))
        dTAUCLSCATdq = np.zeros((NWAVEC,Layer.NLAY,Scatter.NDUST))
        for i in range(Scatter.NDUST):

            #Interpolating the cross sections to the correct grid
            f = interpolate.interp1d(Scatter.WAVE,np.log(Scatter.KEXT[:,i]))
            kext = np.exp(f(WAVEC))
            f = interpolate.interp1d(Scatter.WAVE,np.log(Scatter.KSCA[:,i]))
            ksca = np.exp(f(WAVEC))

            #Calculating the opacity at each layer
            for j in range(Layer.NLAY):
                DUSTCOLDENS = Layer.CONT[j,i]  #particles/m2
                TAUDUST[:,j,i] =  kext * 1.0e-4 * DUSTCOLDENS
                TAUCLSCAT[:,j,i] = ksca * 1.0e-4 * DUSTCOLDENS
                dTAUDUSTdq[:,j,i] = kext * 1.0e-4 #dtau/dAMOUNT (m2)
                dTAUCLSCATdq[:,j,i] = ksca * 1.0e-4 #dtau/dAMOUNT (m2)

        return TAUDUST,TAUCLSCAT,dTAUDUSTdq,dTAUCLSCATdq


    ###############################################################################################
    ###############################################################################################
    # MULTIPLE SCATTERING ROUTINES
    ###############################################################################################
    ###############################################################################################


    def scloud11wave(self,Scatter,Surface,WAVE):
        """

        Compute emergent intensity at top of multilayer cloud using the
        matrix operator algorithm.  Diffuse incident radiation is allowed
        at the bottom and single-beam incident radiation (sunlight) at
        the top. 

        The layer numbers here are 1 at the top increasing to NLAY at the 
        bottom. If a Lambertian reflector is added at the bottom then the 
        number of layers is increased by 1.

        NOTE:  the angle arrays passed from the calling pgm. are assumed to be 
        in order of increasing MU.  However, the CLOUD subroutines use the 
        opposite convention, so we must reverse the order.  (The order has no 
        effect within this routine, other than in using the supplied boundary 
        conditions, but it does affect pre-/post-processing in the calling s/w.)

        Optimised for maximum speed by cutting out any code from scloud6
        which is not actually used for NIMS retrieval runs.
 
        Inputs
        ________

        Scatter :: Python class defining the scattering setup
        Surface :: Python class defining the surface setup

        Outputs
        ________

        SPECOUT(NWAVE) :: Modelled radiance 

        """

        #INITIALISING VARIABLES AND PERFORMING INITIAL CALCULATIONS
        ##############################################################################

        NWAVE = len(WAVE)

        #Find correction for any quadrature errors
        xfac = np.sum(Scatter.MU*Scatter.WTMU)
        xfac = 0.5/xfac

        LTOT = NLAY     # Set internal number of layers
        LT1 = LTOT

        #In case of surface reflection, add extra dummy layer at bottom,
        #whose transmission matrix = (1-A)*Unit-Matrix. This layer must be
        #omitted from computation by doubling

        if Surface.LOWBC>0:
            LTOT = LTOT + 1

        #Reset the order of angles
        Scatter.MU = Scatter.MU.reverse()
        Scatter.WTMU = Scatter.WTMU.reverse()

        #Setting up constant matrices
        E = np.identity(Scatter.NMU)
        MM = np.fill_diagonal(np.zeros((Scatter.NMU,Scatter.NMU)),Scatter.MU)
        MMINV = np.fill_diagonal(np.zeros((Scatter.NMU,Scatter.NMU)),1./Scatter.MU)
        CC = np.fill_diagonal(np.zeros((Scatter.NMU,Scatter.NMU)),Scatter.WTMU)
        CCINV = np.fill_diagonal(np.zeros((Scatter.NMU,Scatter.NMU)),1./Scatter.WTMU)

        #Calculating the phase matrices for each aerosol population and Rayleigh scattering
        phase = np.zeros((NWAVE,Scatter.NMU,Scatter.NDUST+1))
        
        phase[:,:,0:Scatter.NDUST] = Scatter.calc_phase(Scatter.MU)


###############################################################################################

    def calc_phase_matrix(self,Scatter,WAVE):
        """

        Calculate the phase matrix at the different angles required for the multiple
        scattering calculations. These are the P++ and P+- matrices in Plass et al. (1973).
 
        Inputs
        ________

        Scatter :: Python class defining the scattering setup
        WAVE(NWAVE) :: Calculation wavelengths

        Outputs
        ________

        PPLPL(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the + direction (i.e. downward)
        PPLMI(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the - direction (i.e. upward)

        """



        #Calculating the phase function at the scattering angles
        #######################################################################

        NWAVE = len(WAVE)
        dphi = 2.0*np.pi/Scatter.NPHI

        #Defining the angles at which the phase functions must be calculated
        cpl = np.zeros(Scatter.NMU*Scatter.NMU*(Scatter.NPHI+1))
        cmi = np.zeros(Scatter.NMU*Scatter.NMU*(Scatter.NPHI+1))
        ix = 0
        for j in range(Scatter.NMU):
            for i in range(Scatter.NMU):
                sthi = np.sqrt(1.0-Scatter.MU[i]*Scatter.MU[i])   #sin(theta(i))
                sthj = np.sqrt(1.0-Scatter.MU[j]*Scatter.MU[j])   #sin(theta(j))

                for k in range(Scatter.NPHI+1):
                    phi = k*dphi
                    cpl[ix] = sthi*sthj*np.cos(phi) + Scatter.MU[i]*Scatter.MU[j]
                    cmi[ix] = sthi*sthj*np.cos(phi) - Scatter.MU[i]*Scatter.MU[j]
                    ix = ix + 1

        #Calculating the phase function at the required wavelengths and scattering angles
        apl = np.arccos(cpl) / np.pi * 180.
        ami = np.arccos(cmi) / np.pi * 180.
        
        ppl = Scatter.calc_phase(apl,WAVE)  #(NWAVE,NTHETA,NDUST)
        pmi = Scatter.calc_phase(ami,WAVE)  #(NWAVE,NTHETA,NDUST)

        if Scatter.IRAY>0:
            ncont = Scatter.NDUST + 1
            pplr = Scatter.calc_phase_ray(apl) #(NTHETA)
            pmir = Scatter.calc_phase_ray(ami) #(NTHETA)
        else:
            ncont = Scatter.NDUST


        #Integrating the phase function over the azimuth direction
        #####################################################################################

        PPLPL = np.zeros((NWAVE,ncont,Scatter.NF,Scatter.NMU,Scatter.NMU)) #Integrated phase function coefficients in + direction (i.e. downwards)
        PPLMI = np.zeros((NWAVE,ncont,Scatter.NF,Scatter.NMU,Scatter.NMU)) #Integrated phase function coefficients in - direction (i.e. upwards)
        ix = 0
        for j in range(Scatter.NMU):
            for i in range(Scatter.NMU):
                for k in range(Scatter.NPHI+1):
                    phi = k*dphi
                    for kl in range(Scatter.NF+1):
                        
                        plx = ppl[:,ix,:] * np.cos(kl*phi)
                        pmx = pmi[:,ix,:] * np.cos(kl*phi)

                        wphi = 1.0*dphi
                        if k==0:
                            wphi = 0.5*dphi
                        elif k==Scatter.NPHI:
                            wphi = 0.5*dphi

                        if kl==0:
                            wphi = wphi/(2.0*np.pi)
                        else:
                            wphi = wphi/np.pi

                        PPLPL[:,0:Scatter.NDUST,kl,i,j] = PPLPL[0:Scatter.NDUST,kl,i,j] + wphi*plx[:,:]
                        PPLMI[:,0:Scatter.NDUST,kl,i,j] = PPLMI[0:Scatter.NDUST,kl,i,j] + wphi*pmi[:,:]

                        if Scatter.IRAY>0:
                            plrx = pplr[ix] * np.cos(kl*phi)
                            pmrx = pmir[ix] * np.cos(kl*phi)
                            PPLPL[:,Scatter.NDUST,kl,i,j] = PPLPL[Scatter.NDUST,kl,i,j] + wphi*plrx
                            PPLMI[:,Scatter.NDUST,kl,i,j] = PPLMI[Scatter.NDUST,kl,i,j] + wphi*pmrx

                    ix = ix + 1


        #Normalising the phase matrices using the method of Hansen (1971,J.ATM.SCI., V28, 1400)
        ###############################################################################################

        #PPL,PMI ARE THE FORWARD AND BACKWARD PARTS OF THE AZIMUTHALLY-INTEGRATED
        #PHASE FUNCTION.  THE NORMALIZATION OF THE TRUE PHASE FCN. IS:
        #integral over sphere [ P(mu,mu',phi) * dO] = 1
        #WHERE dO IS THE ELEMENT OF SOLID ANGLE AND phi IS THE AZIMUTHAL ANGLE.
 
        IC = 0

        RSUM = np.zeros((NWAVE,ncont,Scatter.NMU))
        for j in range(Scatter.NMU):
            for i in range(Scatter.NMU):
                RSUM[:,:,j] = RSUM[:,:,j] + PPLMI[:,:,IC,i,j] * Scatter.WTMU[i]*2.0*np.pi


        FC = np.ones((NWAVE,Scatter.NMU,Scatter.NMU))
        for icont in range(ncont):
            niter = 1
            converged = False
            while converged==False:
                test = np.zeros(NWAVE)
                TSUM = np.zeros((NWAVE,Scatter.NMU))
                for j in range(Scatter.NMU):
                    for i in range(Scatter.NMU):
                        TSUM[:,j] = TSUM[:,j] + PPLPL[:,icont,IC,i,j]*Scatter.WTMU[i]*FC[i,j]*2.0*np.pi

                    testj = np.abs(  RSUM[:,icont,j] + TSUM[:,icont,j] - 1.0 )
                    isup = np.where(testj>test)[0]
                    test[isup] = testj[isup]

                if test.max()<1.0e-14:
                    converged==True
                else:
                    for j in range(Scatter.NMU):
                        xj = (1.0-RSUM[:,icont,j])/TSUM[:,icont,j]
                        for i in range(j+1):
                            xi = (1.0-RSUM[:,icont,i])/TSUM[:,icont,i]
                            FC[:,i,j] = 0.5*(FC[:,i,j]*xj[:]+FC[:,j,i]*xi[:])
                            FC[:,j,i] = FC[i,j]

                if niter>10000:
                    sys.exit('error in calc_phase_matrix :: Normalisation of phase matrix did not converge')

                niter = niter + 1

            for kl in range(Scatter.NF):
                for j in range(Scatter.NMU):
                    for i in range(Scatter.NMU):
                        PPLPL[:,icont,kl,i,j] = PPLPL[:,icont,kl,i,j] * FC[:,i,j]


        return PPLPL,PPLMI









