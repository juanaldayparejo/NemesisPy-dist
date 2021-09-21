from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def subspecret(Measurement,Variables,SPECMOD,dSPECMOD,MakePlot=False):

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
    for ivar in range(Variables.NVAR):

        if Variables.VARIDENT[ivar,0]==231:
#       Model 231. Scaling of spectra using a linearly varying scaling factor
#       ***********************************************************************************************

            if Measurement.NGEOM>1:

                for i in range(Measurement.NGEOM):
                    T0 = Variables.XN[ix]
                    T1 = Variables.XN[ix+1]
                    WAVE0 = Measurement.WAVE.min()

                    spec = np.zeros(Measurement.NWAVE)
                    spec[:] = SPECMOD[:,i]

                    #Changing the state vector based on this parameterisation
                    SPECMOD[:,i] = SPECMOD[:,i] * ( T0 + T1*(Measurement.WAVE-WAVE0) )

                    #Changing the rest of the gradients based on the impact of this parameterisation
                    for ixn in range(Variables.NX):
                        dSPECMOD[:,i,ixn] = dSPECMOD[:,i,ixn] * ( T0 + T1*(Measurement.WAVE-WAVE0) )

                    #Defining the analytical gradients for this parameterisation
                    dSPECMOD[:,i,ix] = spec[:]
                    dSPECMOD[:,i,ix+1] = spec[:] * (Measurement.WAVE[:]-WAVE0)

                    ix = ix + 2

            else:

                T0 = Variables.XN[ix]
                T1 = Variables.XN[ix+1]
                WAVE0 = Measurement.WAVE.min()

                spec = np.zeros(Measurement.NWAVE)
                spec[:] = SPECMOD
                SPECMOD[:] = SPECMOD[:] * ( T0 + T1*(Measurement.WAVE-WAVE0) )
                for ixn in range(Variables.NX):
                    dSPECMOD[:,ixn] = dSPECMOD[:,ixn] * ( T0 + T1*(Measurement.WAVE-WAVE0) )

                dSPECMOD[:,ix] = spec
                dSPECMOD[:,ix+1] = spec * (Measurement.WAVE-WAVE0)

                ix = ix + 2


        elif Variables.VARIDENT[ivar,0]==667:
#       Model 667. Spectrum scaled by dilution factor to account for thermal gradients in planets
#       **********************************************************************************************

            xfactor = Variables.XN[ix]
            spec = np.zeros(Measurement.NWAVE)
            spec[:] = SPECMOD
            SPECMOD = model667(SPECMOD,xfactor)
            dSPECMOD = dSPECMOD * xfactor
            dSPECMOD[:,ix] = spec[:]
            ix = ix + 1

        else:
            ix = ix + Variables.NXVAR[ivar]

    return SPECMOD,dSPECMOD