from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def subspecret(Measurement,Variables,SPECMOD,MakePlot=False):

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

    OPTIONAL INPUTS:

        MakePlot :: If True, a summary plot is made
            
    OUTPUTS : 

        SPECMOD :: Updated modelled spectrum

    CALLING SEQUENCE:

        SPECMOD = subspecret(Measurement,Variables,SPECMOD)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)

    """

    #Going through the different variables an updating the atmosphere accordingly
    ix = 0
    for ivar in range(Variables.NVAR):

        if Variables.VARIDENT[ivar,0]==231:
#       Model 231. Continuum addition to transmission spectra using a linearly varying scaling factor
#       ***********************************************************************************************

            for i in range(Measurement.NGEOM):
                T0 = Variables.XN[ix]
                T1 = Variables.XN[ix+1]

                WAVE0 = Measurement.WAVE.min()
                SPECMOD[:,i] = SPECMOD[:,i] * ( T0 + T1*(Measurement.WAVE-WAVE0) )

                ix = ix + 2

        elif Variables.VARIDENT[ivar,0]==667:
#       Model 667. Spectrum scaled by dillusion factor to account for thermal gradients in planets
#       **********************************************************************************************

            xfactor = Variables.XN[ix]
            SPECMOD = model667(SPECMOD,xfactor)
            ix = ix + 1

        else:
            ix = ix + Variables.NXVAR[ivar]

    return SPECMOD