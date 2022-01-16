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

        """
        if Variables.VARIDENT[ivar,0]==231:
#       Model 231. Scaling of spectra using a varying scaling factor (following a polynomial of degree N)
#       ****************************************************************************************************

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
        """

        if Variables.VARIDENT[ivar,0]==231:
#       Model 231. Scaling of spectra using a varying scaling factor (following a polynomial of degree N)
#       ****************************************************************************************************

            NGEOM = int(Variables.VARPARAM[ivar,0])
            NDEGREE = int(Variables.VARPARAM[ivar,1])

            if Measurement.NGEOM>1:

                for i in range(Measurement.NGEOM):

                    #Getting the coefficients
                    T = np.zeros(NDEGREE+1)
                    for j in range(NDEGREE+1):
                        T[j] = Variables.XN[ix+j]
                        
                    WAVE0 = Measurement.WAVE.min()
                    spec = np.zeros(Measurement.NWAVE)
                    spec[:] = SPECMOD[:,i]

                    #Changing the state vector based on this parameterisation
                    POL = np.zeros(Measurement.NWAVE)
                    for j in range(NDEGREE+1):
                        POL[:] = POL[:] + T[j]*(Measurement.WAVE[:]-WAVE0)**j

                    SPECMOD[:,i] = SPECMOD[:,i] * POL[:]

                    #Changing the rest of the gradients based on the impact of this parameterisation
                    for ixn in range(Variables.NX):
                        dSPECMOD[:,i,ixn] = dSPECMOD[:,i,ixn] * POL[:]

                    #Defining the analytical gradients for this parameterisation
                    for j in range(NDEGREE+1):
                        dSPECMOD[:,i,ix+j] = spec[:] * (Measurement.WAVE[:]-WAVE0)**j

                    ix = ix + (NDEGREE+1)

        elif Variables.VARIDENT[ivar,0]==232:
#       Model 232. Continuum addition to transmission spectra using the angstrom coefficient
#       ***************************************************************

            #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
            #Where the parameters to fit are TAU0 and ALPHA

            #The effect of this model takes place after the computation of the spectra in CIRSrad!
            if int(Variables.NXVAR[ivar]/2)!=Measurement.NGEOM:
                sys.exit('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

            if Measurement.NGEOM>1:

                for i in range(Measurement.NGEOM):
                    TAU0 = Variables.XN[ix]
                    ALPHA = Variables.XN[ix+1]
                    WAVE0 = Variables.VARPARAM[ivar,1]  

                    spec = np.zeros(Measurement.NWAVE)
                    spec[:] = SPECMOD[:,i]

                    #Changing the state vector based on this parameterisation
                    SPECMOD[:,i] = SPECMOD[:,i] * np.exp ( -TAU0 * (Measurement.WAVE/WAVE0)**(-ALPHA) )

                    #Changing the rest of the gradients based on the impact of this parameterisation
                    for ixn in range(Variables.NX):
                        dSPECMOD[:,i,ixn] = dSPECMOD[:,i,ixn] * np.exp ( -TAU0 * (Measurement.WAVE/WAVE0)**(-ALPHA) )

                    #Defining the analytical gradients for this parameterisation
                    dSPECMOD[:,i,ix] = spec[:] * ( -((Measurement.WAVE/WAVE0)**(-ALPHA)) * np.exp ( -TAU0 * (Measurement.WAVE/WAVE0)**(-ALPHA) ) )
                    dSPECMOD[:,i,ix+1] = spec[:] * TAU0 * np.exp ( -TAU0 * (Measurement.WAVE/WAVE0)**(-ALPHA) ) * np.log(Measurement.WAVE/WAVE0) * (Measurement.WAVE/WAVE0)**(-ALPHA)

                    ix = ix + 2

            else:

                
                T0 = Variables.XN[ix]
                ALPHA = Variables.XN[ix+1]
                WAVE0 = Variables.VARPARAM[ivar,1]

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

        elif Variables.VARIDENT[ivar,0]==233:
#       Model 232. Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)
#       ***************************************************************

            #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
            #Where the aerosol opacity is modelled following

            # np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

            #The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
            #233 converges to model 232 when a2=0.

            #The effect of this model takes place after the computation of the spectra in CIRSrad!
            if int(Variables.NXVAR[ivar]/3)!=Measurement.NGEOM:
                sys.exit('error using Model 233 :: The number of levels for the addition of continuum must be the same as NGEOM')

            if Measurement.NGEOM>1:

                for i in range(Measurement.NGEOM):

                    A0 = Variables.XN[ix]
                    A1 = Variables.XN[ix+1]
                    A2 = Variables.XN[ix+2] 

                    spec = np.zeros(Measurement.NWAVE)
                    spec[:] = SPECMOD[:,i]

                    #Calculating the aerosol opacity at each wavelength
                    TAU = np.exp(A0 + A1 * np.log(Measurement.WAVE) + A2 * np.log(Measurement.WAVE)**2.)

                    #Changing the state vector based on this parameterisation
                    SPECMOD[:,i] = SPECMOD[:,i] * np.exp ( -TAU )

                    #Changing the rest of the gradients based on the impact of this parameterisation
                    for ixn in range(Variables.NX):
                        dSPECMOD[:,i,ixn] = dSPECMOD[:,i,ixn] * np.exp ( -TAU )

                    #Defining the analytical gradients for this parameterisation
                    dSPECMOD[:,i,ix] = spec[:] * (-TAU) * np.exp(-TAU)
                    dSPECMOD[:,i,ix+1] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Measurement.WAVE)
                    dSPECMOD[:,i,ix+2] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Measurement.WAVE)**2.

                    ix = ix + 3

            else:

                A0 = Variables.XN[ix]
                A1 = Variables.XN[ix+1]
                A2 = Variables.XN[ix+2]

                #Getting spectrum
                spec = np.zeros(Measurement.NWAVE)
                spec[:] = SPECMOD

                #Calculating aerosol opacity
                TAU = np.exp(A0 + A1 * np.log(Measurement.WAVE) + A2 * np.log(Measurement.WAVE)**2.)

                SPECMOD[:] = SPECMOD[:] * np.exp(-TAU)
                for ixn in range(Variables.NX):
                    dSPECMOD[:,ixn] = dSPECMOD[:,ixn] * np.exp(-TAU)

                #Defining the analytical gradients for this parameterisation
                dSPECMOD[:,ix] = spec[:] * (-TAU) * np.exp(-TAU)
                dSPECMOD[:,ix+1] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Measurement.WAVE)
                dSPECMOD[:,ix+2] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Measurement.WAVE)**2.

                ix = ix + 3

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