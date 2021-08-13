#!/usr/bin/python
#####################################################################################
#####################################################################################
#                                        nemesisSO
#####################################################################################
#####################################################################################

# Version of Nemesis for doing retrievals in solar occultation observations

#####################################################################################
#####################################################################################

from NemesisPy import *
import matplotlib.pyplot as plt
import numpy as np
import time

runname = input('run name: ')
 
start = time.time()

######################################################
######################################################
#    READING INPUT FILES AND SETTING UP VARIABLES
######################################################
######################################################


#Initialise Atmosphere class and read file (.ref, aerosol.ref)
##############################################################

Atm = Atmosphere_1()

#Read gaseous atmosphere
Atm.read_ref(runname)

#Read aerosol profiles
Atm.read_aerosol()

#Initialise Spectroscopy class and read file (.lls)
##############################################################

Spec = Spectroscopy_0(ILBL=2)
if Spec.ILBL==0:
    Spec.read_kls(runname)
elif Spec.ILBL==2:
    Spec.read_lls(runname)
else:
    sys.exit('error :: ILBL has to be either 0 or 2')


#Reading .set file and starting Scatter, Stellar, Surface and Layer Classes
#############################################################################

Layer = Layer_0(Atm.RADIUS)
Scatter,Stellar,Surface,Layer = read_set(runname,Layer=Layer)

#Reading extinction and scattering cross sections
#############################################################################

Scatter.read_xsc(runname)

if Scatter.NDUST!=Atm.NDUST:
    sys.exit('error :: Number of aerosol populations must be the same in .xsc and aerosol.ref files')

#Initialise Measurement class and read files (.spx, .sha)
##############################################################

Measurement = Measurement_0()
Measurement.read_spx_SO(runname)
Measurement.IFORM = 0
Measurement.ISPACE = 0

#Reading .sha file if FWHM>0.0
if Measurement.FWHM>0.0:
    Measurement.read_sha(runname)
#Reading .fil if FWHM<0.0
elif Measurement.FWHM<0.0:
    Measurement.read_fil(runname)

#Calculating the 'calculation wavelengths'
if Spec.ILBL==0:
    Measurement.wavesetb(Spec,IGEOM=0)
elif Spec.ILBL==2:
    Measurement.wavesetc(Spec,IGEOM=0)
else:
    sys.exit('error :: ILBL has to be either 0 or 2')

#Now, reading k-tables or lbl-tables for the spectral range of interest
Spec.read_tables(wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())

#Initialise CIA class and read files (.cia)
##############################################################

CIA = CIA_0()
CIA.read_cia(runname)

#Reading .apr file and Variables Class
#################################################################

Variables = Variables_0()
Variables.read_apr(runname,Atm.NP)
Variables.XN = copy(Variables.XA)
Variables.SX = copy(Variables.SA)



######################################################
######################################################
#      RUN THE RETRIEVAL USING ANY APPROACH
######################################################
######################################################


IRET = 0    #(0) Optimal Estimation (1) Nested sampling
if IRET==0:
    OptimalEstimation = coreretOE_SO(runname,Variables,Measurement,Atm,Spec,Scatter,Stellar,Surface,CIA,Layer,\
                                     NITER=8,PHILIMIT=0.1)
else:
    sys.exit('error in nemesisSO :: Retrieval scheme has not been implemented yet')


######################################################
######################################################
#                WRITE OUTPUT FILES
######################################################
######################################################

if IRET==0:
    OptimalEstimation.write_cov(runname,Variables)
    OptimalEstimation.write_mre(runname,Variables,Measurement)

#Finishing pogram
end = time.time()
print('Model run OK')
print(' Elapsed time (s) = '+str(end-start))
