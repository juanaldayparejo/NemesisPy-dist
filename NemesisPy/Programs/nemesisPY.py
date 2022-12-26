#!/usr/bin/python
#####################################################################################
#####################################################################################
#                                        nemesis
#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################

from NemesisPy import *
import matplotlib.pyplot as plt
import numpy as np
import time

runname = input('run name: ')
NCores = 1
 
start = time.time()


######################################################
######################################################
#    READING INPUT FILES AND SETTING UP VARIABLES
######################################################
######################################################


#Initialise Atmosphere class and read file (.ref, aerosol.ref)
##############################################################

Atm = Atmosphere_1(runname=runname)

#Read gaseous atmosphere
Atm.read_ref()

#Read aerosol profiles
Atm.read_aerosol()

#Reading .set file and starting Scatter, Stellar, Surface and Layer Classes
#############################################################################

Layer = Layer_0(Atm.RADIUS)
Scatter,Stellar,Surface,Layer = read_set(runname,Layer=Layer)

#Reading .inp file and starting Measurement,Scatter and Spectroscopy classes
#############################################################################

Measurement,Scatter,Spec,WOFF,fmerrname,NITER,PHILIMIT,NSPEC,IOFF,LIN = read_inp(runname,Scatter=Scatter)

#Reading .sur file if planet has surface
#############################################################################

isurf = planet_info[str(Atm.IPLANET)]["isurf"]
if isurf==1:
    Surface.GASGIANT=False
    Surface.read_sur(runname)
else:
    Surface.GASGIANT=True

#Reading Spectroscopy parameters from .lls or .kls files
##############################################################

if Spec.ILBL==0:
    Spec.read_kls(runname)
elif Spec.ILBL==2:
    Spec.read_lls(runname)
else:
    sys.exit('error :: ILBL has to be either 0 or 2')

#Reading extinction and scattering cross sections
#############################################################################

Scatter.read_xsc(runname)

if Scatter.NDUST!=Atm.NDUST:
    sys.exit('error :: Number of aerosol populations must be the same in .xsc and aerosol.ref files')

if Scatter.ISCAT>0:
    if Scatter.IMIE==0:
        Scatter.read_hgphase()
    elif Scatter.IMIE==1:
        Scatter.read_phase()
    elif Scatter.IMIE==2:
        Scatter.read_lpphase()

#Initialise Measurement class and read files (.spx, .sha)
##############################################################

Measurement.runname = runname
Measurement.read_spx()

#Reading .sha file if FWHM>0.0
if Measurement.FWHM>0.0:
    Measurement.read_sha()
#Reading .fil if FWHM<0.0
elif Measurement.FWHM<0.0:
    Measurement.read_fil()

#Calculating the 'calculation wavelengths'
if Spec.ILBL==0:
    Measurement.wavesetb(Spec,IGEOM=0)
elif Spec.ILBL==2:
    Measurement.wavesetc(Spec,IGEOM=0)
else:
    sys.exit('error :: ILBL has to be either 0 or 2')

#Now, reading k-tables or lbl-tables for the spectral range of interest
Spec.read_tables(wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())


#Reading stellar spectrum if required by Measurement units
if( (Measurement.IFORM==1) or (Measurement.IFORM==2) or (Measurement.IFORM==3) or (Measurement.IFORM==4)):
    Stellar.read_sol(runname)

#Initialise CIA class and read files (.cia)
##############################################################

if os.path.exists(runname+'.cia')==True:
    CIA = CIA_0(runname=runname)
    CIA.read_cia()
else:
    CIA = None

#Reading .fla file
#############################################################################

inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv = read_fla(runname)

if CIA is not None:
    CIA.INORMAL = inormal

Scatter.IRAY = iray
Scatter.IMIE = imie

#Reading .apr file and Variables Class
#################################################################

Variables = Variables_0()
Variables.read_apr(runname,Atm.NP)
Variables.XN = copy(Variables.XA)
Variables.SX = copy(Variables.SA)

#If scattering is required, all elements in the jacobian
#are calculated numerically
if Scatter.ISCAT>0:
    Variables.NUM[:] = 1


######################################################
######################################################
#      RUN THE RETRIEVAL USING ANY APPROACH
######################################################
######################################################


IRET = 0    #(0) Optimal Estimation (1) Nested sampling
if IRET==0:
    OptimalEstimation = coreretOE(runname,Variables,Measurement,Atm,Spec,Scatter,Stellar,Surface,CIA,Layer,\
                                     NITER=NITER,PHILIMIT=PHILIMIT,NCores=NCores)
else:
    sys.exit('error in nemesisSO :: Retrieval scheme has not been implemented yet')


######################################################
######################################################
#                WRITE OUTPUT FILES
######################################################
######################################################

if IRET==0:
    OptimalEstimation.write_cov(runname,Variables,pickle=True)
    OptimalEstimation.write_mre(runname,Variables,Measurement)

#Finishing pogram
end = time.time()
print('Model run OK')
print(' Elapsed time (s) = '+str(end-start))
