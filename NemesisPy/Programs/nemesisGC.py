#!/usr/bin/python3
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
from copy import copy
import sys,os
import time
import pickle

runname = input('run name: ')
 NCores=1
 
start = time.time()


#Reading spectrum
######################################

Measurement = Measurement_0()
Measurement.read_spx_SO(runname)
Measurement.IFORM = 0

#Starting spectroscopy class
######################################

Spectroscopy = Spectroscopy_0()
Spectroscopy.read_lls(runname) #Assumed to be filled with (CO2(1-4) and H2O(1))
if Spectroscopy.NGAS!=5:
    sys.exit('error :: nemesisGC is very specific and is just suitable for a retrieval of CO2(1-4) isotopes and H2O(1)')
Measurement.wavesetc(Spectroscopy,IGEOM=0)

#Now, reading k-tables or lbl-tables for the spectral range of interest
Spectroscopy.read_tables(wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())


#Starting variables class
######################################

Variables = Variables_0()
Variables.NX = 7 #P,T, NLOS_CO2(1), C13RATIO, O18RATIO, O17RATIO, NLOS_H2O(1)
Variables.XA = np.ones(Variables.NX)
Variables.XA[0] = np.log(1.0e-9)
Variables.XA[1] = 150. #Temperature (K)
Variables.XA[2] = np.log(0.96) #Line-of-sight density of CO2(1) in cm-2
Variables.XA[3] = 1.0
Variables.XA[4] = 1.0
Variables.XA[5] = 1.0
Variables.XA[6] = np.log(1.0e-6) #Line-of-sight density of H2O(1) in cm-2

Variables.LX = np.zeros(Variables.NX,dtype='int32')
Variables.LX[0] = 1
Variables.LX[1] = 0
Variables.LX[2] = 1
Variables.LX[3] = 0
Variables.LX[4] = 0
Variables.LX[5] = 0
Variables.LX[6] = 1


Variables.SA = np.zeros((Variables.NX,Variables.NX))
Variables.SA[0,0] = 1.0e-30
Variables.SA[1,1] = 30.**2.
Variables.SA[2,2] = 1.
Variables.SA[3,3] = 0.09 #30%uncertainty in ratios
Variables.SA[4,4] = 0.09 #30%uncertainty in ratios
Variables.SA[5,5] = 0.09 #30%uncertainty in ratios
Variables.SA[6,6] = 1.

Variables.XN = copy(Variables.XA)

#Defining the forward model function
##########################################################################################

def nemesisGCfm(IGEOM,Measurement,Spectroscopy,Variables):

    Nlos = 5.0e20
    P = np.exp(Variables.XN[0]) #Pressure in atm
    T = Variables.XN[1] #Temperature in K
    Nlos_CO2 = np.exp(Variables.XN[2]) * Nlos #Line-of-sight density of CO2(1) in cm-2
    C13ratio = Variables.XN[3] #(13C)/(12C) ratio
    O18ratio = Variables.XN[4] #(18O)/(16O) ratio
    O17ratio = Variables.XN[5] #(17O)/(16O) ratio
    Nlos_H2O = np.exp(Variables.XN[6]) * Nlos #Line-of-sight density of CO2(1) in cm-2

    #Determining the absorption cross sections for each gas
    k = Spectroscopy.calc_klbl(1,[P],[T],WAVECALC=Measurement.WAVE)

    TAU = np.zeros(Measurement.NWAVE)

    for i in range(Spectroscopy.NGAS):

        if((Spectroscopy.ID[i]==2) & (Spectroscopy.ISO[i]==1)):
            TAU[:] = TAU[:] + k[:,0,i] * 1.0e-20 * Nlos_CO2 * 0.984204
        elif ((Spectroscopy.ID[i]==2) & (Spectroscopy.ISO[i]==2)):
            TAU[:] = TAU[:] + k[:,0,i] * 1.0e-20 * Nlos_CO2 * C13ratio * 0.01123720
        elif ((Spectroscopy.ID[i]==2) & (Spectroscopy.ISO[i]==3)):
            TAU[:] = TAU[:] + k[:,0,i] * 1.0e-20 * Nlos_CO2 * O18ratio * 0.003947
        elif ((Spectroscopy.ID[i]==2) & (Spectroscopy.ISO[i]==4)):
            TAU[:] = TAU[:] + k[:,0,i] * 1.0e-20 * Nlos_CO2 * O17ratio * 7.339890e-4
        elif((Spectroscopy.ID[i]==1) & (Spectroscopy.ISO[i]==1)):
            TAU[:] = TAU[:] + k[:,0,i] * 1.0e-20 * Nlos_H2O
        else:
            sys.exit('error :: nemesisGCfm just supports CO2 isotopes and H2O(1)')

    TRANS = np.exp(-TAU)
    TRANSCONV = Measurement.lblconv(TRANS,IGEOM=0)

    #fig,ax1 = plt.subplots(1,1,figsize=(10,3))
    #ax1.plot(Measurement.VCONV[0:Measurement.NCONV[IGEOM],IGEOM],TRANSCONV,c='tab:blue')
    #ax1.plot(Measurement.VCONV[0:Measurement.NCONV[IGEOM],IGEOM],Measurement.MEAS[0:Measurement.NCONV[IGEOM],IGEOM],c='black')
    #plt.tight_layout()
    #plt.show()
    

    return TRANSCONV
        
    
#Defining the Jacobian function
##########################################################################################

def jacobian_matrix_nemesisGC(IGEOM,Measurement,Spectroscopy,Variables):

    KK = np.zeros((Measurement.NCONV[IGEOM],Variables.NX))
    XNX = np.zeros((Variables.NX,Variables.NX+1))
    for i in range(Variables.NX+1):

        if i==0:
            XNX[:,i] = Variables.XN[:]
        elif i>0:
            XNX[:,i] = Variables.XN[:]
            XNX[i-1,i] = Variables.XN[i-1] * 1.05

    #Running the forward models
    YN = nemesisGCfm(IGEOM,Measurement,Spectroscopy,Variables)

    for i in range(Variables.NX):

        Variables1 = copy(Variables)
        Variables1.XN[:] = XNX[:,i+1]

        YN1 = nemesisGCfm(IGEOM,Measurement,Spectroscopy,Variables1)

        KK[:,i] = (YN1[:]-YN[:])/(XNX[i,i+1] - XNX[i,0])

    return YN,KK

#Running the retrieval under the Optimal Estimation framework
##########################################################################################

f = open(runname+'.out',"wb")
pickle.dump(Measurement.NGEOM,f,pickle.HIGHEST_PROTOCOL)
pickle.dump(Measurement.TANHE[:,0],f,pickle.HIGHEST_PROTOCOL)

for IGEOM in range(Measurement.NGEOM):
  
    OptimalEstimation = OptimalEstimation_0()

    OptimalEstimation.NITER = 20
    OptimalEstimation.PHILIMIT = 0.1
    OptimalEstimation.NX = Variables.NX
    OptimalEstimation.NY = Measurement.NCONV[IGEOM]
    OptimalEstimation.edit_XA(Variables.XA)
    OptimalEstimation.edit_XN(Variables.XA)
    OptimalEstimation.edit_SA(Variables.SA)
    OptimalEstimation.edit_Y(Measurement.MEAS[0:Measurement.NCONV[IGEOM],IGEOM])
    se = np.zeros((Measurement.NCONV[IGEOM],Measurement.NCONV[IGEOM]))
    for j in range(Measurement.NCONV[IGEOM]):
        se[j,j] = (Measurement.ERRMEAS[j,IGEOM])**2.
    OptimalEstimation.edit_SE(se)

    Variables.edit_XN(OptimalEstimation.XN)

    YN,KK = jacobian_matrix_nemesisGC(IGEOM,Measurement,Spectroscopy,Variables)

    OptimalEstimation.edit_YN(YN)
    OptimalEstimation.edit_KK(KK)

    print('nemesisGC :: Calculating gain matrix')
    OptimalEstimation.calc_gain_matrix()

    print('nemesis :: Calculating cost function')
    OptimalEstimation.calc_phiret()

    OPHI = OptimalEstimation.PHI
    print('chisq/ny = '+str(OptimalEstimation.CHISQ))

    OptimalEstimation.assess()

    #Initializing some variables
    alambda = 1.0   #Marquardt-Levenberg-type 'braking parameter'
    NX11 = np.zeros(OptimalEstimation.NX)
    XN1 = copy(OptimalEstimation.XN)
    NY1 = np.zeros(OptimalEstimation.NY)
    YN1 = copy(OptimalEstimation.YN)

    #Run retrieval for each iteration
    #################################################################

    for it in range(OptimalEstimation.NITER):

        print('nemesis :: Iteration '+str(it)+'/'+str(OptimalEstimation.NITER))

        #Calculating next state vector
        #######################################

        print('nemesis :: Calculating next iterated state vector')
        X_OUT = OptimalEstimation.calc_next_xn()
        #  x_out(nx) is the next iterated value of xn using classical N-L
        #  optimal estimation. However, we want to apply a braking parameter
        #  alambda to stop the new trial vector xn1 being too far from the
        #  last 'best-fit' value xn

        IBRAKE = 0
        while IBRAKE==0: #We continue in this while loop until we do not find problems with the state vector
    
            for j in range(OptimalEstimation.NX):
                XN1[j] = OptimalEstimation.XN[j] + (X_OUT[j]-OptimalEstimation.XN[j])/(1.0+alambda)
            
                #Check to see if log numbers have gone out of range
                if Variables.LX[j]==1:
                    if((XN1[j]>85.) or (XN1[j]<-85.)):
                        print('nemesis :: log(number gone out of range) --- increasing brake')
                        alambda = alambda * 10.
                        IBRAKE = 0
                        if alambda>1.e30:
                            sys.exit('error in nemesis :: Death spiral in braking parameters - stopping')
                        break
                    else:
                        IBRAKE = 1
                else:
                    IBRAKE = 1
                    pass
                        
            if IBRAKE==0:
                continue

        #Calculate test spectrum using trial state vector xn1. 
        #Put output spectrum into temporary spectrum yn1 with
        #temporary kernel matrix kk1. Does it improve the fit? 
        Variables.edit_XN(XN1)
        YN1,KK1 = jacobian_matrix_nemesisGC(IGEOM,Measurement,Spectroscopy,Variables)

        OptimalEstimation1 = copy(OptimalEstimation)
        OptimalEstimation1.edit_YN(YN1)
        OptimalEstimation1.edit_XN(XN1)
        OptimalEstimation1.edit_KK(KK1)
        OptimalEstimation1.calc_phiret()
        print('chisq/ny = '+str(OptimalEstimation1.CHISQ))

        #Does the trial solution fit the data better?
        if (OptimalEstimation1.PHI <= OPHI):
            print('Successful iteration. Updating xn,yn and kk')
            OptimalEstimation.edit_XN(XN1)
            OptimalEstimation.edit_YN(YN1)
            OptimalEstimation.edit_KK(KK1)
            Variables.edit_XN(XN1)

            #Now calculate the gain matrix and averaging kernels
            OptimalEstimation.calc_gain_matrix()

            #Updating the cost function
            OptimalEstimation.calc_phiret()

            #Has the solution converged?
            tphi = 100.0*(OPHI-OptimalEstimation.PHI)/OPHI
            if (tphi>=0.0 and tphi<=OptimalEstimation.PHILIMIT and alambda<1.0):
                print('phi, phlimit : '+str(tphi)+','+str(OptimalEstimation.PHILIMIT))
                print('Phi has converged')
                print('Terminating retrieval')
                break
            else:
                OPHI=OptimalEstimation.PHI
                alambda = alambda*0.3  #reduce Marquardt brake

        else:
            #Leave xn and kk alone and try again with more braking
            alambda = alambda*10.0  #increase Marquardt brake

    #Calculating output parameters
    ######################################################

    #Calculating retrieved covariance matrices
    OptimalEstimation.calc_serr()
 
    pickle.dump(OptimalEstimation.XN,f,pickle.HIGHEST_PROTOCOL)
    pickle.dump(OptimalEstimation.ST,f,pickle.HIGHEST_PROTOCOL)


    print('Results :: ',Measurement.TANHE[IGEOM,0],'km')
    print('T',OptimalEstimation.XN[1],'+-',OptimalEstimation.ST[1,1],'K')
    print('Nlos_CO2',np.exp(OptimalEstimation.XN[2]),'+-',OptimalEstimation.ST[2,2],'cm-2')
    print('(13C)/(12C)',OptimalEstimation.XN[3],'+-',OptimalEstimation.ST[3,3],'VPDB')
    print('(18O)/(16O)',OptimalEstimation.XN[4],'+-',OptimalEstimation.ST[4,4],'VSMOW')
    print('(17O)/(16O)',OptimalEstimation.XN[5],'+-',OptimalEstimation.ST[5,5],'VSMOW')
    print('Nlos_H2O',np.exp(OptimalEstimation.XN[6]),'+-',OptimalEstimation.ST[6,6],'cm-2')


f.close()


