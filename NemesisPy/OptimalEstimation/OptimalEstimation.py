# NAME:
#       files.py (nemesislib)
#
# DESCRIPTION:
#
#	This library contains functions to read and write files that are formatted as 
#	required by the NEMESIS radiative transfer code         
#
# CATEGORY:
#
#	NEMESIS
# 
# MODIFICATION HISTORY: Juan Alday 15/03/2021

import numpy as np
from struct import *
import sys,os
import matplotlib.pyplot as plt
from NemesisPy import *
from copy import *


###############################################################################################

def coreretOE(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,\
                 NITER=10,PHILIMIT=0.1,NCores=1,nemesisSO=False,write_itr=True):


    """
        FUNCTION NAME : coreretOE()
        
        DESCRIPTION : 

            This subroutine runs the Optimal Estimation iterarive algorithm to solve the inverse
            problem and find the set of parameters that fit the spectral measurements and are closest
            to the a priori estimates of the parameters.

        INPUTS :
       
            runname :: Name of the Nemesis run
            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the spectroscopic parameters of gaseous species
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations

        OPTIONAL INPUTS:

            NITER :: Number of iterations in retrieval
            PHILIMIT :: Percentage convergence limit. If the percentage reduction of the cost function PHI
                        is less than philimit then the retrieval is deemed to have converged.

            nemesisSO :: If True, the retrieval uses the function jacobian_nemesisSO(), adapated specifically
                         for solar occultation observations, rather than the more general jacobian_nemesis() function.

        OUTPUTS :

            OptimalEstimation :: Python class defining all the variables required as input or output
                                 from the Optimal Estimation retrieval
 
        CALLING SEQUENCE:
        
            OptimalEstimation = coreretOE(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,Layer)
 
        MODIFICATION HISTORY : Juan Alday (06/08/2021)

    """

    from NemesisPy import OptimalEstimation_0
    from NemesisPy import ForwardModel_0
    from NemesisPy import jacobian_nemesisSO,jacobian_nemesis

    #Creating class and including inputs
    #############################################

    OptimalEstimation = OptimalEstimation_0()

    OptimalEstimation.NITER = NITER
    OptimalEstimation.PHILIMIT = PHILIMIT
    OptimalEstimation.NX = Variables.NX
    OptimalEstimation.NY = Measurement.NY
    OptimalEstimation.edit_XA(Variables.XA)
    OptimalEstimation.edit_XN(Variables.XN)
    OptimalEstimation.edit_SA(Variables.SA)
    OptimalEstimation.edit_Y(Measurement.Y)
    OptimalEstimation.edit_SE(Measurement.SE)

    #Opening .itr file
    #################################################################

    if OptimalEstimation.NITER>0:
        if write_itr==True:
            fitr = open(runname+'.itr','w')
            fitr.write("\t %i \t %i \t %i\n" % (OptimalEstimation.NX,OptimalEstimation.NY,OptimalEstimation.NITER))

    #Calculate the first measurement vector and jacobian matrix
    #################################################################

    ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
    print('nemesis :: Calculating Jacobian matrix KK')
    YN,KK = ForwardModel.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)
    
    #if nemesisSO==True:
    #    print('nemesisSO :: Calculating Jacobian matrix KK')
    #    YN,KK = ForwardModel.jacobian_nemesisSO(NCores=NCores)
    #    #YN,KK = jacobian_nemesisSO(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,NCores=NCores)
    #else:
    #    print('nemesis :: Calculating Jacobian matrix KK')
    #    YN,KK = jacobian_nemesis(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,NCores=NCores)

    OptimalEstimation.edit_YN(YN)
    OptimalEstimation.edit_KK(KK)

    #Calculate gain matrix and average kernels
    #################################################################

    print('nemesis :: Calculating gain matrix')
    OptimalEstimation.calc_gain_matrix()

    #Calculate initial value of cost function phi
    #################################################################

    print('nemesis :: Calculating cost function')
    OptimalEstimation.calc_phiret()

    OPHI = OptimalEstimation.PHI
    print('chisq/ny = '+str(OptimalEstimation.CHISQ))

    #Assessing whether retrieval is going to be OK
    #################################################################

    OptimalEstimation.assess()

    #Run retrieval for each iteration
    #################################################################

    #Initializing some variables
    alambda = 1.0   #Marquardt-Levenberg-type 'braking parameter'
    NX11 = np.zeros(OptimalEstimation.NX)
    XN1 = copy(OptimalEstimation.XN)
    NY1 = np.zeros(OptimalEstimation.NY)
    YN1 = copy(OptimalEstimation.YN)

    for it in range(OptimalEstimation.NITER):

        print('nemesis :: Iteration '+str(it)+'/'+str(OptimalEstimation.NITER))

        if write_itr==True:
            
        #Writing into .itr file
        ####################################

            fitr.write('%10.5f %10.5f \n' % (OptimalEstimation.CHISQ,OptimalEstimation.PHI))
            for i in range(OptimalEstimation.NX):fitr.write('%10.5f \n' % (XN1[i]))
            for i in range(OptimalEstimation.NX):fitr.write('%10.5f \n' % (OptimalEstimation.XA[i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.Y[i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.SE[i,i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (YN1[i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.YN[i]))
            for i in range(OptimalEstimation.NX):
                for j in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.KK[j,i]))


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
                        
            #Check to see if any VMRs or other parameters have gone negative.
            Variables1 = copy(Variables)
            Variables1.XN = XN1

            ForwardModel1 = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables1)
            #Variables1 = copy(Variables)
            #Variables1.XN = XN1
            #Measurement1 = copy(Measurement)
            #Atmosphere1 = copy(Atmosphere)
            #Scatter1 = copy(Scatter)
            #Stellar1 = copy(Stellar)
            #Surface1 = copy(Surface)
            #Spectroscopy1 = copy(Spectroscopy)
            #Layer1 = copy(Layer)
            #flagh2p = False
            #xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)
            ForwardModel1.subprofretg()

            #if(len(np.where(Atmosphere1.VMR<0.0))>0):
            #    print('nemesisSO :: VMR has gone negative --- increasing brake')
            #    alambda = alambda * 10.
            #    IBRAKE = 0
            #    continue
            
            #iwhere = np.where(Atmosphere1.T<0.0)
            iwhere = np.where(ForwardModel1.AtmosphereX.T<0.0)
            if(len(iwhere[0])>0):
                print('nemesis :: Temperature has gone negative --- increasing brake')
                alambda = alambda * 10.
                IBRAKE = 0
                continue


        #Calculate test spectrum using trial state vector xn1. 
        #Put output spectrum into temporary spectrum yn1 with
        #temporary kernel matrix kk1. Does it improve the fit? 
        Variables.edit_XN(XN1)
        print('nemesis :: Calculating Jacobian matrix KK')

        ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
        YN1,KK1 = ForwardModel.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)
        #if nemesisSO==True:
        #    print('nemesisSO :: Calculating Jacobian matrix KK')
        #    ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
        #    YN1,KK1 = ForwardModel.jacobian_nemesisSO(NCores=NCores)
        #    #YN1,KK1 = jacobian_nemesisSO(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,NCores=NCores)
        #else:
        #    print('nemesis :: Calculating Jacobian matrix KK')
        #    YN1,KK1 = jacobian_nemesis(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,NCores=NCores)

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

    #Make sure errors stay as a priori for kiter < 0
    if OptimalEstimation.NITER<0:
        OptimalEstimation.ST = copy(OptimalEstimation.SA)

    #Closing .itr file
    if write_itr==True:
        if OptimalEstimation.NITER>0:
            fitr.close()

    #Writing the contribution of each gas to .gcn file
    #if nemesisSO==True:
    #    calc_gascn(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

    return OptimalEstimation

