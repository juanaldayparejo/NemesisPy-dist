# NAME:
#       Radtrans.py (nemesislib)
#
# DESCRIPTION:
#
#	This library contains functions to perform the radiative transfer calculations
#
#
# CATEGORY:
#
#	NEMESIS
# 
# MODIFICATION HISTORY: Juan Alday 15/03/2021

import numpy as np
from struct import *
import pylab
import sys,os,errno,shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as font_manager
import matplotlib as mpl
from NemesisPy import *
import ray
from numba import jit


###############################################################################################

def calc_gascn(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,nemesisSO=True,Write_GCN=True):
    
    """
        FUNCTION NAME : calc_gascn()
        
        DESCRIPTION : This function computes several forward models using only one radiatively active gas 
                        at a time, in order to evaluate what the contribution from each gas is to the spectrum
        
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
        
        OPTIONAL INPUTS:

            nemesisSO :: If True, then the calculation type is set to model a solar occultation observation
            Write_GCN :: If True, a .gcn file is written with the results
        
        OUTPUTS :
        
            SPECMOD(NCONV,NGEOM,NGAS) :: Modelled spectra for each active gas
        
        CALLING SEQUENCE:
        
            calc_gascn(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    from copy import copy
    from NemesisPy import write_gcn

    #Calling the forward model NGAS times to calculate the measurement vector for each case radiatively active gas

    SPECMODGCN = np.zeros([Measurement.NCONV.max(),Measurement.NGEOM,Spectroscopy.NGAS])

    for igas in range(Spectroscopy.NGAS):

        print('calc_gascn :: Calculating forward model for active gas '+str(igas)+'/'+str(Spectroscopy.NGAS))

        Spectroscopy1 = copy(Spectroscopy)
        Spectroscopy1.NGAS = 1
        Spectroscopy1.ID = [Spectroscopy.ID[igas]] ; Spectroscopy1.ISO = [Spectroscopy.ISO[igas]]
        Spectroscopy1.LOCATION = [Spectroscopy.LOCATION[igas]]
        if Spectroscopy1.ILBL==0:
            k = np.zeros([Spectroscopy1.NWAVE,Spectroscopy1.NG,Spectroscopy1.NP,Spectroscopy1.NT,1])
            k[:,:,:,:,0] = Spectroscopy.K[:,:,:,:,igas]
            Spectroscopy1.edit_K(k)
            del k  #Delete array to free memory
        else:
            k = np.zeros([Spectroscopy1.NWAVE,Spectroscopy1.NP,Spectroscopy1.NT,1])
            k[:,:,:,0] = Spectroscopy.K[:,:,:,igas]
            Spectroscopy1.edit_K(k)
            del k  #Delete array to free memory

        if nemesisSO==True:
            SPECMOD1 = nemesisSOfm(runname,Variables,Measurement,Atmosphere,Spectroscopy1,Scatter,Stellar,Surface,CIA,Layer)
        else:
            sys.exit('error in calc_gascn() :: It has only been implemented yet for solar occultation observations')
            
        SPECMODGCN[:,:,igas] = SPECMOD1[:,:]

        if Write_GCN==True:
            #Writing the .gcn file if required
            write_gcn(runname,Spectroscopy.NGAS,Spectroscopy.ID,Spectroscopy.ISO,Measurement.NGEOM,Measurement.NCONV,Measurement.VCONV,SPECMODGCN)

    return SPECMODGCN

###############################################################################################

def nemesisSOfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer):
    
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

    from NemesisPy.Models import subprofretg,subspecret
    from NemesisPy.Path import AtmCalc_0,Path_0,calc_pathg_SO
    #from NemesisPy.Radtrans import lblconv
    from NemesisPy import find_nearest
    from scipy import interpolate
    from copy import copy

    #First we change the reference atmosphere taking into account the parameterisations in the state vector
    Variables1 = copy(Variables)
    Measurement1 = copy(Measurement)
    Atmosphere1 = copy(Atmosphere)
    Scatter1 = copy(Scatter)
    Stellar1 = copy(Stellar)
    Surface1 = copy(Surface)
    Layer1 = copy(Layer)
    Spectroscopy1 = copy(Spectroscopy)
    CIA1 = copy(CIA)
    flagh2p = False

    xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)

    """
    #Based on the new reference atmosphere, we split the atmosphere into layers
    #In solar occultation LAYANG = 90.0
    LAYANG = 90.0

    BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH, DTE, DAM, DCO\
        = Layer1.integrateg(H=Atmosphere1.H,P=Atmosphere1.P,T=Atmosphere1.T, LAYANG=LAYANG, ID=Atmosphere1.ID,VMR=Atmosphere1.VMR, DUST=Atmosphere1.DUST)

    #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
    NCALC = Layer.NLAY    #Number of calculations (geometries) to be performed
    AtmCalc_List = []
    for ICALC in range(NCALC):
        iAtmCalc = AtmCalc_0(Layer1,LIMB=True,BOTLAY=ICALC,ANGLE=90.0,IPZEN=0)
        AtmCalc_List.append(iAtmCalc)
    
    #We initialise the total Path class, indicating that the calculations can be combined
    Path1 = Path_0(AtmCalc_List,COMBINE=True)
    """

    Layer1,Path1 = calc_pathg_SO(Atmosphere1,Scatter1,Measurement1,Layer1)
    BASEH_TANHE = np.zeros(Path1.NPATH)
    for i in range(Path1.NPATH):
        BASEH_TANHE[i] = Layer1.BASEH[Path1.LAYINC[int(Path1.NLAYIN[i]/2),i]]/1.0e3

    #Calling CIRSrad to calculate the spectra
    print('Running CIRSradg')
    SPECOUT,dSPECOUT2,dTSURF = CIRSradg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,CIA1,Layer1,Path1)

    #Mapping the gradients from Layer properties to Profile properties 
    print('Mapping gradients from Layer to Profile')
    #Calculating the elements from NVMR+2+NDUST that need to be mapped
    incpar = []
    for i in range(Atmosphere1.NVMR+2+Atmosphere1.NDUST):
        if np.mean(xmap[:,i,:])!=0.0:
            incpar.append(i)

    dSPECOUT1 = map2pro(dSPECOUT2,Measurement1.NWAVE,Atmosphere1.NVMR,Atmosphere1.NDUST,Atmosphere1.NP,Path1.NPATH,Path1.NLAYIN,Path1.LAYINC,Layer1.DTE,Layer1.DAM,Layer1.DCO,INCPAR=incpar)
    #(NWAVE,NVMR+2+NDUST,NPRO,NPATH)
    del dSPECOUT2

    #Mapping the gradients from Profile properties to elements in state vector
    print('Mapping gradients from Profile to State Vector')
    dSPECOUT = map2xvec(dSPECOUT1,Measurement1.NWAVE,Atmosphere1.NVMR,Atmosphere1.NDUST,Atmosphere1.NP,Path1.NPATH,Variables1.NX,xmap)
    #(NWAVE,NPATH,NX)
    del dSPECOUT1

    #Interpolating the spectra to the correct altitudes defined in Measurement
    SPECMOD = np.zeros([Measurement1.NWAVE,Measurement1.NGEOM])
    dSPECMOD = np.zeros([Measurement1.NWAVE,Measurement1.NGEOM,Variables.NX])
    for i in range(Measurement.NGEOM):

        #Find altitudes above and below the actual tangent height
        base0,ibase = find_nearest(BASEH_TANHE,Measurement1.TANHE[i])
        if base0<=Measurement1.TANHE[i]:
            ibasel = ibase
            ibaseh = ibase + 1
        else:
            ibasel = ibase - 1
            ibaseh = ibase

        if ibaseh>Path1.NPATH-1:
            SPECMOD[:,i] = SPECOUT[:,ibasel]
            dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]
        else:
            fhl = (Measurement1.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
            fhh = (BASEH_TANHE[ibaseh]-Measurement1.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

            SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)
            dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]*(1.-fhl) + dSPECOUT[:,ibaseh,:]*(1.-fhh)


    """
    NP = Measurement.NWAVE * Atmosphere1.NP
    xx = np.linspace(0,NP-1,NP)
    for ipath in range(Measurement.NGEOM):
        fig,ax1 = plt.subplots(1,1,figsize=(10,3))
        ll = 0
        for i in range(Atmosphere1.NP):
            ax1.plot(xx[ll:ll+Measurement.NWAVE],dSPECMOD[0:Measurement.NWAVE,ipath,i])
            ll = ll + Measurement.NWAVE
        plt.tight_layout()
        plt.show()
    sys.exit()
    """

    #Applying any changes to the spectra required by the state vector
    SPECMOD,dSPECMOD = subspecret(Measurement1,Variables1,SPECMOD,dSPECMOD)

    """
    for ipath in range(Measurement1.NGEOM):
        fig,ax1=plt.subplots(1,1,figsize=(10,3))
        NY1 = Measurement1.NWAVE * Variables1.NX
        xx = np.linspace(0,NY1-1,NY1)
        ll = 0
        for ix in range(Variables1.NX):
            ax1.plot(xx[ll:ll+Measurement1.NWAVE],dSPECMOD[:,ipath,ix])
            ll = ll + Measurement1.NWAVE
        plt.tight_layout()
        plt.show()
    sys.exit()
    """

    #Convolving the spectrum with the instrument line shape
    print('Convolving spectra and gradients with instrument line shape')
    if Spectroscopy1.ILBL==0:
        SPECONV,dSPECONV = Measurement1.convg(SPECMOD,dSPECMOD,IGEOM='All')
    elif Spectroscopy1.ILBL==2:
        SPECONV,dSPECONV = Measurement1.lblconvg(SPECMOD,dSPECMOD,IGEOM='All')

    """
    for ipath in range(Measurement1.NGEOM):
        fig,ax1=plt.subplots(1,1,figsize=(10,3))
        NY1 = Measurement1.NCONV[0] * Variables1.NX
        xx = np.linspace(0,NY1-1,NY1)
        ll = 0
        for ix in range(Variables1.NX):
            ax1.plot(xx[ll:ll+Measurement1.NCONV[ipath]],dSPECONV[:,ipath,ix])
            ll = ll + Measurement1.NCONV[ipath]
        plt.tight_layout()
        plt.show()
    sys.exit()
    """

    return SPECONV,dSPECONV

###############################################################################################

def nemesisSOfm(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer):
    
    """
        FUNCTION NAME : nemesisSOfm()
        
        DESCRIPTION : This function computes a forward model for a solar occultation observation
        
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
        
        CALLING SEQUENCE:
        
            nemesisSOfm(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    from NemesisPy.Models import subprofretg,subspecret
    from NemesisPy.Path import AtmCalc_0,Path_0,calc_path_SO
    from NemesisPy import find_nearest
    from scipy import interpolate
    from copy import copy

    #First we change the reference atmosphere taking into account the parameterisations in the state vector
    Variables1 = copy(Variables)
    Measurement1 = copy(Measurement)
    Atmosphere1 = copy(Atmosphere)
    Scatter1 = copy(Scatter)
    Stellar1 = copy(Stellar)
    Surface1 = copy(Surface)
    Layer1 = copy(Layer)
    Spectroscopy1 = copy(Spectroscopy)
    CIA1 = copy(CIA)
    flagh2p = False

    xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)

    Layer1,Path1 = calc_path_SO(Atmosphere1,Scatter1,Measurement1,Layer1)
    BASEH_TANHE = np.zeros(Path1.NPATH)
    for i in range(Path1.NPATH):
        BASEH_TANHE[i] = Layer1.BASEH[Path1.LAYINC[int(Path1.NLAYIN[i]/2),i]]/1.0e3

    #Calling CIRSrad to calculate the spectra
    SPECOUT = CIRSrad(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,CIA1,Layer1,Path1)
    

    #Interpolating the spectra to the correct altitudes defined in Measurement
    SPECMOD = np.zeros([Measurement1.NWAVE,Measurement1.NGEOM])
    dSPECMOD = np.zeros([Measurement1.NWAVE,Measurement1.NGEOM,Variables.NX])
    for i in range(Measurement.NGEOM):

        #Find altitudes above and below the actual tangent height
        base0,ibase = find_nearest(BASEH_TANHE,Measurement1.TANHE[i])
        if base0<=Measurement1.TANHE[i]:
            ibasel = ibase
            ibaseh = ibase + 1
        else:
            ibasel = ibase - 1
            ibaseh = ibase

        if ibaseh>Path1.NPATH-1:
            SPECMOD[:,i] = SPECOUT[:,ibasel]
        else:
            fhl = (Measurement1.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
            fhh = (BASEH_TANHE[ibaseh]-Measurement1.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

            SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)

    #Applying any changes to the spectra required by the state vector
    SPECMOD,dSPECMOD = subspecret(Measurement1,Variables1,SPECMOD,dSPECMOD)

    #Convolving the spectrum with the instrument line shape
    print('Convolving spectra and gradients with instrument line shape')
    if Spectroscopy1.ILBL==0:
        SPECONV,dSPECONV = Measurement1.convg(SPECMOD,dSPECMOD,IGEOM='All')
    elif Spectroscopy1.ILBL==2:
        SPECONV,dSPECONV = Measurement1.lblconvg(SPECMOD,dSPECMOD,IGEOM='All')

    MakePlot=False
    if MakePlot==True:

        import matplotlib as matplotlib

        fig,ax1 = plt.subplots(1,1,figsize=(13,4))

        colormap = 'nipy_spectral'
        norm = matplotlib.colors.Normalize(vmin=0.,vmax=Measurement.TANHE.max())
        c_m = plt.cm.get_cmap(colormap,360)
        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        for igeom in range(Measurement.NGEOM):
            #for iconv in range(Measurement.NCONV[igeom]):

            ax1.plot(Measurement.VCONV[0:Measurement.NCONV[igeom],igeom],SPECONV[0:Measurement.NCONV[igeom],igeom],c=s_m.to_rgba([Measurement.TANHE[igeom,0]]))

        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Transmission')
        ax1.grid()

        cax = plt.axes([0.92, 0.15, 0.02, 0.7])   #Bottom
        cbar2 = plt.colorbar(s_m,cax=cax,orientation='vertical')
        cbar2.set_label('Altitude (km)')

        plt.show()

    return SPECONV

###############################################################################################

@ray.remote
def nemesisSOfm_parallel(ix,runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,xnx):

    from NemesisPy import nemesisSOfm
    from copy import copy

    Variables1 = copy(Variables)
    Measurement1 = copy(Measurement)
    Atmosphere1 = copy(Atmosphere)
    Spectroscopy1 = copy(Spectroscopy)
    Scatter1 = copy(Scatter)
    Stellar1 = copy(Stellar)
    Surface1 = copy(Surface)
    CIA1 = copy(CIA)
    Layer1 = copy(Layer)
    Variables1.XN = xnx[0:Variables1.NX,ix]
    SPECMOD = nemesisSOfm(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,CIA1,Layer1)
    YN = np.resize(np.transpose(SPECMOD),[Measurement.NY])
    return YN

###############################################################################################

def jacobian_nemesisSO(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,MakePlot=False,NCores=1):

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

    nproc = Variables.NX+1 #Number of times we need to run the forward model

    #Constructing state vector after perturbation of each of the elements and storing in matrix

    Variables.calc_DSTEP() #Calculating the step size for the perturbation of each element
    nxn = Variables.NX+1
    xnx = np.zeros([Variables.NX,nxn])
    for i in range(Variables.NX+1):
        if i==0:   #First element is the normal state vector
            xnx[0:Variables.NX,i] = Variables.XN[0:Variables.NX]
        else:      #Perturbation of each element
            xnx[0:Variables.NX,i] = Variables.XN[0:Variables.NX]
            #xnx[i-1,i] = Variables.XN[i-1]*1.05
            xnx[i-1,i] = Variables.XN[i-1] + Variables.DSTEP[i-1]
            if Variables.XN[i-1]==0.0:
                xnx[i-1,i] = 0.05



    #Because of the parallelisation, the parameters that are kept fixed need to be located at the end of the 
    #state vector, otherwise the code crashes
    ic = 0
    for i in range(Variables.NX):

        if ic==1:
            if Variables.FIX[i]==0:
                sys.exit('error :: Fixed parameters in the state vector must be located at the end of the array')

        if Variables.FIX[i]==1:
            ic = 1

    #################################################################################
    # Calculating the first forward model and the analytical part of Jacobian
    #################################################################################

    #Variables.NUM[:] = 1

    ian1 = np.where(Variables.NUM==0)  #Gradients calculated using CIRSradg
    ian1 = ian1[0]

    iYN = 0
    KK = np.zeros([Measurement.NY,Variables.NX])

    if len(ian1)>0:

        print('Calculating analytical part of the Jacobian :: Calling nemesisSOfmg ')

        SPECMOD,dSPECMOD = nemesisSOfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

        YN = np.resize(np.transpose(SPECMOD),[Measurement.NY])
        for ix in range(Variables.NX):
            KK[:,ix] = np.resize(np.transpose(dSPECMOD[:,:,ix]),[Measurement.NY])

        iYN = 1 #Indicates that some of the gradients and the measurement vector have already been caculated


    #################################################################################
    # Calculating all the required forward models for numerical differentiation
    #################################################################################

    inum1 = np.where( (Variables.NUM==1) & (Variables.FIX==0) )
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
    YNtot = np.zeros([Measurement.NY,nfm])

    print('Calculating numerical part of the Jacobian :: running '+str(nfm)+' forward models ')
    if NCores>1:
        ray.init(num_cpus=NCores)
        YNtot_ids = []
        SpectroscopyP = ray.put(Spectroscopy)
        for ix in range(nfm):
            YNtot_ids.append(nemesisSOfm_parallel.remote(ixrun[ix],runname,Variables,Measurement,Atmosphere,SpectroscopyP,Scatter,Stellar,Surface,CIA,Layer,xnx))

        #Block until the results have finished and get the results.
        YNtot1 = ray.get(YNtot_ids)
        for ix in range(nfm):
            YNtot[0:Measurement.NY,ix] = YNtot1[ix]
        ray.shutdown()

    else:
    
        for ifm in range(nfm):
            print('Calculating forward model '+str(ifm)+'/'+str(nfm))
            Variables1 = copy(Variables)
            Variables1.XN = xnx[0:Variables1.NX,ixrun[ifm]]
            SPECMOD = nemesisSOfm(runname,Variables1,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
            YNtot[0:Measurement.NY,ifm] = np.resize(np.transpose(SPECMOD),[Measurement.NY])

    if iYN==0:
        YN = np.zeros(Measurement.NY)
        YN[:] = YNtot[0:Measurement.NY,0]

    #################################################################################
    # Calculating the Jacobian matrix
    #################################################################################

    for i in range(len(inum)):

        if iYN==0:
            ifm = i + 1
        else:
            ifm = i

        xn1 = Variables.XN[inum[i]] * 1.05
        if xn1==0.0:
            xn1=0.05
        if Variables.FIX[i] == 0:
                KK[:,inum[i]] = (YNtot[:,ifm]-YN)/(xn1-Variables.XN[inum[i]])


    #################################################################################
    # Making summary plot if required
    ################################################################################# 

    if MakePlot==True:

        import matplotlib as matplotlib

        #Plotting the measurement vector

        fig,ax1 = plt.subplots(1,1,figsize=(13,4))

        colormap = 'nipy_spectral'
        norm = matplotlib.colors.Normalize(vmin=0.,vmax=Measurement.TANHE.max())
        c_m = plt.cm.get_cmap(colormap,360)
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        ix = 0
        for igeom in range(Measurement.NGEOM):
            ax1.plot(Measurement.VCONV[0:Measurement.NCONV[igeom],igeom],YN[ix:ix+Measurement.NCONV[igeom]],c=s_m.to_rgba([Measurement.TANHE[igeom,0]]))
            ix = ix + Measurement.NCONV[igeom]

        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Transmission')
        ax1.grid()

        cax = plt.axes([0.92, 0.15, 0.02, 0.7])   #Bottom
        cbar2 = plt.colorbar(s_m,cax=cax,orientation='vertical')
        cbar2.set_label('Altitude (km)')


        fig, ax = plt.subplots(1,1,figsize=(10,3))
        ax.set_xlabel('Measurement vector y')
        ax.set_ylabel('State vector x')
        ax.imshow(np.transpose(KK),cmap='hot',aspect='auto',origin='lower')
        ax.grid()
        plt.tight_layout()
        plt.show()

    return YN,KK


###############################################################################################

def nemesisfm(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer):
    
    """
        FUNCTION NAME : nemesisfm()
        
        DESCRIPTION : This function computes a forward model
        
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
        
        CALLING SEQUENCE:
        
            nemesisfm(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    from NemesisPy.Models import subprofretg
    from NemesisPy.Path import AtmCalc_0,Path_0,calc_path
    from NemesisPy.Radtrans import lblconv
    from NemesisPy import find_nearest,subspecret
    from scipy import interpolate
    from copy import copy

    #Estimating the number of calculations that will need to be computed to model the spectra
    #included in the Measurement class (taking into account al geometries and averaging points)
    NCALC = np.sum(Measurement.NAV)
    SPECONV = np.zeros(Measurement.MEAS.shape) #Initalise the array where the spectra will be stored (NWAVE,NGEOM)
    for IGEOM in range(Measurement.NGEOM):

        #Calculating new wave array
        if Spectroscopy.ILBL==0:
            Measurement.wavesetb(Spectroscopy,IGEOM=IGEOM)
        if Spectroscopy.ILBL==2:
            Measurement.wavesetc(Spectroscopy,IGEOM=IGEOM)

        #Initialise array for averaging spectra (if required by NAV>1)
        SPEC = np.zeros(Measurement.NWAVE)
        dSPEC = np.zeros((Measurement.NWAVE,Variables.NX))
        WGEOMTOT = 0.0
        for IAV in range(Measurement.NAV[IGEOM]):
            
            #Making copy of classes to avoid overwriting them
            Variables1 = copy(Variables)
            Measurement1 = copy(Measurement)
            Atmosphere1 = copy(Atmosphere)
            Scatter1 = copy(Scatter)
            Stellar1 = copy(Stellar)
            Surface1 = copy(Surface)
            Spectroscopy1 = copy(Spectroscopy)
            Layer1 = copy(Layer)
            CIA1 = copy(CIA)
            flagh2p = False

            #Updating the required parameters based on the current geometry
            Scatter1.SOL_ANG = Measurement1.SOL_ANG[IGEOM,IAV]
            Scatter1.EMISS_ANG = Measurement1.EMISS_ANG[IGEOM,IAV]
            Scatter1.AZI_ANG = Measurement1.AZI_ANG[IGEOM,IAV]

            if Spectroscopy.ILBL==0:
                Measurement1.wavesetb(Spectroscopy,IGEOM=IGEOM)
            if Spectroscopy.ILBL==2:
                Measurement1.wavesetc(Spectroscopy,IGEOM=IGEOM)

            #Changing the different classes taking into account the parameterisations in the state vector
            xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)

            #Calling gsetpat to split the new reference atmosphere and calculate the path
            Layer1,Path1 = calc_path(Atmosphere1,Scatter1,Measurement1,Layer1)

            #Calling CIRSrad to perform the radiative transfer calculations
            SPEC1 = CIRSrad(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy,Scatter1,Stellar1,Surface1,CIA1,Layer1,Path1)

            #Averaging the spectra in case NAV>1
            if Measurement.NAV[IGEOM]>1:
                SPEC[:] = SPEC[:] + Measurement.WGEOM[IGEOM,IAV] * SPEC1[:,0]
                WGEOMTOT = WGEOMTOT + Measurement.WGEOM[IGEOM,IAV]
            else:
                SPEC[:] = SPEC1[:,0]

        if Measurement.NAV[IGEOM]>1:
            SPEC[:] = SPEC[:] / WGEOMTOT

        #Applying any changes to the spectra required by the state vector
        SPEC,dSPEC = subspecret(Measurement1,Variables1,SPEC,dSPEC)

        #Convolving the spectra with the Instrument line shape
        if Spectroscopy.ILBL==0: #k-tables

            if os.path.exists(runname+'.fwh')==True:
                FWHMEXIST=runname
            else:
                FWHMEXIST=''

            SPECONV1 = Measurement.conv(SPEC,IGEOM=IGEOM,FWHMEXIST='')

        elif Spectroscopy.ILBL==2: #LBL-tables

            SPECONV1 = Measurement.lblconv(SPEC,IGEOM=IGEOM)

        SPECONV[0:Measurement.NCONV[IGEOM],IGEOM] = SPECONV1[0:Measurement.NCONV[IGEOM]]

    return SPECONV

###############################################################################################

def nemesisfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer):
    
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

    from NemesisPy.Models import subprofretg
    from NemesisPy.Path import AtmCalc_0,Path_0,calc_pathg
    from NemesisPy.Radtrans import lblconv
    from NemesisPy import find_nearest,subspecret
    from scipy import interpolate
    from copy import copy

    #Estimating the number of calculations that will need to be computed to model the spectra
    #included in the Measurement class (taking into account al geometries and averaging points)
    NCALC = np.sum(Measurement.NAV)
    SPECONV = np.zeros(Measurement.MEAS.shape) #Initalise the array where the spectra will be stored (NWAVE,NGEOM)
    dSPECONV = np.zeros((Measurement.NCONV.max(),Measurement.NGEOM,Variables.NX)) #Initalise the array where the gradients will be stored (NWAVE,NGEOM,NX)
    for IGEOM in range(Measurement.NGEOM):

        #Calculating new wave array
        if Spectroscopy.ILBL==0:
            Measurement.wavesetb(Spectroscopy,IGEOM=IGEOM)
        if Spectroscopy.ILBL==2:
            Measurement.wavesetc(Spectroscopy,IGEOM=IGEOM)

        #Initialise array for averaging spectra (if required by NAV>1)
        SPEC = np.zeros(Measurement.NWAVE)
        dSPEC = np.zeros((Measurement.NWAVE,Variables.NX))
        WGEOMTOT = 0.0
        for IAV in range(Measurement.NAV[IGEOM]):
            
            #Making copy of classes to avoid overwriting them
            Variables1 = copy(Variables)
            Measurement1 = copy(Measurement)
            Atmosphere1 = copy(Atmosphere)
            Scatter1 = copy(Scatter)
            Stellar1 = copy(Stellar)
            Surface1 = copy(Surface)
            Spectroscopy1 = copy(Spectroscopy)
            Layer1 = copy(Layer)
            CIA1 = copy(CIA)
            flagh2p = False

            #Updating the required parameters based on the current geometry
            Scatter1.SOL_ANG = Measurement1.SOL_ANG[IGEOM,IAV]
            Scatter1.EMISS_ANG = Measurement1.EMISS_ANG[IGEOM,IAV]
            Scatter1.AZI_ANG = Measurement1.AZI_ANG[IGEOM,IAV]

            if Spectroscopy.ILBL==0:
                Measurement1.wavesetb(Spectroscopy,IGEOM=IGEOM)
            if Spectroscopy.ILBL==2:
                Measurement1.wavesetc(Spectroscopy,IGEOM=IGEOM)

            #Changing the different classes taking into account the parameterisations in the state vector
            xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)

            #Calling gsetpat to split the new reference atmosphere and calculate the path
            Layer1,Path1 = calc_pathg(Atmosphere1,Scatter1,Measurement1,Layer1)

            #Calling CIRSrad to perform the radiative transfer calculations
            SPEC1,dSPEC3,dTSURF = CIRSradg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy,Scatter1,Stellar1,Surface1,CIA1,Layer1,Path1)

            #Mapping the gradients from Layer properties to Profile properties 
            print('Mapping gradients from Layer to Profile')
            dSPEC2 = map2pro(dSPEC3,Measurement1.NWAVE,Atmosphere1.NVMR,Atmosphere1.NDUST,Atmosphere1.NP,Path1.NPATH,Path1.NLAYIN,Path1.LAYINC,Layer1.DTE,Layer1.DAM,Layer1.DCO)
            #(NWAVE,NVMR+2+NDUST,NPRO,NPATH)

            #Mapping the gradients from Profile properties to elements in state vector
            print('Mapping gradients from Profile to State Vector')
            dSPEC1 = map2xvec(dSPEC2,Measurement1.NWAVE,Atmosphere1.NVMR,Atmosphere1.NDUST,Atmosphere1.NP,Path1.NPATH,Variables1.NX,xmap)
            #(NWAVE,NPATH,NX)

            #Adding the temperature surface gradient if required    
            if Variables1.JSURF>=0:
                dSPEC1[:,0,Variables1.JSURF] = dTSURF[:,0]

            #Averaging the spectra in case NAV>1
            if Measurement.NAV[IGEOM]>1:
                SPEC[:] = SPEC[:] + Measurement.WGEOM[IGEOM,IAV] * SPEC1[:,0]
                dSPEC[:,:] = dSPEC[:,:] + Measurement.WGEOM[IGEOM,IAV] * dSPEC1[:,0,:]
                WGEOMTOT = WGEOMTOT + Measurement.WGEOM[IGEOM,IAV]
            else:
                SPEC[:] = SPEC1[:,0]
                dSPEC[:,:] = dSPEC1[:,0,:]

        if Measurement.NAV[IGEOM]>1:
            SPEC[:] = SPEC[:] / WGEOMTOT
            dSPEC[:,:] = dSPEC[:,:] / WGEOMTOT

        #Applying any changes to the spectra required by the state vector
        SPEC,dSPEC = subspecret(Measurement1,Variables1,SPEC,dSPEC)

        #Convolving the spectra with the Instrument line shape
        if Spectroscopy.ILBL==0: #k-tables

            if os.path.exists(runname+'.fwh')==True:
                FWHMEXIST=runname
            else:
                FWHMEXIST=''

            SPECONV1,dSPECONV1 = Measurement.convg(SPEC,dSPEC,IGEOM=IGEOM,FWHMEXIST='')

        elif Spectroscopy.ILBL==2: #LBL-tables

            SPECONV1,dSPECONV1 = Measurement.lblconvg(SPEC,dSPEC,IGEOM=IGEOM)

        SPECONV[0:Measurement.NCONV[IGEOM],IGEOM] = SPECONV1[0:Measurement.NCONV[IGEOM]]
        dSPECONV[0:Measurement.NCONV[IGEOM],IGEOM,:] = dSPECONV1[0:Measurement.NCONV[IGEOM],:]

    return SPECONV,dSPECONV

###############################################################################################

@ray.remote
def nemesisfm_parallel(ix,runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,xnx):

    from NemesisPy import nemesisfm
    from copy import copy

    print('Calculating forward model '+str(ix)+'/'+str(Variables.NX))
    Variables1 = copy(Variables)
    Measurement1 = copy(Measurement)
    Atmosphere1 = copy(Atmosphere)
    Spectroscopy1 = copy(Spectroscopy)
    Scatter1 = copy(Scatter)
    Stellar1 = copy(Stellar)
    Surface1 = copy(Surface)
    CIA1 = copy(CIA)
    Layer1 = copy(Layer)
    Variables1.XN = xnx[0:Variables1.NX,ix]
    SPECMOD = nemesisfm(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,CIA1,Layer1)
    YN = np.resize(np.transpose(SPECMOD),[Measurement.NY])
    return YN


###############################################################################################

def jacobian_nemesis(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,MakePlot=False,NCores=1):

    """

        FUNCTION NAME : jacobian_nemesis()
        
        DESCRIPTION : 

            This function calculates the Jacobian matrix by calling nx+1 times nemesisfm(). 
            This routine is set up so that each forward model is calculated in parallel, 
            increasing the computational speed of the code
 
        INPUTS :
      
            runname :: Name of the NEMESIS run
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
        
            YN,KK = jacobian_nemesis(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
 
        MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    from NemesisPy import nemesisfm
    import ray
    from copy import copy

    #################################################################################
    # Making some calculations for storing all the arrays
    #################################################################################

    nproc = Variables.NX+1 #Number of times we need to run the forward model

    #Constructing state vector after perturbation of each of the elements and storing in matrix
    nxn = Variables.NX+1
    xnx = np.zeros([Variables.NX,nxn])
    for i in range(Variables.NX+1):
        if i==0:   #First element is the normal state vector
            xnx[0:Variables.NX,i] = Variables.XN[0:Variables.NX]
        else:      #Perturbation of each element
            xnx[0:Variables.NX,i] = Variables.XN[0:Variables.NX]
            xnx[i-1,i] = Variables.XN[i-1]*1.01
            if Variables.XN[i-1]==0.0:
                xnx[i-1,i] = 0.05

    #################################################################################
    # Calculating the first forward model and the analytical part of Jacobian
    #################################################################################

    #Variables.NUM[:] = 1

    ian1 = np.where(Variables.NUM==0)  #Gradients calculated using CIRSradg
    ian1 = ian1[0]

    iYN = 0
    KK = np.zeros([Measurement.NY,Variables.NX])

    if len(ian1)>0:

        print('Calculating analytical part of the Jacobian :: Calling nemesisfmg ')

        SPECMOD,dSPECMOD = nemesisfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

        YN = np.resize(np.transpose(SPECMOD),[Measurement.NY])
        KK = np.zeros([Measurement.NY,Variables.NX])
        for ix in range(Variables.NX):
            KK[:,ix] = np.resize(np.transpose(dSPECMOD[:,:,ix]),[Measurement.NY])

        iYN = 1 #Indicates that some of the gradients and the measurement vector have already been caculated


    #################################################################################
    # Calculating all the required forward models for numerical differentiation
    #################################################################################

    inum1 = np.where( (Variables.NUM==1) & (Variables.FIX==0) )
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
    YNtot = np.zeros([Measurement.NY,nfm])

    print('Calculating numerical part of the Jacobian :: running '+str(nfm)+' forward models ')
    if NCores>1:
        ray.init(num_cpus=NCores)
        YNtot_ids = []
        SpectroscopyP = ray.put(Spectroscopy)
        for ix in range(nfm):
            YNtot_ids.append(nemesisfm_parallel.remote(ixrun[ix],runname,Variables,Measurement,Atmosphere,SpectroscopyP,Scatter,Stellar,Surface,CIA,Layer,xnx))

        #Block until the results have finished and get the results.
        YNtot1 = ray.get(YNtot_ids)
        for ix in range(nfm):
            YNtot[0:Measurement.NY,ix] = YNtot1[ix]
        ray.shutdown()

    else:
    
        for ifm in range(nfm):
            print('Calculating forward model '+str(ifm)+'/'+str(nfm))
            Variables1 = copy(Variables)
            Variables1.XN = xnx[0:Variables1.NX,ixrun[ifm]]
            SPECMOD = nemesisfm(runname,Variables1,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
            YNtot[0:Measurement.NY,ifm] = np.resize(np.transpose(SPECMOD),[Measurement.NY])

    if iYN==0:
        YN = np.zeros(Measurement.NY)
        YN[:] = YNtot[0:Measurement.NY,0]

    #################################################################################
    # Calculating the Jacobian matrix
    #################################################################################

    for i in range(len(inum)):

        if iYN==0:
            ifm = i + 1
        else:
            ifm = i

        xn1 = Variables.XN[inum[i]] * 1.01
        if xn1==0.0:
            xn1=0.05
        if Variables.FIX[i] == 0:
                KK[:,inum[i]] = (YNtot[:,ifm]-YN)/(xn1-Variables.XN[inum[i]])


    #################################################################################
    # Making summary plot if required
    ################################################################################# 
    if MakePlot==True:

        import matplotlib as matplotlib

        #Plotting the measurement vector

        fig,ax1 = plt.subplots(1,1,figsize=(13,4))

        ix = 0
        for igeom in range(Measurement.NGEOM):
            ax1.plot(Measurement.VCONV[0:Measurement.NCONV[igeom],igeom],YN[ix:ix+Measurement.NCONV[igeom]])
            ix = ix + Measurement.NCONV[igeom]
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Transmission')
        ax1.grid()
        plt.tight_layout()

        fig, ax = plt.subplots(1,1,figsize=(10,3))
        ax.set_xlabel('Measurement vector y')
        ax.set_ylabel('State vector x')
        ax.imshow(np.transpose(KK),cmap='hot',aspect='auto',origin='lower')
        ax.grid()
        plt.tight_layout()

        for ix in range(Variables.NX):
            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            ax1.plot(range(Measurement.NY),KK[:,ix])
            ax1.grid()
            ax1.set_title('Jacobian matrix ('+str(ix)+'/'+str(Variables.NX)+')')
            plt.tight_layout()

        plt.show()

    return YN,KK

###############################################################################################

def CIRSradg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path):
    
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
    dTAUCON = np.zeros([Measurement.NWAVE,Atmosphere.NVMR+2+Scatter.NDUST,Layer.NLAY]) #(NWAVE,NLAY,NGAS+2+NDUST)
    dTAUSCA = np.zeros([Measurement.NWAVE,Atmosphere.NVMR+2+Scatter.NDUST,Layer.NLAY]) #(NWAVE,NLAY,NGAS+2+NDUST)

    #Calculating the continuum absorption by gaseous species
    #################################################################################################################

    #Computes a polynomial approximation to any known continuum spectra for a particular gas over a defined wavenumber region.

    #To be done

    #Calculating the vertical opacity by CIA
    #################################################################################################################

    print('CIRSradg :: Calculating CIA opacity')
    TAUCIA,dTAUCIA,IABSORB = CIA.calc_tau_cia(Measurement.ISPACE,Measurement.WAVE,Atmosphere,Layer) #(NWAVE,NLAY);(NWAVE,NLAY,7)

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
    TAUDUST1,TAUCLSCAT,dTAUDUST1,dTAUCLSCAT = Scatter.calc_tau_dust(Measurement.WAVE,Layer) #(NWAVE,NLAYER,NDUST)

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

        """
        k_gas = np.zeros((Measurement.NWAVE,Spectroscopy.NG,Layer.NLAY,Spectroscopy.NGAS))
        dkgasdT = np.zeros((Measurement.NWAVE,Spectroscopy.NG,Layer.NLAY,Spectroscopy.NGAS))
        k_gas[:,0,:,:] = k[:,:,:]
        dkgasdT[:,0,:,:] = dkdT[:,:,:]

        f_gas = np.zeros([Spectroscopy.NGAS,Layer.NLAY])
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
            dTAUGAS[:,:,IGAS[0],:] = dk_layer[:,:,:,i] * 1.0e-4 * 1.0e-20  #dTAU/dq (m2)

        dTAUGAS[:,:,Atmosphere.NVMR,:] = dk_layer[:,:,:,Spectroscopy.NGAS] #dTAU/dT
        """

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
        
                #for j1 in range(j):
                #    print(j1,j,dtolddq[0,0,Atmosphere.NVMR,j1])
                #    input ()

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

@jit(nopython=True)
def map2lay(dSPECIN,NWAVE,NVMR,NDUST,NLAY,NPATH,NLAYIN,LAYINC):
    
    """
        FUNCTION NAME : map2lay()
        
        DESCRIPTION : This function maps the analytical gradients along the path to the gradients in each layer
        
        INPUTS :
        
            dSPECIN(NWAVE,NVMR+2+NDUST,NLAYIN,NPATH) :: Rate of change of output spectrum with respect to layer
                                                         properties along the path
            NWAVE :: Number of spectral points
            NVMR :: Number of gases in reference atmosphere
            NDUST :: Number of aerosol populations in reference atmosphere
            NLAY :: Number of layers
            NPATH :: Number of atmospheric paths
            NLAYIN(NPATH) :: Number of layer in each of the paths
            LAYINC(NLAY,NPATH) :: Layers in each path
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            dSPECOUT(NWAVE,NVMR+2+NDUST,NLAY,NPATH) :: Rate of change of output spectrum with respect to the
                                                        atmospheric profile parameters
        
        CALLING SEQUENCE:
        
            dSPECOUT = map2lay(dSPECIN,NWAVE,NVMR,NDUST,NLAY,NPATH,NLAYIN,LAYINC)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    dSPECOUT = np.zeros((NWAVE,NVMR+2+NDUST,NLAY,NPATH))
    for ipath in range(NPATH):
        for iparam in range(NVMR+2+NDUST):
                for ilay in range(NLAYIN[ipath]):
                    if iparam<=NVMR-1: #Gas gradients
                        dSPECOUT[:,iparam,NLAY-LAYINC[ilay,ipath],ipath] = dSPECOUT[:,iparam,NLAY-LAYINC[ilay,ipath],ipath] + dSPECIN[:,iparam,ilay,ipath]
                    elif iparam==NVMR: #Temperature gradient
                        dSPECOUT[:,iparam,NLAY-LAYINC[ilay,ipath],ipath] = dSPECOUT[:,iparam,NLAY-LAYINC[ilay,ipath],ipath] + dSPECIN[:,iparam,ilay,ipath]
                    elif( (iparam>NVMR) & (iparam<=NVMR+NDUST) ): #Dust gradient
                        dSPECOUT[:,iparam,NLAY-LAYINC[ilay,ipath],ipath] = dSPECOUT[:,iparam,NLAY-LAYINC[ilay,ipath],ipath] + dSPECIN[:,iparam,ilay,ipath]
                    elif iparam==NVMR+NDUST+1: #ParaH gradient
                        dSPECOUT[:,iparam,NLAY-LAYINC[ilay,ipath],ipath] = 0.0  #Needs to be included

    return dSPECOUT

###############################################################################################

#@jit(nopython=True)
def map2pro(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NLAYIN,LAYINC,DTE,DAM,DCO,INCPAR=[-1]):
    
    """
        FUNCTION NAME : map2pro()
        
        DESCRIPTION : This function maps the analytical gradients defined with respect to the Layers
                      onto the input atmospheric levels defined in Atmosphere
        
        INPUTS :
        
            dSPECIN(NWAVE,NVMR+2+NDUST,NLAYIN,NPATH) :: Rate of change of output spectrum with respect to layer
                                                         properties along the path
            NWAVE :: Number of spectral points
            NVMR :: Number of gases in reference atmosphere
            NDUST :: Number of aerosol populations in reference atmosphere
            NPRO :: Number of altitude points in reference atmosphere
            NPATH :: Number of atmospheric paths
            NLAYIN(NPATH) :: Number of layer in each of the paths
            LAYINC(NLAY,NPATH) :: Layers in each path
            DTE(NLAY,NPRO) :: Matrix relating the temperature in each layer to the temperature in the profiles
            DAM(NLAY,NPRO) :: Matrix relating the gas amounts in each layer to the gas VMR in the profiles
            DCO(NLAY,NPRO) :: Matrix relating the dust amounts in each layer to the dust abundance in the profiles
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            dSPECOUT(NWAVE,NVMR+2+NDUST,NPRO,NPATH) :: Rate of change of output spectrum with respect to the
                                                        atmospheric profile parameters
        
        CALLING SEQUENCE:
        
            dSPECOUT = map2pro(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NLAYIN,LAYINC,DTE,DAM,DCO)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    DAMx = DAM[LAYINC,:] #NLAYIN,NPATH,NPRO
    DCOx = DCO[LAYINC,:]
    DTEx = DTE[LAYINC,:]

    dSPECOUT = np.zeros((NWAVE,NVMR+2+NDUST,NPRO,NPATH))

    if INCPAR[0]!=-1:
        NPARAM = len(INCPAR)
    else:
        NPARAM = NVMR+2+NDUST
        INCPAR = range(NPARAM)

    for ipath in range(NPATH):
        for iparam in range(NPARAM):

            if INCPAR[iparam]<=NVMR-1: #Gas gradients
                dSPECOUT1 = np.tensordot(dSPECIN[:,INCPAR[iparam],:,ipath], DAMx[:,ipath,:], axes=(1,0))
            elif INCPAR[iparam]<=NVMR: #Temperature gradients
                dSPECOUT1 = np.tensordot(dSPECIN[:,INCPAR[iparam],:,ipath], DTEx[:,ipath,:], axes=(1,0))
            elif( (INCPAR[iparam]>NVMR) & (INCPAR[iparam]<=NVMR+NDUST) ): #Dust gradient
                dSPECOUT1 = np.tensordot(dSPECIN[:,INCPAR[iparam],:,ipath], DCOx[:,ipath,:], axes=(1,0))
            elif INCPAR[iparam]==NVMR+NDUST+1: #ParaH gradient
                dSPECOUT[:,INCPAR[iparam],:,ipath] = 0.0  #Needs to be included

            dSPECOUT[:,INCPAR[iparam],:,ipath] = dSPECOUT1[:,:]

    return dSPECOUT


###############################################################################################

#@jit(nopython=True)
def map2xvec(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NX,xmap):
    
    """
        FUNCTION NAME : map2xvec()
        
        DESCRIPTION : This function maps the analytical gradients defined with respect to the Layers
                      onto the input atmospheric levels defined in Atmosphere
        
        INPUTS :
        
            dSPECIN(NWAVE,NVMR+2+NDUST,NPRO,NPATH) :: Rate of change of output spectrum with respect to profiles
            NWAVE :: Number of spectral points
            NVMR :: Number of gases in reference atmosphere
            NDUST :: Number of aerosol populations in reference atmosphere
            NPRO :: Number of altitude points in reference atmosphere
            NPATH :: Number of atmospheric paths
            NX :: Number of elements in state vector
            XMAP(NX,NVMR+2+NDUST,NPRO) :: Matrix relating the gradients in the profiles to the elemenents in state vector

        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            dSPECOUT(NWAVE,NPATH,NX) :: Rate of change of output spectrum with respect to the elements in the state vector
        
        CALLING SEQUENCE:
        
            dSPECOUT = map2xvec(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NX,xmap)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    #Mapping the gradients to the elements in the state vector
    dSPECOUT = np.tensordot(dSPECIN, xmap, axes=([1,2],[1,2])) #NWAVE,NPATH,NX

    return dSPECOUT


###############################################################################################

def CIRSrad(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path):
    
    """
        FUNCTION NAME : CIRSrad()
        
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
        
            SPECOUT = CIRSrad(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    import matplotlib as matplotlib
    from scipy import interpolate
    from NemesisPy import k_overlap, k_overlapg, planck
    from copy import copy

    #Initialise some arrays
    ###################################

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

    TAUCIA,dTAUCIA,IABSORB = CIA.calc_tau_cia(Measurement.ISPACE,Measurement.WAVE,Atmosphere,Layer) #(NWAVE,NLAY);(NWAVE,NLAY,7)

    #Calculating the vertical opacity by Rayleigh scattering
    #################################################################################################################

    if Scatter.IRAY==0:
        TAURAY = np.zeros([Measurement.NWAVE,Layer.NLAY])
    elif Scatter.IRAY==1:
        TAURAY,dTAURAY = Scatter.calc_tau_rayleighj(Measurement.ISPACE,Measurement.WAVE,Layer) #(NWAVE,NLAY)
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

    TAUDUST1,TAUCLSCAT,dTAUDUST1,dTAUCLSCAT = Scatter.calc_tau_dust(Measurement.WAVE,Layer) #(NWAVE,NLAYER,NDUST)

    #Adding the opacity by the different dust populations
    TAUDUST = np.sum(TAUDUST1,2)  #(NWAVE,NLAYER)
    TAUSCAT = np.sum(TAUCLSCAT,2)  #(NWAVE,NLAYER)

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
        k_gas = Spectroscopy.calc_k(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE) # (NWAVE,NG,NLAY,NGAS)

        f_gas = np.zeros([Spectroscopy.NGAS,Layer.NLAY])
        utotl = np.zeros(Layer.NLAY)
        for i in range(Spectroscopy.NGAS):
            IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
            IGAS = IGAS[0]
    
            f_gas[i,:] = Layer.PP[:,IGAS].T / Layer.PRESS                     #VMR of each radiatively active gas
            utotl[:] = utotl[:] + Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #Vertical column density of the radiatively active gases

        #Combining the k-distributions of the different gases in each layer
        k_layer = k_overlap(Measurement.NWAVE,Spectroscopy.NG,Spectroscopy.DELG,Spectroscopy.NGAS,Layer.NLAY,k_gas,f_gas)  #(NWAVE,NG,NLAY)

        #Calculating the opacity of each layer
        TAUGAS = k_layer * utotl   #(NWAVE,NG,NLAY)

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


    #Now integrate over g-ordinates
    SPECOUT = np.tensordot(SPECOUT, Spectroscopy.DELG, axes=([1],[0])) #NWAVE,NPATH

    return SPECOUT
