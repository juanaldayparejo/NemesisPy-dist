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

    from NemesisPy.Models import subprofretg
    from NemesisPy.Models import subspecret
    from NemesisPy.Layer import AtmCalc_0
    from NemesisPy.Layer import Path_0
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
    flagh2p = False

    xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)


    #Based on the new reference atmosphere, we split the atmosphere into layers
    #In solar occultation LAYANG = 90.0
    LAYANG = 90.0

    BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH\
        = Layer.integrate(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)

    #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
    NCALC = Layer.NLAY    #Number of calculations (geometries) to be performed
    AtmCalc_List = []
    for ICALC in range(NCALC):
        iAtmCalc = AtmCalc_0(Layer,LIMB=True,BOTLAY=ICALC,ANGLE=90.0,IPZEN=0)
        AtmCalc_List.append(iAtmCalc)
    

    #We initialise the total Path class, indicating that the calculations can be combined
    Path = Path_0(AtmCalc_List,COMBINE=True)

    #Calling CIRSrad to calculate the spectra
    SPECOUT = CIRSrad(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path)


    #Interpolating the spectra to the correct altitudes defined in Measurement
    SPECMOD = np.zeros([Measurement.NWAVE,Measurement.NGEOM])
    for i in range(Measurement.NGEOM):

        #Find altitudes above and below the actual tangent height
        base0,ibase = find_nearest(Layer.BASEH/1.0e3,Measurement.TANHE[i])
        if base0<=Measurement.TANHE[i]:
            ibasel = ibase
            ibaseh = ibase + 1
        else:
            ibasel = ibase - 1
            ibaseh = ibase

        if ibaseh>Layer.NLAY-1:
            SPECMOD[:,i] = SPECOUT[:,ibasel]
        else:
            fhl = (Measurement.TANHE[i]*1.0e3-Layer.BASEH[ibasel])/(Layer.BASEH[ibaseh]-Layer.BASEH[ibasel])
            fhh = (Layer.BASEH[ibaseh]-Measurement.TANHE[i]*1.0e3)/(Layer.BASEH[ibaseh]-Layer.BASEH[ibasel])

            SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)


    #Applying any changes to the spectra required by the state vector
    SPECMOD = subspecret(Measurement,Variables,SPECMOD)

    #Convolving the spectrum with the instrument line shape
    SPECONV = Measurement.lblconv(SPECMOD,IGEOM='All')

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

    #Defining the parallel version of nemesisSOfm
def nemesisSOfm_parallel(ix,runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,Layer,xnx):

    Variables1 = copy(Variables)
    Measurement1 = copy(Measurement)
    Atmosphere1 = copy(Atmosphere)
    Spectroscopy1 = copy(Spectroscopy)
    Scatter1 = copy(Scatter)
    Stellar1 = copy(Stellar)
    Surface1 = copy(Surface)
    Layer1 = copy(Layer)
    Variables1.XN = xnx[0:Variables1.NX,ix]
    SPECMOD = nemesisSOfm(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,Layer1)
    #YN = np.resize(np.transpose(SPECMOD),[Measurement.NY])
    return SPECMOD  

###############################################################################################

def jacobian_nemesisSO(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,MakePlot=False):

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
        
        OUTPUTS :

            YN(NY) :: New measurement vector
            KK(NY,NX) :: Jacobian matrix

        CALLING SEQUENCE:
        
            YN,KK = jacobian_nemesisSO(Variables,Measurement,Atmosphere,Scatter,Stellar,Surface,CIA,Layer)
 
        MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    from NemesisPy import nemesisSOfm
    from NemesisPy import nemesisSOfm_parallel
    import ray
    from copy import copy

    #import multiprocessing
    #from functools import partial
    #from shutil import copyfile

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
    # Calculating all the required forward models
    #################################################################################

    """
    #Checking how many forward models need to be computed
    ifix1 = np.zeros([Variables.NX+1],dtype='int')
    ifix1[1:Variables.NX+1] = Variables.FIX[0:Variables.NX]
    ifw1 = np.where((ifix1==0))
    ifw = ifw1[0]
    nfw = len(ifw)

    #Calculating the array indices that need to be calculated in each set of runs
    nmaxproc = 1
    nrun = int(nfw/nmaxproc)     #Number of set of runs 
    nrunf = float(nfw)/float(nmaxproc)
    if nrun==0:
        nrun = 1
    if (nrunf - float(nrun)) > 0.0:
        nrun = nrun + 1

    nprocrun = np.zeros([nrun],dtype='int')  #Number of parallel forward models per run
    ix = 0
    for i in range(nrun):
        nprocrun[i] = nmaxproc
        ix = ix + nprocrun[i]
        if ix > nfw:
            ix = ix - nprocrun[i]
            nprocrun[i] = nfw - ix

    #Running the forward models
    ix = 0
    YNtot = np.zeros([Measurement.NY,nxn]) #Array in which to store the forward models
    for i in range(nrun):
        iproc = np.zeros([nprocrun[i]],dtype='int')
        iproc[:] = ifw[ix:ix+nprocrun[i]]
        pool = multiprocessing.Pool(processes=nprocrun[i])
        fmproc=partial(nemesisSOfm_parallel,runname=runname,Variables=Variables,Measurement=Measurement,Atmosphere=Atmosphere,\
                                            Spectroscopy=Spectroscopy,Scatter=Scatter,Stellar=Stellar,Surface=Surface,Layer=Layer,xnx=xnx)
        result_list = pool.map(fmproc,iproc)
        sys.exit()
        for j in range(nprocrun[i]):
            specret2_1 = result_list[j]
            specret2 = np.zeros([Measurement.NCONV[0],Measurement.NGEOM])
            specret2[:,:] = specret2_1[:,0:Measurement.NGEOM]
            yn1 = np.resize(np.transpose(specret2),[Measurement.NY])   #Modelled measurement vector
            YNtot[:,iproc[j]] = yn1[:]

        ix = ix + nprocrun[i]
    """

    """
    ray.init(num_cpus=4)
    @ray.remote
    def nemesisSOfm_parallel(ix,runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,xnx):
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
    """

    #Calling the forward model NXN times to calculate the measurement vector for each case
    YNtot = np.zeros([Measurement.NY,nxn])
    #YNtot = []
    for ix in range(nxn):
        print('Calculating forward model '+str(ix)+'/'+str(Variables.NX))

        #Serial approach
        Variables1 = copy(Variables)
        Variables1.XN = xnx[0:Variables1.NX,ix]
        SPECMOD = nemesisSOfm(runname,Variables1,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
        YNtot[0:Measurement.NY,ix] = np.resize(np.transpose(SPECMOD),[Measurement.NY])
        
        #Parallel approach using Ray
        #YN = nemesisSOfm_parallel.remote(ix,runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,xnx)
        #YNtot.append(YN)

    #Block until the results have finished and get the results.
    #YNtot = ray.get(YNtot)


    #################################################################################
    # Calculating the Jacobian matrix
    #################################################################################

    KK = np.zeros([Measurement.NY,Variables.NX])
    for i in range(Variables.NX):
        xn1 = Variables.XN[i] * 1.01
        if xn1==0.0:
            xn1=0.05
        if Variables.FIX[i] == 0:
                KK[:,i] = (YNtot[:,i+1]-YNtot[:,0])/(xn1-Variables.XN[i])

    YN = np.zeros(Measurement.NY)
    YN[:] = YNtot[:,0]

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

def nemesisfm(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,Layer):
    
    """
        FUNCTION NAME : nemesisSOfm()
        
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
        
            nemesisfm(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,Layer)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    from NemesisPy.Models import subprofretg
    from NemesisPy.Layer import AtmCalc_0
    from NemesisPy.Layer import Path_0
    from NemesisPy.Radtrans import lblconv
    from NemesisPy import find_nearest
    from scipy import interpolate

    #First we change the reference atmosphere taking into account the parameterisations in the state vector
    Variables1 = copy(Variables)
    Measurement1 = copy(Measurement)
    Atmosphere1 = copy(Atmosphere)
    Scatter1 = copy(Scatter)
    Stellar1 = copy(Stellar)
    Surface1 = copy(Surface)
    Layer1 = copy(Layer)
    flagh2p = False

    xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)

    #Calling gsetrad to set the required flags in Path and Scatter to perform the correct calculations
    



    #Based on the new reference atmosphere, we split the atmosphere into layers
    BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH\
        = Layer.integrate(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)


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
    from NemesisPy import k_overlap, k_overlap_v2, planck
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


    #Calculating the vertical opacity by CIA
    #################################################################################################################

    TAUCIA = CIA.calc_tau_cia(Measurement.ISPACE,Measurement.WAVE,Atmosphere,Layer) #(NWAVE,NLAY)

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

    TAUDUST1,TAUCLSCAT = Scatter.calc_tau_dust(Measurement.WAVE,Layer) #(NWAVE,NLAYER,NDUST)

    #Adding the opacity by the different dust populations
    TAUDUST = np.sum(TAUDUST1,2)  #(NWAVE,NLAYER)
    TAUSCAT = np.sum(TAUCLSCAT,2)  #(NWAVE,NLAYER)

    #Calculating the gaseous line opacity in each layer
    ########################################################################################################

    if Spectroscopy.ILBL==2:  #LBL-table

        TAUGAS = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Layer.NLAY,Spectroscopy.NGAS])  #Vertical opacity of each gas in each layer

        #Calculating the cross sections for each gas in each layer
        k = Spectroscopy.calc_klbl(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())

        for i in range(Spectroscopy.NGAS):
            IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
            IGAS = IGAS[0]
    
            #Calculating vertical column density in each layer
            VLOSDENS = Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #cm-2

            #Calculating vertical opacity for each gas in each layer
            TAUGAS[:,0,:,i] = k[:,:,i] * VLOSDENS

        #Combining the gaseous opacity in each layer
        TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)

    elif Spectroscopy.ILBL==0:    #K-table

        #Calculating the k-coefficients for each gas in each layer
        k_gas = Spectroscopy.calc_k(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE)

        f_gas = np.zeros([Spectroscopy.NGAS,Layer.NLAY])
        utotl = np.zeros(Layer.NLAY)
        for i in range(Spectroscopy.NGAS):
            IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
            IGAS = IGAS[0]
    
            f_gas[i,:] = Layer.PP[:,IGAS].T / Layer.PRESS                     #VMR of each radiatively active gas
            utotl[:] = utotl[:] + Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #Vertical column density of the radiatively active gases

        #Combining the k-distributions of the different gases in each layer
        #k_layer = k_overlap(NWAVECALC,Spec.NG,Spec.DELG,Spec.NGAS,Layer.NLAY,k_gas,f_gas)  #(NWAVE,NG,NLAY)
        k_layer = k_overlap_v2(Measurement.NWAVE,Spectroscopy.NG,Spectroscopy.G_ORD,Spectroscopy.DELG,Spectroscopy.NGAS,Layer.NLAY,k_gas,f_gas)  #(NWAVE,NG,NLAY)

        #Calculating the opacity of each layer
        TAUGAS = k_layer * utotl   #(NWAVE,NG,NLAY)

    else:
        sys.exit('error in CIRSrad :: ILBL must be either 0 or 2')

    #Calculating the line-of-sight opacities
    #################################################################################################################

    #Calculating the opacity of each layer along the line-of-sight
    TAUGAS_LAYINC = TAUGAS[:,:,Path.LAYINC[:,:]] * Path.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)
    TAUDUST_LAYINC = TAUDUST[:,Path.LAYINC[:,:]] * Path.SCALE[:,:]  #(NWAVE,NLAYIN,NPATH)
    TAUCIA_LAYINC = TAUCIA[:,Path.LAYINC[:,:]] * Path.SCALE[:,:]  #(NWAVE,NLAYIN,NPATH)

    #Combining the different sources of opacity
    TAUTOT_LAYINC = np.zeros(TAUGAS_LAYINC.shape)
    for ig in range(Spectroscopy.NG):
        TAUTOT_LAYINC[:,ig,:,:] = TAUGAS_LAYINC[:,ig,:,:] + TAUCIA_LAYINC[:,:,:] + TAUDUST_LAYINC[:,:,:]


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

        if Measurement.IFORM==4:  #If IFORM=4 we should multiply the transmission by solar flux
            Stellar.calc_solar_flux()
            #Interpolating to the calculation wavelengths
            f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLFLUX)
            solflux = f(Measurement.WAVE)
            for ipath in range(npath):
                SPECOUT[:,:,ipat] = SPECOUT[:,:,ipat] * solflux

    elif IMODM==1:

        #Calculating the total opacity over the path
        TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH) 

        #Absorption spectrum (useful for small transmissions)
        SPECOUT = 1.0 - np.exp(-(TAUTOT_PATH)) #(NWAVE,NG,NPATH)


    elif IMODM==3: #Thermal emission from planet

        SPECOUT = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Path.NPATH])

        for ipath in range(Path.NPATH):

            taud = np.zeros([Measurement.NWAVE,Spectroscopy.NG])
            trold = np.ones([Measurement.NWAVE,Spectroscopy.NG])
            specg = np.zeros([Measurement.NWAVE,Spectroscopy.NG])

            for j in range(Path.NLAYIN[ipath]):

                taud[:,:] = taud[:,:] + TAUTOT_LAYINC[:,:,j,ipath]
                tr = np.exp(-taud)
        
                bb = planck(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[j,ipath])
                for ig in range(Spectroscopy.NG):
                    specg[:,ig] = specg[:,ig] + (trold[:,ig]-tr[:,ig])*bb[:]

                trold = copy(tr)

            SPECOUT[:,:,ipath] = specg[:,:]


    #Now integrate over g-ordinates
    SPECOUT = np.tensordot(SPECOUT, Spectroscopy.DELG, axes=([1],[0]))

    return SPECOUT