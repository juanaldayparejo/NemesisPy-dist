# NAME:
#       Models.py (NemesisPy)
#
# DESCRIPTION:
#
#	This library contains functions to change the Nemesis atmospheric profiles or the rest
#   of the model parameters based on the different parameterisations      
#
# CATEGORY:
#
#	NEMESIS
# 
# MODIFICATION HISTORY: Juan Alday 15/07/2021


from NemesisPy.Profile import *
from NemesisPy.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys

###############################################################################################

def modelm1(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : modelm1()
        
        DESCRIPTION :
        
            Function defining the model parameterisation -1 in NEMESIS.
            In this model, the aerosol profiles is modelled as a continuous profile in units
            of particles per cm3. Note that typical units of aerosol profiles in NEMESIS
            are in particles per gram of atmosphere
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xprof(npro) :: Atmospheric aerosol profile in particles/cm3
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = modelm1(atm,ipar,xprof)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npro = len(xprof)
    if npro!=atm.NP:
        sys.exit('error in model -1 :: Number of levels in atmosphere does not match and profile')

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([npro,npar,npro])

    if ipar<atm.NVMR:  #Gas VMR
        sys.exit('error :: Model -1 is just compatible with aerosol populations (Gas VMR given)')
    elif ipar==atm.NVMR: #Temperature
        sys.exit('error :: Model -1 is just compatible with aerosol populations (Temperature given)')
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        x1 = np.exp(xprof)
        if jtmp<atm.NDUST:
            rho = atm.calc_rho()  #kg/m3
            rho = rho / 1.0e3 #g/cm3
            atm.DUST[:,jtmp] = x1 / rho
        elif jtmp==atm.NDUST:
            sys.exit('error :: Model -1 is just compatible with aerosol populations')
        elif jtmp==atm.NDUST+1:
            sys.exit('error :: Model -1 is just compatible with aerosol populations')
    
    for j in range(npro):
        xmap[0:npro,ipar,j] = x1[:] / rho[:]
        

    if MakePlot==True:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(7,5))

        for i in range(atm.NDUST):
            ax1.semilogx(atm.DUST[:,i]*rho,atm.H/1000.)
            ax2.semilogx(atm.DUST[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax1.set_xlabel('Aerosol density (particles per cm$^{-3}$)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Aerosol density (particles per gram of atm)')
        ax2.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model0(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : model0()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 0 in NEMESIS.
            In this model, the atmospheric parameters are modelled as continuous profiles
            in which each element of the state vector corresponds to the atmospheric profile 
            at each altitude level
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xprof(npro) :: Atmospheric profile
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model0(atm,ipar,xprof)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npro = len(xprof)
    if npro!=atm.NP:
        sys.exit('error in model 0 :: Number of levels in atmosphere does not match and profile')

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([npro,npar,npro])

    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        x1 = np.exp(xprof)
        vmr = np.zeros([atm.NP,atm.NVMR])
        vmr[:,:] = atm.VMR
        vmr[:,jvmr] = x1
        atm.edit_VMR(vmr)
        for j in range(npro):
            xmap[j,ipar,j] = x1[j]
    elif ipar==atm.NVMR: #Temperature
        x1 = xprof
        atm.edit_T(x1)
        for j in range(npro):
            xmap[j,ipar,j] = 1.
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        x1 = np.exp(xprof)
        if jtmp<atm.NDUST: #Dust in m-3
            dust = np.zeros([atm.NP,atm.NDUST])
            dust[:,:] = atm.DUST
            dust[:,jtmp] = x1
            atm.edit_DUST(dust)
            for j in range(npro):
                xmap[j,ipar,j] = x1[j]
        elif jtmp==atm.NDUST:
            atm.PARAH2 = x1
            for j in range(npro):
                xmap[j,ipar,j] = x1[j]
        elif jtmp==atm.NDUST+1:
            atm.FRAC = x1
            for j in range(npro):
                xmap[j,ipar,j] = x1[j]

    if MakePlot==True:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

        ax1.semilogx(atm.P/101325.,atm.H/1000.)
        ax2.plot(atm.T,atm.H/1000.)
        for i in range(atm.NVMR):
            ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Altitude (km)')
        ax3.set_xlabel('Volume mixing ratio')
        ax3.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model2(atm,ipar,scf,MakePlot=False):
    
    """
        FUNCTION NAME : model2()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 2 in NEMESIS.
            In this model, the atmospheric parameters are scaled using a single factor with 
            respect to the vertical profiles in the reference atmosphere
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            scf :: Scaling factor
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model2(atm,ipar,scf)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([1,npar,atm.NP])

    x1 = np.zeros(atm.NP)
    xref = np.zeros(atm.NP)
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        xref[:] = atm.VMR[:,jvmr]
        x1[:] = atm.VMR[:,jvmr] * scf
        atm.VMR[:,jvmr] =  x1
    elif ipar==atm.NVMR: #Temperature
        xref[:] = atm.T[:]
        x1[:] = atm.T[:] * scf
        atm.T[:] = x1 
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST:
            xref[:] = atm.DUST[:,jtmp]
            x1[:] = atm.DUST[:,jtmp] * scf
            atm.DUST[:,jtmp] = x1
        elif jtmp==atm.NDUST:
            xref[:] = atm.PARAH2
            x1[:] = atm.PARAH2 * scf
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            xref[:] = atm.FRAC
            x1[:] = atm.FRAC * scf
            atm.FRAC = x1

    xmap[0,ipar,:] = xref[:]
    
    if MakePlot==True:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

        ax1.semilogx(atm.P/101325.,atm.H/1000.)
        ax2.plot(atm.T,atm.H/1000.)
        for i in range(atm.NVMR):
            ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Altitude (km)')
        ax3.set_xlabel('Volume mixing ratio')
        ax3.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model3(atm,ipar,scf,MakePlot=False):
    
    """
        FUNCTION NAME : model3()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 2 in NEMESIS.
            In this model, the atmospheric parameters are scaled using a single factor 
            in logscale with respect to the vertical profiles in the reference atmosphere
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            scf :: Log scaling factor
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model2(atm,ipar,scf)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([1,npar,atm.NP])

    x1 = np.zeros(atm.NP)
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        x1[:] = atm.VMR[:,jvmr] * np.exp(scf)
        atm.VMR[:,jvmr] =  x1 
    elif ipar==atm.NVMR: #Temperature
        x1[:] = atm.T[:] * np.exp(scf)
        atm.T[:] = x1 
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST:
            x1[:] = atm.DUST[:,jtmp] * np.exp(scf)
            atm.DUST[:,jtmp] = x1
        elif jtmp==atm.NDUST:
            x1[:] = atm.PARAH2 * np.exp(scf)
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            x1[:] = atm.FRAC * np.exp(scf)
            atm.FRAC = x1

    xmap[0,ipar,:] = x1[:]
    
    if MakePlot==True:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

        ax1.semilogx(atm.P/101325.,atm.H/1000.)
        ax2.plot(atm.T,atm.H/1000.)
        for i in range(atm.NVMR):
            ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Altitude (km)')
        ax3.set_xlabel('Volume mixing ratio')
        ax3.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model9(atm,ipar,href,fsh,tau,MakePlot=False):
    
    """
        FUNCTION NAME : model9()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 9 in NEMESIS.
            In this model, the profile (cloud profile) is represented by a value
            at a certain height, plus a fractional scale height. Below the reference height 
            the profile is set to zero, while above it the profile decays exponentially with
            altitude given by the fractional scale height. In addition, this model scales
            the profile to give the requested integrated cloud optical depth.
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            href :: Base height of cloud profile (km)

            fsh :: Fractional scale height (km)

            tau :: Total integrated column density of the cloud (m-2)
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model9(atm,ipar,href,fsh,tau)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    from scipy.integrate import simpson

    #Checking that profile is for aerosols
    if(ipar<=atm.NVMR):
        sys.exit('error in model 9 :: This model is defined for aerosol profiles only')

    if(ipar>atm.NVMR+atm.NDUST):
        sys.exit('error in model 9 :: This model is defined for aerosol profiles only')
    

    #Calculating the actual atmospheric scale height in each level
    R = const["R"]
    scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

    #This gradient is calcualted numerically (in this function) as it is too hard otherwise
    xprof = np.zeros(atm.NP)
    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([3,npar,atm.NP])
    for itest in range(4):

        xdeep = tau
        xfsh = fsh
        hknee = href

        if itest==0:
            dummy = 1
        elif itest==1: #For calculating the gradient wrt tau
            dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
            if dx==0.0:
                dx = 0.1
            xdeep = np.exp( np.log(tau) + dx )
        elif itest==2: #For calculating the gradient wrt fsh
            dx = 0.05 * np.log(fsh)  #In the state vector this variable is passed in log-scale
            if dx==0.0:
                dx = 0.1
            xfsh = np.exp( np.log(fsh) + dx )
        elif itest==3: #For calculating the gradient wrt href
            dx = 0.05 * href
            if dx==0.0:
                dx = 0.1
            hknee = href + dx

        #Initialising some arrays
        ND = np.zeros(atm.NP)   #Dust density (m-3)

        #Calculating the density in each level
        jfsh = -1
        if atm.H[0]/1.0e3>=hknee:
            jfsh = 1
            ND[0] = 1.

        for jx in range(atm.NP-1):
            j = jx + 1
            delh = atm.H[j] - atm.H[j-1]
            xfac = scale[j] * xfsh

            if atm.H[j]/1.0e3>=hknee:
                
                if jfsh<0:
                    ND[j]=1.0
                    jfsh = 1
                else:
                    ND[j]=ND[j-1]*np.exp(-delh/xfac)


        for j in range(atm.NP):
            if(atm.H[j]/1.0e3<hknee):
                if(atm.H[j+1]/1.0e3>=hknee):
                    ND[j] = ND[j] * (1.0 - (hknee*1.0e3-atm.H[j])/(atm.H[j+1]-atm.H[j]))
                else:
                    ND[j] = 0.0

        #Calculating column density (m-2) by integrating the number density (m-3) over column (m)
        #Note that when doing the layering, the total column density in the atmosphere might not be
        #exactly the same as in xdeep due to misalignments at the boundaries of the cloud
        totcol = simpson(ND,x=atm.H)
        ND = ND / totcol * xdeep

        if itest==0:
            xprof[:] = ND[:]
        else:
            xmap[itest-1,ipar,:] = (ND[:]-xprof[:])/dx

    icont = ipar - (atm.NVMR+1)
    atm.DUST[0:atm.NP,icont] = xprof

    return atm,xmap


###############################################################################################

def model32(atm,ipar,href,fsh,tau,MakePlot=False):
    
    """
        FUNCTION NAME : model9()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 32 in NEMESIS.
            In this model, the profile (cloud profile) is represented by a value
            at a certain pressure level, plus a fractional scale height. 
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            href :: Base height of cloud profile (km)

            fsh :: Fractional scale height (km)

            tau :: Total integrated column density of the cloud (m-2)
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model9(atm,ipar,href,fsh,tau)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    from scipy.integrate import simpson

    #Checking that profile is for aerosols
    if(ipar<=atm.NVMR):
        sys.exit('error in model 9 :: This model is defined for aerosol profiles only')

    if(ipar>atm.NVMR+atm.NDUST):
        sys.exit('error in model 9 :: This model is defined for aerosol profiles only')
    

    #Calculating the actual atmospheric scale height in each level
    R = const["R"]
    scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

    #This gradient is calcualted numerically (in this function) as it is too hard otherwise
    xprof = np.zeros(atm.NP)
    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros([3,npar,atm.NP])
    for itest in range(4):

        xdeep = tau
        xfsh = fsh
        hknee = href

        if itest==0:
            dummy = 1
        elif itest==1: #For calculating the gradient wrt tau
            dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
            if dx==0.0:
                dx = 0.1
            xdeep = np.exp( np.log(tau) + dx )
        elif itest==2: #For calculating the gradient wrt fsh
            dx = 0.05 * np.log(fsh)  #In the state vector this variable is passed in log-scale
            if dx==0.0:
                dx = 0.1
            xfsh = np.exp( np.log(fsh) + dx )
        elif itest==3: #For calculating the gradient wrt href
            dx = 0.05 * href
            if dx==0.0:
                dx = 0.1
            hknee = href + dx

        #Initialising some arrays
        ND = np.zeros(atm.NP)   #Dust density (m-3)

        #Calculating the density in each level
        jfsh = -1
        if atm.H[0]/1.0e3>=hknee:
            jfsh = 1
            ND[0] = 1.

        for jx in range(atm.NP-1):
            j = jx + 1
            delh = atm.H[j] - atm.H[j-1]
            xfac = scale[j] * xfsh

            if atm.H[j]/1.0e3>=hknee:
                
                if jfsh<0:
                    ND[j]=1.0
                    jfsh = 1
                else:
                    ND[j]=ND[j-1]*np.exp(-delh/xfac)


        for j in range(atm.NP):
            if(atm.H[j]/1.0e3<hknee):
                if(atm.H[j+1]/1.0e3>=hknee):
                    ND[j] = ND[j] * (1.0 - (hknee*1.0e3-atm.H[j])/(atm.H[j+1]-atm.H[j]))
                else:
                    ND[j] = 0.0

        #Calculating column density (m-2) by integrating the number density (m-3) over column (m)
        #Note that when doing the layering, the total column density in the atmosphere might not be
        #exactly the same as in xdeep due to misalignments at the boundaries of the cloud
        totcol = simpson(ND,x=atm.H)
        ND = ND / totcol * xdeep

        if itest==0:
            xprof[:] = ND[:]
        else:
            xmap[itest-1,ipar,:] = (ND[:]-xprof[:])/dx

    icont = ipar - (atm.NVMR+1)
    atm.DUST[0:atm.NP,icont] = xprof

    return atm,xmap


###############################################################################################

def model47(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : model47()
        
        DESCRIPTION :
        
            Profile is represented by a Gaussian with a specified optical thickness centred
            at a variable pressure level plus a variable FWHM (log press) in height.
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xdeep :: 
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model50(atm,ipar,xprof)
        
        MODIFICATION HISTORY : Juan Alday (08/06/2022)
        
    """


###############################################################################################

def model49(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : model0()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 49 in NEMESIS.
            In this model, the atmospheric parameters are modelled as continuous profiles
             in linear space. This parameterisation allows the retrieval of negative VMRs.
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xprof(npro) :: Scaling factor at each altitude level
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model50(atm,ipar,xprof)
        
        MODIFICATION HISTORY : Juan Alday (08/06/2022)
        
    """

    npro = len(xprof)
    if npro!=atm.NP:
        sys.exit('error in model 49 :: Number of levels in atmosphere and scaling factor profile does not match')

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros((npro,npar,npro))

    x1 = np.zeros(atm.NP)
    xref = np.zeros(atm.NP)
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        xref[:] = atm.VMR[:,jvmr]
        x1[:] = xprof
        vmr = np.zeros((atm.NP,atm.NVMR))
        vmr[:,:] = atm.VMR
        vmr[:,jvmr] = x1[:]
        atm.edit_VMR(vmr)
    elif ipar==atm.NVMR: #Temperature
        xref = atm.T
        x1 = xprof
        atm.edit_T(x1)
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST: #Dust in m-3
            xref[:] = atm.DUST[:,jtmp]
            x1[:] = xprof
            dust = np.zeros((atm.NP,atm.NDUST))
            dust[:,:] = atm.DUST
            dust[:,jtmp] = x1
            atm.edit_DUST(dust)
        elif jtmp==atm.NDUST:
            xref[:] = atm.PARAH2
            x1[:] = xprof
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            xref[:] = atm.FRAC
            x1[:] = xprof
            atm.FRAC = x1

    for j in range(npro):
        xmap[j,ipar,j] = 1.

    return atm,xmap


###############################################################################################

def model50(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : model0()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 50 in NEMESIS.
            In this model, the atmospheric parameters are modelled as continuous profiles
            multiplied by a scaling factor in linear space. Each element of the state vector
            corresponds to this scaling factor at each altitude level. This parameterisation
            allows the retrieval of negative VMRs.
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xprof(npro) :: Scaling factor at each altitude level
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model50(atm,ipar,xprof)
        
        MODIFICATION HISTORY : Juan Alday (08/06/2022)
        
    """

    npro = len(xprof)
    if npro!=atm.NP:
        sys.exit('error in model 50 :: Number of levels in atmosphere and scaling factor profile does not match')

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros((npro,npar,npro))

    x1 = np.zeros(atm.NP)
    xref = np.zeros(atm.NP)
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        xref[:] = atm.VMR[:,jvmr]
        x1[:] = atm.VMR[:,jvmr] * xprof
        vmr = np.zeros((atm.NP,atm.NVMR))
        vmr[:,:] = atm.VMR
        vmr[:,jvmr] = x1[:]
        atm.edit_VMR(vmr)
    elif ipar==atm.NVMR: #Temperature
        xref = atm.T
        x1 = atm.T * xprof
        atm.edit_T(x1)
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST: #Dust in m-3
            xref[:] = atm.DUST[:,jtmp]
            x1[:] = atm.DUST[:,jtmp] * xprof
            dust = np.zeros((atm.NP,atm.NDUST))
            dust[:,:] = atm.DUST
            dust[:,jtmp] = x1
            atm.edit_DUST(dust)
        elif jtmp==atm.NDUST:
            xref[:] = atm.PARAH2
            x1[:] = atm.PARAH2 * xprof
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            xref[:] = atm.FRAC
            x1[:] = atm.FRAC * xprof
            atm.FRAC = x1

    for j in range(npro):
        xmap[j,ipar,j] = xref[j]

    return atm,xmap



###############################################################################################

def model228(Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3,MakePlot=False):
    
    """
        FUNCTION NAME : model228()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 228 in NEMESIS.

            In this model, the wavelength calibration of a given spectrum is performed, as well as the fit
            of a double Gaussian ILS suitable for ACS MIR solar occultation observations
            
            The wavelength calibration is performed such that the first wavelength or wavenumber is given by V0. 
            Then the rest of the wavelengths of the next data points are calculated by calculating the wavelength
            step between data points given by dV = C0 + C1*data_number + C2*data_number, where data_number 
            is an array going from 0 to NCONV-1.

            The ILS is fit using the approach of Alday et al. (2019, A&A). In this approach, the parameters to fit
            the ILS are the Offset of the second gaussian with respect to the first one (P0), the FWHM of the main 
            gaussian (P1), Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber (P2)
            , Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (P3), and
            a linear variation of the relative amplitude.
        
        INPUTS :
        
            Measurement :: Python class defining the Measurement
            Spectroscopy :: Python class defining the Spectroscopy
            V0 :: Wavelength/Wavenumber of the first data point
            C0,C1,C2 :: Coefficients to calculate the step size in wavelength/wavenumbers between data points
            P0,P1,P2,P3 :: Parameters used to define the double Gaussian ILS of ACS MIR
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Updated Measurement and Spectroscopy classes
        
        CALLING SEQUENCE:
        
            Measurement,Spectroscopy = model228(Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3)
        
        MODIFICATION HISTORY : Juan Alday (20/12/2021)
        
    """

    from NemesisPy import ngauss

    #1.: Defining the new wavelength array
    ##################################################

    nconv = Measurement.NCONV[0]
    vconv1 = np.zeros(nconv)
    vconv1[0] = V0

    xx = np.linspace(0,nconv-2,nconv-1)
    dV = C0 + C1*xx + C2*(xx)**2.

    for i in range(nconv-1):
        vconv1[i+1] = vconv1[i] + dV[i]

    for i in range(Measurement.NGEOM):
        Measurement.VCONV[0:Measurement.NCONV[i],i] = vconv1[:]

    #2.: Calculating the new ILS function based on the new convolution wavelengths
    ###################################################################################

    ng = 2 #Number of gaussians to include

    #Wavenumber offset of the two gaussians
    offset = np.zeros([nconv,ng])
    offset[:,0] = 0.0
    offset[:,1] = P0

    #FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
    fwhm = np.zeros([nconv,ng])
    fwhml = P1 / vconv1[0]**2.0
    for i in range(nconv):
        fwhm[i,0] = fwhml * (vconv1[i])**2.
        fwhm[i,1] = fwhm[i,0]

    #Amplitde of the second gaussian with respect to the main one
    amp = np.zeros([nconv,ng])
    ampgrad = (P3 - P2)/(vconv1[nconv-1]-vconv1[0])
    for i in range(nconv):
        amp[i,0] = 1.0
        amp[i,1] = (vconv1[i] - vconv1[0]) * ampgrad + P2

    #Running for each spectral point
    nfil = np.zeros(nconv,dtype='int32')
    mfil1 = 200
    vfil1 = np.zeros([mfil1,nconv])
    afil1 = np.zeros([mfil1,nconv])
    for i in range(nconv):

        #determining the lowest and highest wavenumbers to calculate
        xlim = 0.0
        xdist = 5.0 
        for j in range(ng):
            xcen = offset[i,j]
            xmin = abs(xcen - xdist*fwhm[i,j]/2.)
            if xmin > xlim:
                xlim = xmin
            xmax = abs(xcen + xdist*fwhm[i,j]/2.)
            if xmax > xlim:
                xlim = xmax

        #determining the wavenumber spacing we need to sample properly the gaussians
        xsamp = 7.0   #number of points we require to sample one HWHM 
        xhwhm = 10000.0
        for j in range(ng):
            xhwhmx = fwhm[i,j]/2. 
            if xhwhmx < xhwhm:
                xhwhm = xhwhmx
        deltawave = xhwhm/xsamp
        np1 = 2.0 * xlim / deltawave
        npx = int(np1) + 1

        #Calculating the ILS in this spectral point
        iamp = np.zeros([ng])
        imean = np.zeros([ng])
        ifwhm = np.zeros([ng])
        fun = np.zeros([npx])
        xwave = np.linspace(vconv1[i]-deltawave*(npx-1)/2.,vconv1[i]+deltawave*(npx-1)/2.,npx)        
        for j in range(ng):
            iamp[j] = amp[i,j]
            imean[j] = offset[i,j] + vconv1[i]
            ifwhm[j] = fwhm[i,j]

        fun = ngauss(npx,xwave,ng,iamp,imean,ifwhm)  
        nfil[i] = npx
        vfil1[0:nfil[i],i] = xwave[:]
        afil1[0:nfil[i],i] = fun[:]

    mfil = nfil.max()
    vfil = np.zeros([mfil,nconv])
    afil = np.zeros([mfil,nconv])
    for i in range(nconv):
        vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
        afil[0:nfil[i],i] = afil1[0:nfil[i],i]
    
    Measurement.NFIL = nfil
    Measurement.VFIL = vfil
    Measurement.AFIL = afil

    #3. Defining new calculations wavelengths and reading again lbl-tables in correct range
    ###########################################################################################

    #Spectroscopy.read_lls(Spectroscopy.RUNNAME)
    #Measurement.wavesetc(Spectroscopy,IGEOM=0)
    #Spectroscopy.read_tables(wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())

    return Measurement,Spectroscopy


###############################################################################################

def model229(Measurement,par1,par2,par3,par4,par5,par6,par7,MakePlot=False):
    
    """
        FUNCTION NAME : model2()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 229 in NEMESIS.
            In this model, the ILS of the measurement is defined from every convolution wavenumber
            using the double-Gaussian parameterisation created for analysing ACS MIR spectra
        
        INPUTS :
        
            Measurement :: Python class defining the Measurement
            par1 :: Wavenumber offset of main at lowest wavenumber
            par2 :: Wavenumber offset of main at wavenumber in the middle
            par3 :: Wavenumber offset of main at highest wavenumber 
            par4 :: Offset of the second gaussian with respect to the first one (assumed spectrally constant)
            par5 :: FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units)
            par6 :: Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
            par7 :: Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var)
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            Updated Measurement class
        
        CALLING SEQUENCE:
        
            Measurement = model229(Measurement,par1,par2,par3,par4,par5,par6,par7)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    from NemesisPy import ngauss

    #Calculating the parameters for each spectral point
    nconv = Measurement.NCONV[0]
    vconv1 = Measurement.VCONV[0:nconv,0]
    ng = 2

    # 1. Wavenumber offset of the two gaussians
    #    We divide it in two sections with linear polynomials     
    iconvmid = int(nconv/2.)
    wavemax = vconv1[nconv-1]
    wavemin = vconv1[0]
    wavemid = vconv1[iconvmid]
    offgrad1 = (par2 - par1)/(wavemid-wavemin)
    offgrad2 = (par2 - par3)/(wavemid-wavemax)
    offset = np.zeros([nconv,ng])
    for i in range(iconvmid):
        offset[i,0] = (vconv1[i] - wavemin) * offgrad1 + par1
        offset[i,1] = offset[i,0] + par4
    for i in range(nconv-iconvmid):
        offset[i+iconvmid,0] = (vconv1[i+iconvmid] - wavemax) * offgrad2 + par3
        offset[i+iconvmid,1] = offset[i+iconvmid,0] + par4

    # 2. FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
    fwhm = np.zeros([nconv,ng])
    fwhml = par5 / wavemin**2.0
    for i in range(nconv):
        fwhm[i,0] = fwhml * (vconv1[i])**2.
        fwhm[i,1] = fwhm[i,0]

    # 3. Amplitde of the second gaussian with respect to the main one
    amp = np.zeros([nconv,ng])
    ampgrad = (par7 - par6)/(wavemax-wavemin)
    for i in range(nconv):
        amp[i,0] = 1.0
        amp[i,1] = (vconv1[i] - wavemin) * ampgrad + par6

    #Running for each spectral point
    nfil = np.zeros(nconv,dtype='int32')
    mfil1 = 200
    vfil1 = np.zeros([mfil1,nconv])
    afil1 = np.zeros([mfil1,nconv])
    for i in range(nconv):

        #determining the lowest and highest wavenumbers to calculate
        xlim = 0.0
        xdist = 5.0 
        for j in range(ng):
            xcen = offset[i,j]
            xmin = abs(xcen - xdist*fwhm[i,j]/2.)
            if xmin > xlim:
                xlim = xmin
            xmax = abs(xcen + xdist*fwhm[i,j]/2.)
            if xmax > xlim:
                xlim = xmax

        #determining the wavenumber spacing we need to sample properly the gaussians
        xsamp = 7.0   #number of points we require to sample one HWHM 
        xhwhm = 10000.0
        for j in range(ng):
            xhwhmx = fwhm[i,j]/2. 
            if xhwhmx < xhwhm:
                xhwhm = xhwhmx
        deltawave = xhwhm/xsamp
        np1 = 2.0 * xlim / deltawave
        npx = int(np1) + 1

        #Calculating the ILS in this spectral point
        iamp = np.zeros([ng])
        imean = np.zeros([ng])
        ifwhm = np.zeros([ng])
        fun = np.zeros([npx])
        xwave = np.linspace(vconv1[i]-deltawave*(npx-1)/2.,vconv1[i]+deltawave*(npx-1)/2.,npx)        
        for j in range(ng):
            iamp[j] = amp[i,j]
            imean[j] = offset[i,j] + vconv1[i]
            ifwhm[j] = fwhm[i,j]

        fun = ngauss(npx,xwave,ng,iamp,imean,ifwhm)  
        nfil[i] = npx
        vfil1[0:nfil[i],i] = xwave[:]
        afil1[0:nfil[i],i] = fun[:]

    mfil = nfil.max()
    vfil = np.zeros([mfil,nconv])
    afil = np.zeros([mfil,nconv])
    for i in range(nconv):
        vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
        afil[0:nfil[i],i] = afil1[0:nfil[i],i]
    
    Measurement.NFIL = nfil
    Measurement.VFIL = vfil
    Measurement.AFIL = afil

    if MakePlot==True:

        fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))
        
        ix = 0  #First wavenumber
        ax1.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
        ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
        ax1.set_ylabel(r'f($\nu$)')
        ax1.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
        ax1.ticklabel_format(useOffset=False)
        ax1.grid()
        
        ix = int(nconv/2)-1  #Centre wavenumber
        ax2.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
        ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
        ax2.set_ylabel(r'f($\nu$)')
        ax2.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
        ax2.ticklabel_format(useOffset=False)
        ax2.grid()
        
        ix = nconv-1  #Last wavenumber
        ax3.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
        ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
        ax3.set_ylabel(r'f($\nu$)')
        ax3.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
        ax3.ticklabel_format(useOffset=False)
        ax3.grid()
        
        plt.tight_layout()
        plt.show()

    return Measurement


###############################################################################################

def model230(Measurement,nwindows,liml,limh,par,MakePlot=False):
    
    """
        FUNCTION NAME : model230()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 230 in NEMESIS.
            In this model, the ILS of the measurement is defined from every convolution wavenumber
            using the double-Gaussian parameterisation created for analysing ACS MIR spectra.
            However, we can define several spectral windows where the ILS is different
        
        INPUTS :
        
            Measurement :: Python class defining the Measurement
            nwindows :: Number of spectral windows in which to fit the ILS
            liml(nwindows) :: Low wavenumber limit of each spectral window
            limh(nwindows) :: High wavenumber limit of each spectral window
            par(0,nwindows) :: Wavenumber offset of main at lowest wavenumber for each window
            par(1,nwindows) :: Wavenumber offset of main at wavenumber in the middle for each window
            par(2,nwindows) :: Wavenumber offset of main at highest wavenumber for each window
            par(3,nwindows) :: Offset of the second gaussian with respect to the first one (assumed spectrally constant) for each window
            par(4,nwindows) :: FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units) for each window
            par(5,nwindows) :: Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber for each window
            par(6,nwindows) :: Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var) for each window
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Updated Measurement class
        
        CALLING SEQUENCE:
        
            Measurement = model230(Measurement,nwindows,liml,limh,par)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    from NemesisPy import ngauss

    #Calculating the parameters for each spectral point
    nconv = Measurement.NCONV[0]
    vconv2 = Measurement.VCONV[0:nconv,0]
    ng = 2

    
    nfil2 = np.zeros(nconv,dtype='int32')
    mfil2 = 200
    vfil2 = np.zeros([mfil2,nconv])
    afil2 = np.zeros([mfil2,nconv])

    ivtot = 0
    for iwindow in range(nwindows):

        #Calculating the wavenumbers at which each spectral window applies
        ivwin = np.where( (vconv2>=liml[iwindow]) & (vconv2<=limh[iwindow]) )
        ivwin = ivwin[0]

        vconv1 = vconv2[ivwin]
        nconv1 = len(ivwin)
        

        par1 = par[0,iwindow]
        par2 = par[1,iwindow]
        par3 = par[2,iwindow]
        par4 = par[3,iwindow]
        par5 = par[4,iwindow]
        par6 = par[5,iwindow]
        par7 = par[6,iwindow]

        # 1. Wavenumber offset of the two gaussians
        #    We divide it in two sections with linear polynomials     
        iconvmid = int(nconv1/2.)
        wavemax = vconv1[nconv1-1]
        wavemin = vconv1[0]
        wavemid = vconv1[iconvmid]
        offgrad1 = (par2 - par1)/(wavemid-wavemin)
        offgrad2 = (par2 - par3)/(wavemid-wavemax)
        offset = np.zeros([nconv,ng])
        for i in range(iconvmid):
            offset[i,0] = (vconv1[i] - wavemin) * offgrad1 + par1
            offset[i,1] = offset[i,0] + par4
        for i in range(nconv1-iconvmid):
            offset[i+iconvmid,0] = (vconv1[i+iconvmid] - wavemax) * offgrad2 + par3
            offset[i+iconvmid,1] = offset[i+iconvmid,0] + par4

        # 2. FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
        fwhm = np.zeros([nconv1,ng])
        fwhml = par5 / wavemin**2.0
        for i in range(nconv1):
            fwhm[i,0] = fwhml * (vconv1[i])**2.
            fwhm[i,1] = fwhm[i,0]

        # 3. Amplitde of the second gaussian with respect to the main one
        amp = np.zeros([nconv1,ng])
        ampgrad = (par7 - par6)/(wavemax-wavemin)
        for i in range(nconv1):
            amp[i,0] = 1.0
            amp[i,1] = (vconv1[i] - wavemin) * ampgrad + par6


        #Running for each spectral point
        nfil = np.zeros(nconv1,dtype='int32')
        mfil1 = 200
        vfil1 = np.zeros([mfil1,nconv1])
        afil1 = np.zeros([mfil1,nconv1])
        for i in range(nconv1):

            #determining the lowest and highest wavenumbers to calculate
            xlim = 0.0
            xdist = 5.0 
            for j in range(ng):
                xcen = offset[i,j]
                xmin = abs(xcen - xdist*fwhm[i,j]/2.)
                if xmin > xlim:
                    xlim = xmin
                xmax = abs(xcen + xdist*fwhm[i,j]/2.)
                if xmax > xlim:
                    xlim = xmax

            #determining the wavenumber spacing we need to sample properly the gaussians
            xsamp = 7.0   #number of points we require to sample one HWHM 
            xhwhm = 10000.0
            for j in range(ng):
                xhwhmx = fwhm[i,j]/2. 
                if xhwhmx < xhwhm:
                    xhwhm = xhwhmx
            deltawave = xhwhm/xsamp
            np1 = 2.0 * xlim / deltawave
            npx = int(np1) + 1

            #Calculating the ILS in this spectral point
            iamp = np.zeros([ng])
            imean = np.zeros([ng])
            ifwhm = np.zeros([ng])
            fun = np.zeros([npx])
            xwave = np.linspace(vconv1[i]-deltawave*(npx-1)/2.,vconv1[i]+deltawave*(npx-1)/2.,npx)        
            for j in range(ng):
                iamp[j] = amp[i,j]
                imean[j] = offset[i,j] + vconv1[i]
                ifwhm[j] = fwhm[i,j]

            fun = ngauss(npx,xwave,ng,iamp,imean,ifwhm)  
            nfil[i] = npx
            vfil1[0:nfil[i],i] = xwave[:]
            afil1[0:nfil[i],i] = fun[:]

        

        nfil2[ivtot:ivtot+nconv1] = nfil[:]
        vfil2[0:mfil1,ivtot:ivtot+nconv1] = vfil1[0:mfil1,:]
        afil2[0:mfil1,ivtot:ivtot+nconv1] = afil1[0:mfil1,:]

        ivtot = ivtot + nconv1

    if ivtot!=nconv:
        sys.exit('error in model 230 :: The spectral windows must cover the whole measured spectral range')

    mfil = nfil2.max()
    vfil = np.zeros([mfil,nconv])
    afil = np.zeros([mfil,nconv])
    for i in range(nconv):
        vfil[0:nfil2[i],i] = vfil2[0:nfil2[i],i]
        afil[0:nfil2[i],i] = afil2[0:nfil2[i],i]
    
    Measurement.NFIL = nfil2
    Measurement.VFIL = vfil
    Measurement.AFIL = afil

    if MakePlot==True:

        fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))
        
        ix = 0  #First wavenumber
        ax1.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
        ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
        ax1.set_ylabel(r'f($\nu$)')
        ax1.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
        ax1.ticklabel_format(useOffset=False)
        ax1.grid()
        
        ix = int(nconv/2)-1  #Centre wavenumber
        ax2.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
        ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
        ax2.set_ylabel(r'f($\nu$)')
        ax2.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
        ax2.ticklabel_format(useOffset=False)
        ax2.grid()
        
        ix = nconv-1  #Last wavenumber
        ax3.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
        ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
        ax3.set_ylabel(r'f($\nu$)')
        ax3.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
        ax3.ticklabel_format(useOffset=False)
        ax3.grid()
        
        plt.tight_layout()
        plt.show()

    return Measurement


###############################################################################################

def model446(Scatter,idust,wavenorm,xwave,rsize,lookupfile,MakePlot=False):
    
    """
        FUNCTION NAME : model446()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 446 in NEMESIS.
            
            In this model, we change the extinction coefficient and single scattering albedo 
            of a given aerosol population based on its particle size, and based on the extinction 
            coefficients tabulated in a look-up table
        
        INPUTS :
        
            Scatter :: Python class defining the scattering parameters
            idust :: Index of the aerosol distribution to be modified (from 0 to NDUST-1)
            wavenorm :: Flag indicating if the extinction coefficient needs to be normalised to a given wavelength (1 if True)
            xwave :: If wavenorm=1, then this indicates the normalisation wavelength/wavenumber
            rsize :: Particle size at which to interpolate the extinction cross section
            lookupfile :: Name of the look-up file storing the extinction cross section data
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            Scatter :: Updated Scatter class
        
        CALLING SEQUENCE:
        
            Scatter = model446(Scatter,idust,wavenorm,xwave,rsize,lookupfile)
        
        MODIFICATION HISTORY : Juan Alday (25/11/2021)
        
    """
    
    import h5py
    from scipy.interpolate import interp1d

    #Reading the look-up table file
    f = h5py.File(lookupfile,'r')
    
    NWAVE = np.int32(f.get('NWAVE'))
    NSIZE = np.int32(f.get('NSIZE'))
     
    WAVE = np.array(f.get('WAVE'))
    REFF = np.array(f.get('REFF'))
     
    KEXT = np.array(f.get('KEXT'))      #(NWAVE,NSIZE)
    SGLALB = np.array(f.get('SGLALB'))  #(NWAVE,NSIZE)
    
    f.close()
    
    #First we interpolate to the wavelengths in the Scatter class
    sext = interp1d(WAVE,KEXT,axis=0)
    KEXT1 = sext(Scatter.WAVE)
    salb = interp1d(WAVE,SGLALB,axis=0)
    SGLALB1 = salb(Scatter.WAVE)
    
    #Second we interpolate to the required particle size
    if rsize<REFF.min():
        rsize =REFF.min()
    if rsize>REFF.max():
        rsize=REFF.max()
        
    sext = interp1d(REFF,KEXT1,axis=1)
    KEXTX = sext(rsize)
    salb = interp1d(REFF,SGLALB1,axis=1)
    SGLALBX = salb(rsize)
    
    #Now check if we need to normalise the extinction coefficient
    if wavenorm==1:
        snorm = interp1d(Scatter.WAVE,KEXTX)
        vnorm = snorm(xwave)
      
        KEXTX[:] = KEXTX[:] / vnorm
      
    KSCAX = SGLALBX * KEXTX
    
    #Now we update the Scatter class with the required results
    Scatter.KEXT[:,idust] = KEXTX[:]
    Scatter.KSCA[:,idust] = KSCAX[:]
    Scatter.SGLALB[:,idust] = SGLALBX[:]
      
    f.close()
    
    if MakePlot==True:
        
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6),sharex=True)
    
        for i in range(NSIZE):
            
            ax1.plot(WAVE,KEXT[:,i])
            ax2.plot(WAVE,SGLALB[:,i])
            
        ax1.plot(Scatter.WAVE,Scatter.KEXT[:,idust],c='black')
        ax2.plot(Scatter.WAVE,Scatter.SGLALB[:,idust],c='black')

        if Scatter.ISPACE==0:
            label='Wavenumber (cm$^{-1}$)'
        else:
            label='Wavelength ($\mu$m)'
        ax2.set_xlabel(label)
        ax1.set_xlabel('Extinction coefficient')
        ax2.set_xlabel('Single scattering albedo')
        
        ax1.set_facecolor('lightgray')
        ax2.set_facecolor('lightgray')
        
        plt.tight_layout()


    return Scatter


###############################################################################################

def model447(Measurement,v_doppler):
    
    """
        FUNCTION NAME : model447()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 447 in NEMESIS.
            In this model, we fit the Doppler shift of the observation. Currently this Doppler shift
            is common to all geometries, but in the future it will be updated so that each measurement
            can have a different Doppler velocity (in order to retrieve wind speeds).
        
        INPUTS :
        
            Measurement :: Python class defining the measurement
            v_doppler :: Doppler velocity (km/s)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Measurement :: Updated measurement class with the correct Doppler velocity
        
        CALLING SEQUENCE:
        
            Measurement = model447(Measurement,v_doppler)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2023)
        
    """
    
    Measurement.V_DOPPLER = v_doppler
    
    return Measurement


###############################################################################################

def model667(Spectrum,xfactor,MakePlot=False):
    
    """
        FUNCTION NAME : model667()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 667 in NEMESIS.
            In this model, the output spectrum is scaled using a dillusion factor to account
            for strong temperature gradients in exoplanets
        
        INPUTS :
        
            Spectrum :: Modelled spectrum 
            xfactor :: Dillusion factor
        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            Spectrum :: Modelled spectrum scaled by the dillusion factor
        
        CALLING SEQUENCE:
        
            Spectrum = model667(Spectrum,xfactor)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    Spectrum = Spectrum * xfactor

    return Spectrum


###############################################################################################

def model777(Measurement,hcorr,MakePlot=False):
    
    """
        FUNCTION NAME : model777()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 777 in NEMESIS.
            In this model, we apply a correction to the tangent heights listed on the 
            Measurement class
        
        INPUTS :
        
            Measurement :: Measurement class
            hcorr :: Correction to the tangent heights (km)
        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            Measurement :: Updated Measurement class with corrected tangent heights
        
        CALLING SEQUENCE:
        
            Measurement = model777(Measurement,hcorr)
        
        MODIFICATION HISTORY : Juan Alday (15/02/2023)
        
    """
    
    #Getting the tangent heights
    tanhe = np.zeros(Measurement.NGEOM)
    tanhe[:] = Measurement.TANHE[:,0]
    
    #Correcting tangent heights
    tanhe_new = tanhe + hcorr
    
    #Updating Measurement class
    Measurement.TANHE[:,0] = tanhe_new
    
    if MakePlot==True:
        
        fig,ax1 = plt.subplots(1,1,figsize=(3,4))
        ax1.scatter(np.arange(0,Measurement.NGEOM,1),tanhe,label='Uncorrected')
        ax1.scatter(np.arange(0,Measurement.NGEOM,1),Measurement.TANHE[:,0],label='Corrected')
        ax1.set_xlabel('Geometry #')
        ax1.set_ylabel('Tangent height (km)')
        plt.tight_layout()

    return Measurement


###############################################################################################

def model887(Scatter,xsc,idust,MakePlot=False):
    
    """
        FUNCTION NAME : model887()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 887 in NEMESIS.
            In this model, the cross-section spectrum of IDUST is changed given the parameters in 
            the state vector
        
        INPUTS :
        
            Scatter :: Python class defining the spectral properties of aerosols in the atmosphere
            xsc :: New cross-section spectrum of aerosol IDUST
            idust :: Index of the aerosol to be changed (from 0 to NDUST-1)
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            Scatter :: Updated Scatter class
        
        CALLING SEQUENCE:
        
            Scatter = model887(Scatter,xsc,idust)
        
        MODIFICATION HISTORY : Juan Alday (29/03/2021)
        
    """

    if len(xsc)!=Scatter.NWAVE:
        sys.exit('error in model 887 :: Cross-section array must be defined at the same wavelengths as in .xsc')
    else:
        kext = np.zeros([Scatter.NWAVE,Scatter.DUST])
        kext[:,:] = Scatter.KEXT
        kext[:,idust] = xsc[:]
        Scatter.KEXT = kext

    if MakePlot==True:
        fig,ax1=plt.subplots(1,1,figsize=(10,3))
        ax1.semilogy(Scatter.WAVE,Scatter.KEXT[:,idust])
        ax1.grid()
        if Scatter.ISPACE==1:
            ax1.set_xlabel('Wavelength ($\mu$m)')
        else:
            ax1.set_xlabel('Wavenumber (cm$^{-1}$')
        plt.tight_layout()
        plt.show()
    

###############################################################################################

def model1002(atm,ipar,scf,MakePlot=False):
    
    """
        FUNCTION NAME : model2()
        
        DESCRIPTION :
        
            Function defining the model parameterisation 1002 in NEMESIS.
            
            This is the same as model 2, but applied simultaneously in different planet locations
            In this model, the atmospheric parameters are scaled using a single factor with 
            respect to the vertical profiles in the reference atmosphere
        
        INPUTS :
        
            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            scf(nlocations) :: Scaling factors at the different locations
        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            atm :: Updated atmospheric class
            xmap(nlocations,ngas+2+ncont,npro,nlocations) :: Matrix of relating funtional derivatives to 
                                                             elements in state vector
        
        CALLING SEQUENCE:
        
            atm,xmap = model1002(atm,ipar,scf)
        
        MODIFICATION HISTORY : Juan Alday (19/04/2023)
        
    """

    npar = atm.NVMR+2+atm.NDUST
    xmap = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))
    xmap1 = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))

    if len(scf)!=atm.NLOCATIONS:
        sys.exit('error in model 1002 :: The number of scaling factors must be the same as the number of locations in Atmosphere')

    if atm.NLOCATIONS<=1:
        sys.exit('error in model 1002 :: This model can be applied only if NLOCATIONS>1')

    x1 = np.zeros((atm.NP,atm.NLOCATIONS))
    xref = np.zeros((atm.NP,atm.NLOCATIONS))
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        xref[:,:] = atm.VMR[:,jvmr,:]
        x1[:,:] = atm.VMR[:,jvmr,:] * scf[:]
        atm.VMR[:,jvmr,:] =  x1
    elif ipar==atm.NVMR: #Temperature
        xref[:] = atm.T[:,:]
        x1[:] = np.transpose(np.transpose(atm.T[:,:]) * scf[:])
        atm.T[:,:] = x1 
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST:
            xref[:] = atm.DUST[:,jtmp,:]
            x1[:] = np.transpose(np.transpose(atm.DUST[:,jtmp,:]) * scf[:])
            atm.DUST[:,jtmp,:] = x1
        elif jtmp==atm.NDUST:
            xref[:] = atm.PARAH2[:,:]
            x1[:] = np.transpose(np.transpose(atm.PARAH2[:,:]) * scf)
            atm.PARAH2[:,:] = x1
        elif jtmp==atm.NDUST+1:
            xref[:] = atm.FRAC[:,:]
            x1[:] = np.transpose(np.transpose(atm.FRAC[:,:]) * scf)
            atm.FRAC[:,:] = x1


    #This calculation takes a long time for big arrays
    #for j in range(atm.NLOCATIONS):
    #    xmap[j,ipar,:,j] = xref[:,j]
    
        
    if MakePlot==True:
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        fig,ax1 = plt.subplots(1,1,figsize=(6,4))
        im1 = ax1.scatter(atm.LONGITUDE,atm.LATITUDE,c=scf,cmap='jet',vmin=scf.min(),vmax=scf.max())
        ax1.grid()
        ax1.set_xlabel('Longitude / deg')
        ax1.set_ylabel('Latitude / deg')
        ax1.set_xlim(-180.,180.)
        ax1.set_ylim(-90.,90.)
        ax1.set_title('Model 1002')
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax)
        cbar1.set_label('Scaling factor')
        
        plt.tight_layout()
        plt.show()
 
    return atm,xmap