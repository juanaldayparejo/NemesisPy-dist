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
    elif ipar==atm.NVMR: #Temperature
        x1 = xprof
        atm.edit_T(x1)
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        x1 = np.exp(xprof)
        if jtmp<atm.NDUST: #Dust in cm-3
            dust = np.zeros([atm.NP,atm.NDUST])
            dust[:,:] = atm.DUST
            dust[:,jtmp] = x1
            atm.edit_DUST(dust)
        elif jtmp==atm.NDUST:
            atm.PARAH2 = x1
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
    if ipar<atm.NVMR:  #Gas VMR
        jvmr = ipar
        x1[:] = atm.VMR[:,jvmr] * scf
        atm.VMR[:,jvmr] =  x1
    elif ipar==atm.NVMR: #Temperature
        x1[:] = atm.T[:] * scf
        atm.T[:] = x1 
    elif ipar>atm.NVMR:
        jtmp = ipar - (atm.NVMR+1)
        if jtmp<atm.NDUST:
            x1[:] = atm.DUST[:,jtmp] * scf
            atm.DUST[:,jtmp] = x1
        elif jtmp==atm.NDUST:
            x1[:] = atm.PARAH2 * scf
            atm.PARAH2 = x1
        elif jtmp==atm.NDUST+1:
            x1[:] = atm.FRAC * scf
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

def model3(atm,ipar,scf,MakePlot=False):
    
    """
        FUNCTION NAME : model2()
        
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
    
