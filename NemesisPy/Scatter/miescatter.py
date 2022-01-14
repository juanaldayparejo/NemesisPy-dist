# NAME:
#       miescatter.py (NemesisPy)
#
# DESCRIPTION:
#
#	This library contains functions to handle the scattering and absorption properties of 
#   aerosols       
#
# CATEGORY:
#
#	NEMESIS
# 
# MODIFICATION HISTORY: Juan Alday 15/07/2021

import numpy as np
from struct import *
import sys,os
import matplotlib.pyplot as plt
from NemesisPy import *
from copy import *
import miepython

###############################################################################################

def read_ref_index(filename,MakePlot=False):


    """
        FUNCTION NAME : calc_serr()
        
        DESCRIPTION : Read a file of the refractive index using the format required by NEMESIS

        INPUTS :
       
            filename :: Name of the file

        OPTIONAL INPUTS: none
        
        OUTPUTS :

            nwave :: Number of spectral points
            ispace :: Wavenumber (0) or wavelength (1)
            wave(nwave) :: Spectral points
            refind_real(nwave) :: Real part of refractive index
            refind_im(nwave) :: Imaginary part of refractive index
 
        CALLING SEQUENCE:
        
            nwave,ispace,wave,refind_real,refind_im = read_ref_index(filename)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    from NemesisPy import file_lines

    nlines = file_lines(filename)

    #Reading buffer
    ibuff = 0
    with open(filename,'r') as fsol:
        for curline in fsol:
            if curline.startswith("#"):
                ibuff = ibuff + 1
            else:
                break

    f = open(filename,'r')

    #Reading buffer
    for i in range(ibuff):
        s = f.readline().split()

    #Reading file
    s = f.readline().split()
    nwave = int(s[0])
    s = f.readline().split()
    ispace = int(s[0])

    wave = np.zeros(nwave)
    refind_real = np.zeros(nwave)
    refind_im = np.zeros(nwave)
    for i in range(nwave):
        s = f.readline().split()
        wave[i] = float(s[0])
        refind_real[i] = float(s[1])
        refind_im[i] = float(s[2])

    f.close()

    if MakePlot==True:

        fig,ax1 = plt.subplots(1,1,figsize=(8,3))

        ax1.plot(wave,refind_real,c='tab:blue',label='Real part')
        ax1.plot(wave,refind_im,c='tab:red',label='Imaginary part')
        ax1.grid()
        ax1.legend()
        if ispace==0:
            ax1.set_xlabel('Wavenumber (cm$^{-1}$')
        elif ispace==1:
            ax1.set_xlabel('Wavelength ($\mu$m)')
        ax1.set_ylabel('Refractive index')
        plt.tight_layout()
        plt.show()


    return nwave,ispace,wave,refind_real,refind_im

###############################################################################################

def miescat(ispace,wave,refind_real,refind_im,psdist,pardist,theta,MakePlot=False,rdist=None,Ndist=None):


    """
        FUNCTION NAME : calc_serr()
        
        DESCRIPTION : Read a file of the refractive index using the format required by NEMESIS

        INPUTS :
       
            ispace :: Spectral units (0) Wavenumber (cm-1) (1) Wavelength (um)
            wave(nwave) :: Spectral points
            refind_real(nwave) :: Real part of refractive index
            refind_im(nwave) :: Imaginary part of refractive index
            psdist :: Flag indicating the particle size distribution
                0 :: Single particle size
                1 :: Log-normal distribution
                2 :: Standard gamma distribution
               -1 :: Other input particle size distribution (using optional inputs)
            pardist :: Particle size distribution parameters
                0 :: pardist(0) == particle size in microns
                1 :: pardist(0) == mu
                     pardist(1) == sigma
                2 :: 
            theta(ntheta) :: Angles at which to return the phase function

        OPTIONAL INPUTS:

            rdist :: Array of particle sizes (to be used if pardist==-1)
            Ndist :: Density of particles of each particle size (to be used if pardist==-1)
        
        OUTPUTS :

            XEXT(nwave) :: Mean extinction cross-section (cm**2)
            XSCA(nwave) :: Mean scattering cross-section (cm**2)
            XABS(nwave) :: Mean absorption cross-section (cm**2)
            XPHASE(nwave,ntheta) :: Mean Phase function

 
        CALLING SEQUENCE:
        
            xext,xsca,xabs,phase = miescat(ispace,wave,refind_real,refind_im,psdist,pardist,theta)
 
        MODIFICATION HISTORY : Juan Alday (21/07/2021)

    """

    from scipy.integrate import simpson
    from scipy.stats import lognorm
    from NemesisPy import lognormal_dist

    #First we determine the number of particle sizes to later perform the integration
    ######################################################################################
    
    nwave = len(wave)
    ntheta = len(theta)


    if psdist==0:   #Single particle size
        nr = 1
        rd = np.zeros(nr)   #Array of particle sizes
        Nd = np.zeros(nr)   #Weight (tipycally density) of each particle size for the integration
        rd[0] = pardist[0]
        Nd[0] = 1.0
    elif psdist==1: #Log-normal distribution
        mu = pardist[0]
        sigma = pardist[1]
        r0 = lognorm.ppf(0.0001, sigma, 0.0, mu)
        r1 = lognorm.ppf(0.9999, sigma, 0.0, mu)
        rmax = np.exp( np.log(mu) - sigma**2.)
        delr = (rmax-r0)/10.
        nr = int((r1-r0)/delr) + 1
        rd = np.linspace(r0,r1,nr) #Array of particle sizes
        Nd = lognormal_dist(rd,mu,sigma) #Density of each particle size for the integration
    elif psdist==2: #Standard gamma distribution
        sys.exit('error in miescat :: Standard gamma distribution has not yet been implemented')
    elif psdist==-1:
        if rdist[0]==None:
            sys.exit('error in miescat :: If psdist=-1 rdist and Ndist must be filled')
        if Ndist[0]==None:
            sys.exit('error in miescat :: If psdist=-1 rdist and Ndist must be filled')
        nr = len(rdist)
        rd = np.zeros(nr)   #Array of particle sizes
        Nd = np.zeros(nr)   #Weight (tipycally density) of each particle size for the integration
        rd[:] = rdist
        Nd[:] = Ndist
    else:
        sys.exit('error in miescat :: Type of distribution not yet implemented')


    #Second we change the units of the spectral points if they are in wavenumbers
    ######################################################################################

    if ispace==0:  #Wavenumber
        wave1 = copy(wave)
        wave = 1.0e4/wave1
        isort = np.argsort(wave)
        wave = wave[isort]
        refind_real1 = refind_real[isort]
        refind_im1 = refind_im[isort]
        refind = refind_real1 - 1.0j * refind_im1
    elif ispace==1:  #Wavelength
        refind = refind_real - 1.0j * refind_im
    else:
        sys.exit('error in miescat :: ISPACE must be either 0 or 1')

    #We calculate the scattering and absorption properties of each particle size
    ######################################################################################

    kext = np.zeros([nwave,nr])
    ksca = np.zeros([nwave,nr])
    phase = np.zeros([nwave,ntheta,nr])
    for ir in range(nr):

        r0 = rd[ir]
        x = np.zeros(nwave)
        x[:] = 2.*np.pi*r0/(wave[:])
        qext, qsca, qback, g = miepython.mie(refind,x)

        ksca[:,ir] = qsca * np.pi * (r0/1.0e4)**2.   #Cross section in cm2
        kext[:,ir] = qext * np.pi * (r0/1.0e4)**2.   #Cross section in cm2

        mu = np.cos(theta/180.*np.pi)
        for iwave in range(nwave):
            unpolar = miepython.i_unpolarized(refind[iwave],x[iwave],mu)
            phase[iwave,:,ir] = unpolar


    #Now integrating over particle size to find the mean scattering and absorption properties
    ###########################################################################################

    if nr>1:
        kext1 = np.zeros([nwave,nr])
        ksca1 = np.zeros([nwave,nr])
        phase1 = np.zeros([nwave,ntheta,nr])
        for ir in range(nr):
            kext1[:,ir] = kext[:,ir] * Nd[ir]
            ksca1[:,ir] = ksca[:,ir] * Nd[ir]
            phase1[:,:,ir] = phase[:,:,ir] * Nd[ir]

        #Integrating the arrays
        kextout = simpson(kext1,x=rd,axis=1)
        kscaout = simpson(ksca1,x=rd,axis=1)
        phaseout = simpson(phase1,x=rd,axis=2)

        #Integrating the particle size distribution
        pnorm = simpson(Nd,x=rd)

        #Calculating the mean properties
        xext = np.zeros(nwave)
        xsca = np.zeros(nwave)
        xabs = np.zeros(nwave)
        xext[:] = kextout/pnorm
        xsca[:] = kscaout/pnorm
        xabs = xext - xsca
        xphase = np.zeros([nwave,ntheta])
        xphase[:,:] = phaseout/pnorm

    else:
        xext = np.zeros(nwave)
        xsca = np.zeros(nwave)
        xabs = np.zeros(nwave)
        xext[:] = kext[:,0]  
        xsca[:] = ksca[:,0]
        xabs = xext - xsca
        xphase = np.zeros([nwave,ntheta])
        xphase = phase[:,:,0]

    #Sorting again the arrays if ispace=0
    if ispace==0:  #Wavenumber
        wave = 1.0e4/wave
        isort = np.argsort(wave)
        wave = wave[isort]
        xext = xext[isort]
        xsca = xsca[isort]
        xabs = xabs[isort]
        xphase[:,:] = xphase[isort,:]


    if MakePlot==True:

        fig,ax1 = plt.subplots(1,1,figsize=(14,7))

        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1,rowspan=2)
        ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2,rowspan=1)
        ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2,rowspan=1)

        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1,rowspan=1)
        ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1,rowspan=1)
        ax6 = plt.subplot2grid((3, 3), (2, 2), colspan=1,rowspan=1)

        if nr>1:

            import matplotlib

            colormap = 'viridis'
            norm = matplotlib.colors.Normalize(vmin=Nd.min(),vmax=Nd.max())
            c_m = plt.cm.get_cmap(colormap,360)
            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])

            #Plotting the extinction cross section 
            iwave0 = 0
            iwave1 = int(nwave/2-1)
            iwave2 = nwave-1
            for ir in range(nr):
                ax1.scatter(rd[ir],Nd[ir],c=s_m.to_rgba([Nd[ir]]),edgecolors='black')
                ax2.semilogy(wave,kext[:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                ax3.semilogy(wave,ksca[:,ir]/kext[:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                ax4.plot(theta,phase[iwave0,:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                ax5.plot(theta,phase[iwave1,:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
                ax6.plot(theta,phase[iwave2,:,ir],c=s_m.to_rgba([Nd[ir]]),linewidth=0.75)
            ax2.semilogy(wave,xext,c='black')
            ax3.semilogy(wave,xsca/xext,c='black')
            ax4.plot(theta,xphase[iwave0,:],c='black')
            ax5.plot(theta,xphase[iwave1,:],c='black')
            ax6.plot(theta,xphase[iwave2,:],c='black')
            if ispace==0:
                ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
                ax3.set_xlabel('Wavenumber (cm$^{-1}$)')
            else:
                ax2.set_xlabel('Wavelength ($\mu$m)')
                ax3.set_xlabel('Wavelength ($\mu$m)')
            ax1.set_ylabel('Density distribution')
            ax2.set_ylabel('Extinction cross section (cm$^2$)')
            ax3.set_ylabel('Single scattering albedo')
            ax4.set_ylabel('Phase function')
            ax5.set_ylabel('Phase function')
            ax6.set_ylabel('Phase function')

            ax1.set_xlabel('Particle size radii ($\mu$m)')
            ax4.set_xlabel('Angle (deg)')
            ax5.set_xlabel('Angle (deg)')
            ax6.set_xlabel('Angle (deg)')
            

            cax = plt.axes([0.92, 0.15, 0.02, 0.7])   #Bottom
            cbar2 = plt.colorbar(s_m,cax=cax,orientation='vertical')
            cbar2.set_label('Density distribution')

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax4.grid()
            ax5.grid()
            ax6.grid()


            plt.subplots_adjust(left=0.05, bottom=0.08, right=0.9, top=0.95, wspace=0.35, hspace=0.35)

        #plt.show()

    return xext,xsca,xabs,xphase