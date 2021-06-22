# NAME:
#       spectroscopy.py (nemesislib)
#
# DESCRIPTION:
#
#	This library contains functions to perform spectroscopic calculations of gaseous species
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
from NemesisPy.Utils.Utils import find_nearest
#import nemesislib.files as files

###############################################################################################

def read_ktahead(filename):
    
    """
        FUNCTION NAME : read_ktahead_nemesis()
        
        DESCRIPTION : Read the header information in a correlated-k look-up table written with the standard format of Nemesis
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .kta extension)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            nwave :: Number of wavelength points
            vmin :: Minimum wavelength
            delv :: Spectral sampling
            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: RADTRAN gas ID
            isoID :: RADTRAN isotopologue ID
            pressleves(np) :: Pressure levels (atm)
            templeves(np) :: Temperature levels (K)
        
        CALLING SEQUENCE:
        
            nwave,vmin,delv,fwhm,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels = read_ktahead(filename)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    #Opening file
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'kta':
        f = open(filename,'r')
    else:
        f = open(filename+'.kta','r')

    irec0 = np.fromfile(f,dtype='int32',count=1)
    nwave = np.fromfile(f,dtype='int32',count=1)
    vmin = np.fromfile(f,dtype='float32',count=1)
    delv = np.fromfile(f,dtype='float32',count=1)
    fwhm = np.fromfile(f,dtype='float32',count=1)
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    ng = int(np.fromfile(f,dtype='int32',count=1))
    gasID = int(np.fromfile(f,dtype='int32',count=1))
    isoID = int(np.fromfile(f,dtype='int32',count=1))

    g_ord = np.fromfile(f,dtype='float32',count=ng)
    del_g = np.fromfile(f,dtype='float32',count=ng)

    presslevels = np.fromfile(f,dtype='float32',count=npress)

    N1 = abs(ntemp)
    if ntemp < 0:
        templevels = np.zeros([npress,n1])
        for i in range(npress):
            for j in range(n1):
                templevels[i,j] =  np.fromfile(f,dtype='float32',count=1)
    else:
        templevels = np.fromfile(f,dtype='float32',count=ntemp)
    
    return nwave,vmin,delv,fwhm,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels


###############################################################################################


def read_ltahead(filename):
    
    """
        FUNCTION NAME : read_ltahead()
        
        DESCRIPTION : Read the header information in a line-by-line look-up table written with the standard format of Nemesis
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .lta extension)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            nwave :: Number of wavelength points
            vmin :: Minimum wavelength
            delv :: Spectral sampling
            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: RADTRAN gas ID
            isoID :: RADTRAN isotopologue ID
            pressleves(np) :: Pressure levels (atm)
            templeves(np) :: Temperature levels (K)
        
        CALLING SEQUENCE:
        
            nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels = read_ltahead(filename)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    #Opening file
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'lta':
        f = open(filename,'r')
    else:
        f = open(filename+'.lta','r')
    
    irec0 = np.fromfile(f,dtype='int32',count=1)
    nwave = np.fromfile(f,dtype='int32',count=1)
    vmin = np.fromfile(f,dtype='float32',count=1)
    delv = np.fromfile(f,dtype='float32',count=1)
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    gasID = int(np.fromfile(f,dtype='int32',count=1))
    isoID = int(np.fromfile(f,dtype='int32',count=1))

    presslevels = np.fromfile(f,dtype='float32',count=npress)
    templevels = np.fromfile(f,dtype='float32',count=ntemp)

    return nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels


###############################################################################################
def read_lbltable(filename,wavemin,wavemax):
    
    """
        FUNCTION NAME : read_lbltable()
        
        DESCRIPTION : Read the line-by-line look-up table written with the standard format of Nemesis
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .lta extension)
            wavemin :: Minimum wavenumber to read (cm-1)
            wavemax :: Maximum wavenumber to read (cm-1)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: RADTRAN gas ID
            isoID :: RADTRAN isotopologue ID
            presslevels(np) :: Pressure levels (atm)
            templevels(np) :: Temperature levels (K)
            nwave :: Number of wavenumbers
            wave :: Wavenumber array (cm-1)
            k(nwave,np,nt) :: Absorption coefficient at each p-T point (cm2)
        
        CALLING SEQUENCE:
        
            npress,ntemp,gasID,isoID,presslevels,templevels,nwave,wave,k = read_lbltable(filename,wavemin,wavemax)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """
    
    #Opening file
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'lta':
        f = open(filename,'rb')
    else:
        f = open(filename+'.lta','rb')
    
    nbytes_int32 = 4
    nbytes_float32 = 4
    
    #Reading header
    irec0 = int(np.fromfile(f,dtype='int32',count=1))
    nwavelta = np.fromfile(f,dtype='int32',count=1)
    vmin = np.fromfile(f,dtype='float32',count=1)
    delv = np.fromfile(f,dtype='float32',count=1)
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    gasID = int(np.fromfile(f,dtype='int32',count=1))
    isoID = int(np.fromfile(f,dtype='int32',count=1))
    
    presslevels = np.fromfile(f,dtype='float32',count=npress)
    templevels = np.fromfile(f,dtype='float32',count=ntemp)
    
    ioff = 8*nbytes_int32+npress*nbytes_float32+ntemp*nbytes_float32
    
    #Calculating the wavenumbers to be read
    wavelta = np.arange(nwavelta)*delv[0] + vmin[0]
    wavelta = np.round(wavelta,3)
    ins1 = np.where( (wavelta>=wavemin) & (wavelta<=wavemax) )
    ins = ins1[0]
    nwave = len(ins)
    wave = np.zeros([nwave])
    wave[:] = wavelta[ins]
    
    #Reading the absorption coefficients
    #######################################
    
    k = np.zeros([nwave,npress,ntemp])
    
    #Jumping until we get to the minimum wavenumber
    njump = npress*ntemp*ins[0]
    ioff = njump*nbytes_float32 + (irec0-1)*nbytes_float32
    f.seek(ioff,0)
    
    #Reading the coefficients we require
    k_out = np.fromfile(f,dtype='float32',count=ntemp*npress*nwave)
    il = 0
    for ik in range(nwave):
        for i in range(npress):
            k[ik,i,:] = k_out[il:il+ntemp]
            il = il + ntemp

    f.close()

    return npress,ntemp,gasID,isoID,presslevels,templevels,nwave,wave,k


###############################################################################################
def read_ktable(filename,wavemin,wavemax):
    
    """
        FUNCTION NAME : read_ktable()
        
        DESCRIPTION : Read the correlated-k look-up table written with the standard format of Nemesis
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .kta extension)
            wavemin :: Minimum wavenumber to read (cm-1)
            wavemax :: Maximum wavenumber to read (cm-1)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            gasID :: Nemesis gas identifier
            isoID :: Nemesis isotopologue identifier
            nwave :: Number of wavenumbers
            wave(nwave) :: Wavenumbers or wavelengths
            fwhm :: Full width at half maximum
            ng :: Number of g-ordinates
            g_ord(ng) :: g-ordinates
            del_g(ng) :: Intervals of g-ordinates
            npress :: Number of pressure levels
            presslevels(npress) :: Pressure levels (atm)
            ntemp :: Number of temperature levels
            templevels(ntemp) :: Temperature levels (K)
            k_g(nwave,ng,npress,ntemp) :: K coefficients
        
        CALLING SEQUENCE:
        
            gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g = read_ktable(filename,wavemin,wavemax)
        
        MODIFICATION HISTORY : Juan Alday (05/03/2021)
        
    """
    
    #Opening file
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'kta':
        f = open(filename,'rb')
    else:
        f = open(filename+'.kta','rb')

    nbytes_int32 = 4
    nbytes_float32 = 4
    ioff = 0
    
    #Reading header
    irec0 = int(np.fromfile(f,dtype='int32',count=1))
    nwavekta = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    fwhm = float(np.fromfile(f,dtype='float32',count=1))
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    ng = int(np.fromfile(f,dtype='int32',count=1))
    gasID = int(np.fromfile(f,dtype='int32',count=1))
    isoID = int(np.fromfile(f,dtype='int32',count=1))
    
    ioff = ioff + 10 * nbytes_int32

    g_ord = np.zeros([ng])
    del_g = np.zeros([ng])
    templevels = np.zeros([ntemp])
    presslevels = np.zeros([npress])
    g_ord[:] = np.fromfile(f,dtype='float32',count=ng)
    del_g[:] = np.fromfile(f,dtype='float32',count=ng)
    
    ioff = ioff + 2*ng*nbytes_float32

    dummy = np.fromfile(f,dtype='float32',count=1)
    dummy = np.fromfile(f,dtype='float32',count=1)

    ioff = ioff + 2*nbytes_float32

    presslevels[:] = np.fromfile(f,dtype='float32',count=npress)
    templevels[:] = np.fromfile(f,dtype='float32',count=ntemp)
    
    ioff = ioff + npress*nbytes_float32+ntemp*nbytes_float32

    dummy = np.fromfile(f,dtype='float32',count=1) 
    dummy = np.fromfile(f,dtype='float32',count=1)
    
    ioff = ioff + 2*nbytes_float32

    #Reading central wavelengths in non-uniform grid
    if delv>0.0:
        vmax = delv*nwavekta + vmin
        wavetot = np.linspace(vmin,vmax,nwavekta)
    else:
        wavetot = np.zeros([nwavekta])
        wavetot[:] = np.fromfile(f,dtype='float32',count=nwavekta)
        ioff = ioff + nwavekta*nbytes_float32

    #Calculating the wavenumbers to be read
    ins1 = np.where( (wavetot>=wavemin) & (wavetot<=wavemax) )
    ins = ins1[0]
    nwave = len(ins)
    wave = np.zeros([nwave])
    wave[:] = wavetot[ins]

    #Reading the k-coefficients
    #######################################

    k_g = np.zeros([nwave,ng,npress,ntemp])

    #Jumping until we get to the minimum wavenumber
    njump = npress*ntemp*ng*ins[0]
    ioff = njump*nbytes_float32 + (irec0-1)*nbytes_float32
    f.seek(ioff,0)
    
    
    #Reading the coefficients we require
    k_out = np.fromfile(f,dtype='float32',count=ntemp*npress*ng*nwave)
    il = 0
    for ik in range(nwave):
        for i in range(npress):
            for j in range(ntemp):
                k_g[ik,:,i,j] = k_out[il:il+ng]
                il = il + ng

    f.close()

    return gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g

###############################################################################################

def calc_klbl(filename,nwave,wave,npoints,press,temp,MakePlot=False):
    
    """
        
        FUNCTION NAME : calc_klbl()
        
        DESCRIPTION : Calculate the absorption coefficient of a gas at a given pressure and temperature
                      looking at pre-tabulated line-by-line tables
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .lta extension)
            nwave :: Number of wavenumbers (cm-1)
            wave :: Wavenumber (cm-1)
            npoints :: Number of p-T levels at which the absorption coefficient must be computed
            press(npoints) :: Pressure (atm)
            temp(npoints) :: Temperature (K)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            wavelta :: Calculation wavenumbers (cm-1)
            k(nwave,npoints) :: Absorption coefficient (cm2)
        
        CALLING SEQUENCE:
        
            wavelta,k = calc_klbl(filename,nwave,wave,npoints,press,temp)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """
    
    #Reading the lbl-tables
    wavemin = wave.min()
    wavemax = wave.max()
    #wavemin = wave[0] - (wave[1]-wave[0])
    #wavemax = wave[nwave-1] + (wave[nwave-1]-wave[nwave-2])
    
    npress,ntemp,gasID,isoID,presslevels,templevels,nwavelta,wavelta,k = read_lbltable(filename,wavemin,wavemax)
    
    #Interpolating to the correct pressure and temperature
    ########################################################
    
    kgood = np.zeros([nwavelta,npoints])
    for ipoint in range(npoints):
        press1 = press[ipoint]
        temp1 = temp[ipoint]
        
        #Getting the levels just above and below the desired points
        lpress  = np.log(press1)
        press0,ip = find_nearest(presslevels,press1)

        if presslevels[ip]>=press1:
            iphi = ip
            if ip==0:
                ipl = 0
            else:
                ipl = ip - 1
        elif presslevels[ip]<press1:
            ipl = ip
            if ip==npress-1:
                iphi = npress - 1
            else:
                iphi = ip + 1
        
        temp0,it = find_nearest(templevels,temp1)

        if templevels[it]>=temp1:
            ithi = it
            if it==0:
                itl = 0
            else:
                itl = it - 1
        elif templevels[it]<temp1:
            itl = it
            if it==ntemp-1:
                ithi = ntemp - 1
            else:
                ithi = it + 1
    
        plo = np.log(presslevels[ipl])
        phi = np.log(presslevels[iphi])
        tlo = templevels[itl]
        thi = templevels[ithi]
        klo1 = np.zeros([nwavelta])
        klo2 = np.zeros([nwavelta])
        khi1 = np.zeros([nwavelta])
        khi2 = np.zeros([nwavelta])
        klo1[:] = k[:,ipl,itl]
        klo2[:] = k[:,ipl,ithi]
        khi1[:] = k[:,iphi,itl]
        khi2[:] = k[:,iphi,ithi]
        
        #Interpolating to get the absorption coefficient at desired p-T
        v = (lpress-plo)/(phi-plo)
        u = (temp1-tlo)/(thi-tlo)

        kgood[:,ipoint] = (1.0-v)*(1.0-u)*klo1[:] + v*(1.0-u)*khi1[:] + v*u*khi2[:] + (1.0-v)*u*klo2[:]

    if MakePlot==True:
        fig, ax = plt.subplots(1,1,figsize=(10,6))
    
        ax.semilogy(wavelta,klo1*1.0e-20,label='p = '+str(np.exp(plo))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wavelta,klo2*1.0e-20,label='p = '+str(np.exp(plo))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wavelta,khi1*1.0e-20,label='p = '+str(np.exp(phi))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wavelta,khi2*1.0e-20,label='p = '+str(np.exp(phi))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wavelta,kgood[:,ipoint]*1.0e-20,label='p = '+str(press1)+' atm - T = '+str(temp1)+' K',color='black')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()
    
    return wavelta,kgood


###############################################################################################

def calc_k(filename,wavemin,wavemax,npoints,press,temp,MakePlot=False):
    
    """
        
        FUNCTION NAME : calc_k()
        
        DESCRIPTION : Calculate the k coefficients of a gas at a given pressure and temperature
                      looking at pre-tabulated correlated-k tables
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .lta extension)
            nwave :: Number of wavenumbers (cm-1)
            wave :: Wavenumber (cm-1)
            npoints :: Number of p-T levels at which the absorption coefficient must be computed
            press(npoints) :: Pressure (atm)
            temp(npoints) :: Temperature (K)
        
        OPTIONAL INPUTS:
        
            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            wavek :: Calculation wavenumbers (cm-1)
            ng :: Number of g-ordinates
            g_ord :: G-ordinates
            del_g :: Interval between contiguous g-ordinates
            k(nwave,ng,npoints) :: K coefficients 
        
        CALLING SEQUENCE:
        
            wavekta,k = calc_k(filename,wavemin,wavemax,npoints,press,temp)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """

    gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g = read_ktable(filename,wavemin,wavemax)

    #Interpolating to the correct pressure and temperature
    ########################################################
    
    k_good = np.zeros([nwave,ng,npoints])
    for ipoint in range(npoints):
        press1 = press[ipoint]
        temp1 = temp[ipoint]

        #Getting the levels just above and below the desired points
        lpress  = np.log(press1)
        press0,ip = find_nearest(presslevels,press1)

        if presslevels[ip]>=press1:
            iphi = ip
            if ip==0:
                ipl = 0
            else:
                ipl = ip - 1
        elif presslevels[ip]<press1:
            ipl = ip
            if ip==npress-1:
                iphi = npress - 1
            else:
                iphi = ip + 1
        
        temp0,it = find_nearest(templevels,temp1)

        if templevels[it]>=temp1:
            ithi = it
            if it==0:
                itl = 0
            else:
                itl = it - 1
        elif templevels[it]<temp1:
            itl = it
            if it==ntemp-1:
                ithi = ntemp - 1
            else:
                ithi = it + 1
    
        plo = np.log(presslevels[ipl])
        phi = np.log(presslevels[iphi])
        tlo = templevels[itl]
        thi = templevels[ithi]
        klo1 = np.zeros([nwave,ng])
        klo2 = np.zeros([nwave,ng])
        khi1 = np.zeros([nwave,ng])
        khi2 = np.zeros([nwave,ng])
        klo1[:] = k_g[:,:,ipl,itl]
        klo2[:] = k_g[:,:,ipl,ithi]
        khi2[:] = k_g[:,:,iphi,ithi]
        khi1[:] = k_g[:,:,iphi,itl]

        #Interpolating to get the k-coefficients at desired p-T
        if ipl==iphi:
            v = 0.5
        else:
            v = (lpress-plo)/(phi-plo)
        
        if itl==ithi:
            u = 0.5
        else:
            u = (temp1-tlo)/(thi-tlo)

        k_good[:,:,ipoint] = (1.0-v)*(1.0-u)*klo1[:,:] + v*(1.0-u)*khi1[:,:] + v*u*khi2[:,:] + (1.0-v)*u*klo2[:,:]
    
    if MakePlot==True:
        fig, ax = plt.subplots(1,1,figsize=(10,6))
    
        k_abs = np.matmul(k_good[:,:,npoints-1], del_g)
        k_abslo1 = np.matmul(klo1[:,:], del_g)
        k_abslo2 = np.matmul(klo2[:,:], del_g)
        k_abshi1 = np.matmul(khi1[:,:], del_g)
        k_abshi2 = np.matmul(khi2[:,:], del_g)
        ax.semilogy(wave,k_abslo1,label='p = '+str(np.exp(plo))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wave,k_abslo2,label='p = '+str(np.exp(plo))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wave,k_abshi1,label='p = '+str(np.exp(phi))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wave,k_abshi2,label='p = '+str(np.exp(phi))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wave,k_abs,label='p = '+str(press1)+' atm - T = '+str(temp1)+' K',color='black')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()

    return wave,ng,g_ord,del_g,k_good

