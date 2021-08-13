from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from numba import jit

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Spectroscopy_0:

    def __init__(self, ISPACE=0, ILBL=2, NGAS=2, ID=[1,2], ISO=[0,0], LOCATION=['',''], NWAVE=2, WAVE=[0.,100.], \
                 NP=2, NT=2, PRESS=[1.0e2,1.0e-10], TEMP=[30.,300.], NG=1, G_ORD=[0.], DELG=[1.], FWHM=0.0):

        """
        Inputs
        ------
        @param ISPACE: int,
            Flag indicating the units of the spectral coordinate (0) Wavenumber cm-1 (1) Wavelength um
        @param ILBL: int,
            Flag indicating if the calculations are performed using pre-tabulated 
            correlated-K tables (0) or line-by-line tables (2)
        @param NGAS: int,
            Number of active gases to include in the atmosphere    
        @param ID: 1D array,
            Gas ID for each active gas         
        @param ISO: 1D array,
            Isotope ID for each gas, default 0 for all isotopes in terrestrial relative abundance     
        @param LOCATION: 1D array,
            List of strings indicating where the .lta or .kta tables are stored for each of the gases
        @param NWAVE: int,
            Number of wavelengths included in the K-tables or LBL-tables
        @param WAVE: 1D array,
            Wavelengths at which the K-tables or LBL-tables are defined
        @param NP: int,
            Number of pressure levels at which the K-tables or LBL-tables were computed
        @param NT: int,
            Number of temperature levels at which the K-tables or LBL-tables were computed
        @param PRESS: 1D array
            Pressure levels at which the K-tables or LBL-tables were computed (Pa)
        @param TEMP: 1D array
            Temperature levels at which the K-tables or LBL-tables were computed (K)
        @param NG: int,
            Number of g-ordinates included in the k-tables (NG=1 for line-by-line)
        @param G_ORD: 1D array,
            G-ordinates
        @param DELG: 1D array,
            Intervals of g-ordinates
        @param FWHM: real,
            Full-width at half maximum (only in case of K-tables)
        

        Methods
        -------
        Spectroscopy_0.edit_K
        Spectroscopy_0.read_lls
        Spectroscopy_0.read_kls
        Spectroscopy_0.read_tables
        """

        #Input parameters
        self.ISPACE = ISPACE
        self.ILBL = ILBL
        self.NGAS = NGAS
        self.ID = ID
        self.ISO = ISO
        self.LOCATION = LOCATION
        self.NWAVE = NWAVE
        self.WAVE = WAVE
        self.NP = NP
        self.NT = NT
        self.PRESS = PRESS
        self.TEMP = TEMP
        self.NG = NG
        self.G_ORD = G_ORD
        self.DELG = DELG
        self.FWHM = FWHM

        self.K = None #(NWAVE,NG,NP,NT,NGAS)

    def edit_K(self, K_array):
        """
        Edit the k-coefficients (ILBL=0) or absorption cross sections (ILBL=2)
        @param K_array: 5D array (NWAVE,NG,NP,NT,NGAS) or 4D array (NWAVE,NP,NT,NGAS)
            K-coefficients or absorption cross sections
        """
        K_array = np.array(K_array)
        
        if self.ILBL==0: #K-tables
            assert K_array.shape == (self.NWAVE, self.NG, self.NP, self.NT, self.NGAS),\
                'K should be (NWAVE,NG,NP,NT,NGAS) if ILBL=0 (K-tables)'
        elif self.ILBL==2: #LBL-tables
            assert K_array.shape == (self.NWAVE, self.NP, self.NT, self.NGAS),\
                'K should be (NWAVE,NP,NT,NGAS) if ILBL=2 (LBL-tables)'
        else:
            sys.exit('ILBL needs to be either 0 (K-tables) or 2 (LBL-tables)')

        self.K = K_array


    def read_lls(self, runname):
        """
        Read the .lls file and store the parameters into the Spectroscopy Class

        @param runname: str
            Name of the Nemesis run
        """

        ngasact = len(open(runname+'.lls').readlines(  ))
    
        #Opening .lls file 
        f = open(runname+'.lls','r')
        strlta = [''] * ngasact
        for i in range(ngasact):
            s = f.readline().split()
            strlta[i] = s[0]

        self.NGAS = ngasact
        self.LOCATION = strlta
    
        #Now reading the head of the binary files included in the .lls file
        nwavelta = np.zeros([ngasact],dtype='int')
        npresslta = np.zeros([ngasact],dtype='int')
        ntemplta = np.zeros([ngasact],dtype='int')
        gasIDlta = np.zeros([ngasact],dtype='int')
        isoIDlta = np.zeros([ngasact],dtype='int')
        for i in range(ngasact):
            nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels = read_ltahead(strlta[i])
            nwavelta[i] = nwave
            npresslta[i] = npress
            ntemplta[i] = ntemp
            gasIDlta[i] = gasID
            isoIDlta[i] = isoID

        if len(np.unique(nwavelta)) != 1:
            sys.exit('error :: Number of wavenumbers in all .lta files must be the same')
        if len(np.unique(npresslta)) != 1:
            sys.exit('error :: Number of pressure levels in all .lta files must be the same')
        if len(np.unique(ntemplta)) != 1:
            sys.exit('error :: Number of temperature levels in all .lta files must be the same')

        self.ID = gasIDlta
        self.ISO = isoIDlta
        self.NP = npress
        self.NG = 1
        self.NT = ntemp
        self.PRESS = presslevels
        self.TEMP = templevels
        self.NWAVE = nwave

        vmax = vmin + delv * (nwave-1)
        wavelta = np.linspace(vmin,vmax,nwave)
        #wavelta = np.round(wavelta,5)
        self.WAVE = wavelta


    def read_kls(self, runname):
        """
        Read the .kls file and store the parameters into the Spectroscopy Class

        @param runname: str
            Name of the Nemesis run
        """
        
        from NemesisPy import read_ktahead

        ngasact = len(open(runname+'.kls').readlines(  ))
    
        #Opening file
        f = open(runname+'.kls','r')
        strkta = [''] * ngasact
        for i in range(ngasact):
            s = f.readline().split()
            strkta[i] = s[0]

        self.NGAS = ngasact
        self.LOCATION = strkta

        #Now reading the head of the binary files included in the .lls file
        nwavekta = np.zeros([ngasact],dtype='int')
        npresskta = np.zeros([ngasact],dtype='int')
        ntempkta = np.zeros([ngasact],dtype='int')
        ngkta = np.zeros([ngasact],dtype='int')
        gasIDkta = np.zeros([ngasact],dtype='int')
        isoIDkta = np.zeros([ngasact],dtype='int')
        for i in range(ngasact):
            nwave,wavekta,fwhmk,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels = read_ktahead(strkta[i])
            nwavekta[i] = nwave
            npresskta[i] = npress
            ntempkta[i] = ntemp
            ngkta[i] = ng
            gasIDkta[i] = gasID
            isoIDkta[i] = isoID

        if len(np.unique(nwavekta)) != 1:
            sys.exit('error :: Number of wavenumbers in all .kta files must be the same')
        if len(np.unique(npresskta)) != 1:
            sys.exit('error :: Number of pressure levels in all .kta files must be the same')
        if len(np.unique(ntempkta)) != 1:
            sys.exit('error :: Number of temperature levels in all .kta files must be the same')
        if len(np.unique(ngkta)) != 1:
            sys.exit('error :: Number of g-ordinates in all .kta files must be the same')

        self.ID = gasIDkta
        self.ISO = isoIDkta
        self.NP = npress
        self.NT = ntemp
        self.PRESS = presslevels
        self.TEMP = templevels
        self.NWAVE = nwave
        self.NG = ng
        self.DELG = del_g
        self.G_ORD = g_ord
        self.FWHM = fwhmk
        self.WAVE = wavekta

    def read_tables(self, wavemin=0., wavemax=1.0e10):
        """
        Reads the .kta or .lta tables and stores the results into this class

        Optional parameters
        -----------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um) 
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)
        """

        iwave1 = np.where( (self.WAVE>=wavemin) & (self.WAVE<=wavemax) )
        iwave = iwave1[0]
        self.NWAVE = len(iwave)
        self.WAVE = self.WAVE[iwave1]

        if self.ILBL==0: #K-tables

            from NemesisPy import read_ktable

            kstore = np.zeros([self.NWAVE,self.NG,self.NP,self.NT,self.NGAS])
            for igas in range(self.NGAS):
                gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g = read_ktable(self.LOCATION[igas],wavemin,wavemax)
                kstore[:,:,:,:,igas] = k_g[:,:,:,:]
            self.edit_K(kstore)


        elif self.ILBL==2: #LBL-tables

            from NemesisPy import read_lbltable

            kstore = np.zeros([self.NWAVE,self.NP,self.NT,self.NGAS])
            for igas in range(self.NGAS):
                npress,ntemp,gasID,isoID,presslevels,templevels,nwave,wave,k = read_lbltable(self.LOCATION[igas],wavemin,wavemax)
                kstore[:,:,:,igas] = k[:,:,:]
            self.edit_K(kstore)

        else:
            sys.exit('error in Spectroscopy :: ILBL must be either 0 (K-tables) or 2 (LBL-tables)')


    def calc_klbl(self,npoints,press,temp,wavemin=0.,wavemax=1.0e10,MakePlot=False):
        """
        Calculate the absorption coefficient at a given pressure and temperature
        looking at pre-tabulated line-by-line tables (assumed to be already stored in this class)

        Input parameters
        -------------------
        @param npoints: int
            Number of p-T points at which to calculate the cross sections
        @param press: 1D array
            Pressure levels (atm)
        @param temp: 1D array
            Temperature levels (K)

        Optional parameters 
        ---------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um)
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)
        """

        from NemesisPy import find_nearest

        #Interpolating to the correct pressure and temperature
        ########################################################
    
        #K (NWAVE,NP,NT,NGAS)

        kgood = np.zeros([self.NWAVE,npoints,self.NGAS])
        for ipoint in range(npoints):

            press1 = press[ipoint]
            temp1 = temp[ipoint]
        
            #Getting the levels just above and below the desired points
            lpress  = np.log(press1)
            press0,ip = find_nearest(self.PRESS,press1)

            if self.PRESS[ip]>=press1:
                iphi = ip
                if ip==0:
                    ipl = 0
                else:
                    ipl = ip - 1
            elif self.PRESS[ip]<press1:
                ipl = ip
                if ip==self.NP-1:
                    iphi = self.NP - 1
                else:
                    iphi = ip + 1

            temp0,it = find_nearest(self.TEMP,temp1)

            if self.TEMP[it]>=temp1:
                ithi = it
                if it==0:
                    itl = 0
                else:
                    itl = it - 1
            elif self.TEMP[it]<temp1:
                itl = it
                if it==self.NT-1:
                    ithi = self.NT - 1
                else:
                    ithi = it + 1

            plo = np.log(self.PRESS[ipl])
            phi = np.log(self.PRESS[iphi])
            tlo = self.TEMP[itl]
            thi = self.TEMP[ithi]
            klo1 = np.zeros([self.NWAVE,self.NGAS])
            klo2 = np.zeros([self.NWAVE,self.NGAS])
            khi1 = np.zeros([self.NWAVE,self.NGAS])
            khi2 = np.zeros([self.NWAVE,self.NGAS])
            klo1[:] = self.K[:,ipl,itl,:]
            klo2[:] = self.K[:,ipl,ithi,:]
            khi1[:] = self.K[:,iphi,itl,:]
            khi2[:] = self.K[:,iphi,ithi,:]

            #Interpolating to get the absorption coefficient at desired p-T
            v = (lpress-plo)/(phi-plo)
            u = (temp1-tlo)/(thi-tlo)

            if(thi==tlo):
                u = 0.0
            if(phi==plo):
                v = 0.0

            for igas in range(self.NGAS):
                igood = np.where((klo1[:,igas]>0.0) & (klo2[:,igas]>0.0) & (khi1[:,igas]>0.0) & (khi2[:,igas]>0.0))
                igood = igood[0]
                kgood[igood,ipoint,igas] = (1.0-v)*(1.0-u)*np.log(klo1[igood,igas]) + v*(1.0-u)*np.log(khi1[igood,igas]) + v*u*np.log(khi2[igood,igas]) + (1.0-v)*u*np.log(klo2[igood,igas])
                kgood[igood,ipoint,igas] = np.exp(kgood[igood,ipoint,igas])

        return kgood


    def calc_k(self,npoints,press,temp,WAVECALC=[12345678.],MakePlot=False):
        """
        Calculate the k-coefficients at a given pressure and temperature
        looking at pre-tabulated k-tables (assumed to be already stored in this class)

        Input parameters
        -------------------
        @param npoints: int
            Number of p-T points at which to calculate the cross sections
        @param press: 1D array
            Pressure levels (atm)
        @param temp: 1D array
            Temperature levels (K)

        Optional parameters 
        ---------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um)
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)
        """

        from NemesisPy.Utils import find_nearest
        from scipy import interpolate

        #Interpolating the k-coefficients to the correct pressure and temperature
        #############################################################################

        #K (NWAVE,NG,NP,NT,NGAS)

        kgood = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS])
        for ipoint in range(npoints):
            press1 = press[ipoint]
            temp1 = temp[ipoint]

            #Getting the levels just above and below the desired points
            lpress  = np.log(press1)
            press0,ip = find_nearest(self.PRESS,press1)

            if self.PRESS[ip]>=press1:
                iphi = ip
                if ip==0:
                    ipl = 0
                else:
                    ipl = ip - 1
            elif self.PRESS[ip]<press1:
                ipl = ip
                if ip==self.NP-1:
                    iphi = self.NP - 1
                else:
                    iphi = ip + 1

            temp0,it = find_nearest(self.TEMP,temp1)

            if self.TEMP[it]>=temp1:
                ithi = it
                if it==0:
                    itl = 0
                else:
                    itl = it - 1
            elif self.TEMP[it]<temp1:
                itl = it
                if it==self.NT-1:
                    ithi = self.NT - 1
                else:
                    ithi = it + 1

            plo = np.log(self.PRESS[ipl])
            phi = np.log(self.PRESS[iphi])
            tlo = self.TEMP[itl]
            thi = self.TEMP[ithi]
            klo1 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            klo2 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            khi1 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            khi2 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            klo1[:] = self.K[:,:,ipl,itl,:]
            klo2[:] = self.K[:,:,ipl,ithi,:]
            khi2[:] = self.K[:,:,iphi,ithi,:]
            khi1[:] = self.K[:,:,iphi,itl,:]

            #Interpolating to get the k-coefficients at desired p-T
            if ipl==iphi:
                v = 0.5
            else:
                v = (lpress-plo)/(phi-plo)
        
            if itl==ithi:
                u = 0.5
            else:
                u = (temp1-tlo)/(thi-tlo)

            igood = np.where( (klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0) )
            kgood[igood[0],igood[1],ipoint,igood[2]] = np.exp((1.0-v)*(1.0-u)*np.log(klo1[igood[0],igood[1],igood[2]]) + v*(1.0-u)*np.log(khi1[igood[0],igood[1],igood[2]]) + v*u*np.log(khi2[igood[0],igood[1],igood[2]]) + (1.0-v)*u*np.log(klo2[igood[0],igood[1],igood[2]]))


        #Checking that the calculation wavenumbers coincide with the wavenumbers in the k-tables
        ##########################################################################################

        if WAVECALC[0]!=12345678.:

            NWAVEC = len(WAVECALC)
            kret = np.zeros([NWAVEC,self.NG,npoints,self.NGAS])

            #Checking if k-tables are defined in irregularly spaced wavenumber grid
            delv = 0.0
            Irr = 0
            for iv in range(self.NWAVE-1):
                delv1 = self.WAVE[iv+1] - self.WAVE[iv]
                if iv==0:
                    delv = delv1
                    pass

                if abs((delv1-delv)/(delv))>0.001:
                    Irr = 1
                    break
                else:
                    delv = delv1
                    continue

            #If they are defined in a regular grid, we interpolate to the nearest value
            if Irr==0:
                for i in range(npoints):
                    for j in range(self.NGAS):
                        for k in range(self.NG):
                            f = interpolate.interp1d(self.WAVE,kgood[:,k,i,j])
                            kret[:,k,i,j] = f(WAVECALC)
            else:
                for i in range(NWAVEC):
                    wave0,iv = find_nearest(self.WAVE,WAVECALC[i])
                    kret[i,:,:,:] = kgood[iv,:,:,:]
        
        else:

            kret = kgood

        return kret


###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

Other functions interacting with the Spectroscopy class
"""


def read_ltahead(filename):
    """
    Read the header information in a line-by-line look-up table 
    written with the standard format of Nemesis

    @param filename: str
        Name of the .lta file
    """

    #Opening file
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'lta':
        f = open(filename,'r')
    else:
        f = open(filename+'.lta','r')
    
    irec0 = int(np.fromfile(f,dtype='int32',count=1))
    nwave = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    gasID = int(np.fromfile(f,dtype='int32',count=1))
    isoID = int(np.fromfile(f,dtype='int32',count=1))

    presslevels = np.fromfile(f,dtype='float32',count=npress)
    templevels = np.fromfile(f,dtype='float32',count=ntemp)

    return nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels


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
            wave :: Wavelength (um) / Wavenumber (cm-1) array
            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: RADTRAN gas ID
            isoID :: RADTRAN isotopologue ID
            pressleves(np) :: Pressure levels (atm)
            templeves(np) :: Temperature levels (K)
        
        CALLING SEQUENCE:
        
            nwave,wave,fwhm,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels = read_ktahead(filename)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    #Opening file
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'kta':
        f = open(filename,'r')
    else:
        f = open(filename+'.kta','r')

    irec0 = int(np.fromfile(f,dtype='int32',count=1))
    nwave = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    fwhm = float(np.fromfile(f,dtype='float32',count=1))
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    ng = int(np.fromfile(f,dtype='int32',count=1))
    gasID = int(np.fromfile(f,dtype='int32',count=1))
    isoID = int(np.fromfile(f,dtype='int32',count=1))

    g_ord = np.fromfile(f,dtype='float32',count=ng)
    del_g = np.fromfile(f,dtype='float32',count=ng)

    dummy = np.fromfile(f,dtype='float32',count=1)
    dummy = np.fromfile(f,dtype='float32',count=1)

    presslevels = np.fromfile(f,dtype='float32',count=npress)

    N1 = abs(ntemp)
    if ntemp < 0:
        templevels = np.zeros([npress,n1])
        for i in range(npress):
            for j in range(n1):
                templevels[i,j] =  np.fromfile(f,dtype='float32',count=1)
    else:
        templevels = np.fromfile(f,dtype='float32',count=ntemp)

    #Reading central wavelengths in non-uniform grid
    if delv>0.0:
        vmax = delv*(nwave-1) + vmin
        wavetot = np.linspace(vmin,vmax,nwave)
    else:
        wavetot = np.zeros(nwave)
        wavetot[:] = np.fromfile(f,dtype='float32',count=nwave)
    
    return nwave,wavetot,fwhm,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels


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
    nwavelta = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    gasID = int(np.fromfile(f,dtype='int32',count=1))
    isoID = int(np.fromfile(f,dtype='int32',count=1))

    presslevels = np.fromfile(f,dtype='float32',count=npress)
    templevels = np.fromfile(f,dtype='float32',count=ntemp)

    ioff = 8*nbytes_int32+npress*nbytes_float32+ntemp*nbytes_float32
    
    #Calculating the wavenumbers to be read
    vmax = vmin + delv * (nwavelta-1)
    wavelta = np.linspace(vmin,vmax,nwavelta)
    #wavelta = np.round(wavelta,5)
    ins1 = np.where( (wavelta>=wavemin) & (wavelta<=wavemax) )
    ins = ins1[0]
    nwave = len(ins)
    wave = np.zeros(nwave)
    wave[:] = wavelta[ins]
    
    #Reading the absorption coefficients
    #######################################
    
    k = np.zeros([nwave,npress,ntemp])
    
    #Jumping until we get to the minimum wavenumber
    njump = npress*ntemp*(ins[0])
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

    g_ord = np.zeros(ng)
    del_g = np.zeros(ng)
    templevels = np.zeros(ntemp)
    presslevels = np.zeros(npress)
    g_ord[:] = np.fromfile(f,dtype='float32',count=ng)
    del_g[:] = np.fromfile(f,dtype='float32',count=ng)
    
    ioff = ioff + 2*ng*nbytes_float32

    dummy = np.fromfile(f,dtype='float32',count=1)
    dummy = np.fromfile(f,dtype='float32',count=1)

    ioff = ioff + 2*nbytes_float32

    presslevels[:] = np.fromfile(f,dtype='float32',count=npress)
    templevels[:] = np.fromfile(f,dtype='float32',count=ntemp)
    
    ioff = ioff + npress*nbytes_float32+ntemp*nbytes_float32

    #Reading central wavelengths in non-uniform grid
    if delv>0.0:
        vmax = delv*(nwavekta-1) + vmin
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

######################################################################################################

def write_lbltable(filename,npress,ntemp,gasID,isoID,presslevels,templevels,nwave,vmin,delv,k,DOUBLE=False):
    
    """
        FUNCTION NAME : write_lbltable()
        
        DESCRIPTION : Read a .lta file (binary file) with the information about the absorption cross-section
                      of a given gas at different pressure and temperature levels
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .kta extension)
        
        OPTIONAL INPUTS: 

            DOUBLE :: If True, the parameters are written with double precision (double) rather than single (float)
        
        OUTPUTS :
        
            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: NEMESIS gas ID (see manual)
            isoID :: NEMESIS isotopologue ID (0 for all isotopes)
            presslevels(npress) :: Pressure levels (atm)
            templevels(ntemp) :: Temperature levels (K)
            nwave :: Number of spectral points in lbl-table
            vmin :: Minimum wavelength/wavenumber (um/cm-1)
            delv :: Wavelength/wavenumber step (um/cm-1)
            k(nwave,npress,ntemp) :: Absorption cross-section (cm2)
        
        CALLING SEQUENCE:
        
            write_lbltable(filename,npress,ntemp,gasID,isoID,presslevels,templevels,nwave,vmin,delv,k)
        
        MODIFICATION HISTORY : Juan Alday (06/08/2021)
        
    """

    import struct

    #Opening file
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'lta':
        f = open(filename,'w+b')
    else:
        f = open(filename+'.lta','w+b')

    irec0 = 8 + npress + ntemp
    bin=struct.pack('i',irec0) #IREC0
    f.write(bin)

    bin=struct.pack('i',nwave) #NWAVE
    f.write(bin)

    if DOUBLE==True:
        df = 'd'
    else:
        df = 'f'

    bin=struct.pack(df,vmin) #VMIN
    f.write(bin)

    bin=struct.pack(df,delv) #DELV
    f.write(bin)

    bin=struct.pack('i',npress) #NPRESS
    f.write(bin)

    bin=struct.pack('i',ntemp) #NTEMP
    f.write(bin)

    bin=struct.pack('i',gasID) #GASID
    f.write(bin)

    bin=struct.pack('i',isoID) #ISOID
    f.write(bin)

    myfmt=df*len(presslevels)
    bin=struct.pack(myfmt,*presslevels) #PRESSLEVELS
    f.write(bin)

    myfmt=df*len(templevels)
    bin=struct.pack(myfmt,*templevels) #TEMPLEVELS
    f.write(bin)

    for i in range(nwave):
        for j in range(npress):
            tmp = k[i,j,:] * 1.0e20
            myfmt=df*len(tmp)
            bin=struct.pack(myfmt,*tmp) #K
            f.write(bin)

    f.close()

