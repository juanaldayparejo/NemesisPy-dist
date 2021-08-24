from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import os,sys
from numba import jit

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Measurement_0:

    def __init__(self, NGEOM=1, FWHM=0.0, ISHAPE=2, IFORM=0, ISPACE=0, LATITUDE=0.0, LONGITUDE=0.0, NCONV=[1], NAV=[1]):

        """
        Inputs
        ------
        @param NGEOM: int,
            Number of observing geometries
        @param FWHM: real,
            Full-width at half-maximum of the instrument  
        @param ISHAPE: int,
            Instrument lineshape.
            (0) Square lineshape
            (1) Triangular
            (2) Gaussian
            (3) Hamming
            (4) Hanning      
        @param ISPACE: int,
            Spectral units 
            (0) Wavenumber (cm-1)
            (1) Wavelength (um)
        @param IFORM: int,
            Units of the spectra
            (0) Radiance - W cm-2 sr-1 (cm-1)-1 if ISPACE=0 ---- W cm-2 sr-1 μm-1 if ISPACE=1
            (1) F_planet/F_star - Dimensionsless
            (2) A_planet/A_star - 100.0 * A_planet/A_star (dimensionsless)
            (3) Integrated spectral power of planet - W (cm-1)-1 if ISPACE=0 ---- W um-1 if ISPACE=1
            (4) Atmospheric transmission multiplied by solar flux
        @param LATITUDE: int,
            Planetocentric latitude at centre of the field of view  
        @param LONGITUDE: int,
            Planetocentric longitude at centre of the field of view     
        @param NCONV: 1D array
            Number of convolution spectral points in each spectrum     
        @attribute NAV: 1D array
            For each geometry, number of individual geometries need to be calculated
            and averaged to reconstruct the field of view 
        
        Attributes
        ----------
        @attribute VCONV: 2D array
            Convolution spectral points (wavelengths/wavenumbers) in each spectrum
        @attribute MEAS: 2D array
            Measured spectrum for each geometry
        @attribute ERRMEAS: 2D array
            Noise in the measured spectrum for each geometry        
        @attribute FLAT: 2D array
            Latitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute FLON: 2D array
            Longitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute SOL_ANG: 2D array
            Solar indicent angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute EMISS_ANG: 2D array
            Emission angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute AZI_ANG: 2D array
            Azimuth angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
        @attribute TANHE: 2D array
            Tangent height of each averaging point needed to reconstruct the FOV (when NAV > 1) 
            (For limb or solar occultation observations)   
        @attribute WGEOM: 2D array
            Weights of each point for the averaging of the FOV (when NAV > 1)
        @attribute NWAVE: int
            Number of calculation wavelengths required to model the convolution wavelengths 
        @attribute VWAVE: int
            Calculation wavenumbers for one particular geometry
        @attribute NFIL: 1D array
            If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
            NFIL represents the number of spectral points to defined the ILS for each convolution wavenumber.
        @attribute VFIL: 2D array
            If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
            VFIL represents the calculation wavenumbers at which the ILS is defined for each each convolution wavenumber.
        @attribute AFIL: 2D array
            If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
            AFIL represents the value of the ILS at each VFIL for each convolution wavenumber.

        Methods
        -------
        Measurement.edit_VCONV
        Measurement.edit_MEAS
        Measurement.edit_ERRMEAS
        Measurement.edit_FLAT
        Measurement.edit_FLON
        Measurement.edit_SOL_ANG
        Measurement.edit_EMISS_ANG
        Measurement.edit_AZI_ANG
        Measurement.edit_WGEOM
        Measurement.read_spx_SO
        Measurement.read_spx
        Measurement.read_sha
        Measurement.read_fil
        Measurement.wavesetc     
        Measurement.wavesetb
        """

        #Input parameters
        self.NGEOM = NGEOM
        self.FWHM = FWHM
        self.ISHAPE = ISHAPE
        self.LATITUDE = LATITUDE        
        self.LONGITUDE = LONGITUDE
        self.NAV = NAV       #np.zeros(NGEOM)
        self.NCONV = NCONV   #np.zeros(NGEOM)

        # Input the following profiles using the edit_ methods.
        self.VCONV = None # np.zeros(NCONV,NGEOM)
        self.MEAS =  None # np.zeros(NCONV,NGEOM)
        self.ERRMEAS = None # np.zeros(NCONV,NGEOM)
        self.FLAT = None # np.zeros(NGEOM,NAV)
        self.FLON = None # np.zeros(NGEOM,NAV)
        self.SOL_ANG = None # np.zeros(NGEOM,NAV)
        self.EMISS_ANG = None # np.zeros(NGEOM,NAV)
        self.AZI_ANG = None # np.zeros(NGEOM,NAV)
        self.TANHE = None # np.zeros(NGEOM,NAV)
        self.WGEOM = None # np.zeros(NGEOM,NAV)
        self.NY = None #np.sum(NCONV)
        self.Y = None #np.zeros(NY)
        self.SE = None #np.zeros(NY,NY)

    def edit_VCONV(self, VCONV_array):
        """
        Edit the convolution wavelengths/wavenumbers array in each geometry
        @param VCONV_array: 2D array
            Convolution wavelengths/wavenumbers in each geometry
        """
        VCONV_array = np.array(VCONV_array)
        try:
            assert VCONV_array.shape == (self.NCONV.max(), self.NGEOM),\
                'VCONV should be NCONV by NGEOM.'
        except:
            assert VCONV_array.shape == (self.NCONV[0]) and self.NGEOM==1,\
                'VCONV should be NCONV.'

        self.VCONV = VCONV_array

    def edit_MEAS(self, MEAS_array):
        """
        Edit the measured spectrum in each geometry in each geometry
        @param MEAS_array: 2D array
            Measured spectrum in each geometry
        """
        MEAS_array = np.array(MEAS_array)
        try:
            assert MEAS_array.shape == (self.NCONV.max(), self.NGEOM),\
                'MEAS should be NCONV by NGEOM.'
        except:
            assert MEAS_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'MEAS should be NCONV.'

        self.MEAS = MEAS_array

    def edit_ERRMEAS(self, ERRMEAS_array):
        """
        Edit the measured uncertainty of the spectrum in each geometry
        @param ERRMEAS_array: 2D array
            Measured uncertainty of the spectrum in each geometry
        """
        ERRMEAS_array = np.array(ERRMEAS_array)
        try:
            assert ERRMEAS_array.shape == (self.NCONV.max(), self.NGEOM),\
                'ERRMEAS should be NCONV by NGEOM.'
        except:
            assert ERRMEAS_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'ERRMEAS should be NCONV.'

        self.ERRMEAS = ERRMEAS_array

    def edit_FLAT(self, FLAT_array):
        """
        Edit the latitude of each averaging point needed to 
            reconstruct the FOV (when NAV > 1)
        @param FLAT_array: 2D array
            Latitude of each averaging point needed to reconstruct 
            the FOV (when NAV > 1)
        """
        FLAT_array = np.array(FLAT_array)
        try:
            assert FLAT_array.shape == (self.NGEOM, self.NAV.max()),\
                'FLAT should be NGEOM by NAV.'
        except:
            assert FLAT_array.shape == (self.NGEOM,self.NAV) and self.NGEOM==1,\
                'FLAT should be NAV.'

        self.FLAT = FLAT_array

    def edit_FLON(self, FLON_array):
        """
        Edit the longitude of each averaging point needed to 
            reconstruct the FOV (when NAV > 1)
        @param FLON_array: 2D array
            Longitude of each averaging point needed to reconstruct 
            the FOV (when NAV > 1)
        """
        FLON_array = np.array(FLON_array)

        assert FLON_array.shape == (self.NGEOM, self.NAV.max()),\
            'FLON should be NGEOM by NAV.'

        self.FLON = FLON_array

    def edit_SOL_ANG(self, SOL_ANG_array):
        """
        Edit the solar indicent angle of each averaging point 
            needed to reconstruct the FOV (when NAV > 1)
        @param SOL_ANG_array: 2D array
            Solar indicent angle of each averaging point needed 
            to reconstruct the FOV (when NAV > 1)
        """
        SOL_ANG_array = np.array(SOL_ANG_array)
        
        assert SOL_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'SOL_ANG should be NGEOM by NAV.'

        self.SOL_ANG = SOL_ANG_array

    def edit_EMISS_ANG(self, EMISS_ANG_array):
        """
        Edit the emission angle of each averaging point 
            needed to reconstruct the FOV (when NAV > 1)
        @param EMISS_ANG_array: 2D array
            Emission angle of each averaging point needed 
            to reconstruct the FOV (when NAV > 1)
        """
        EMISS_ANG_array = np.array(EMISS_ANG_array)
        
        assert EMISS_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'EMISS_ANG should be NGEOM by NAV.'

        self.EMISS_ANG = EMISS_ANG_array

    def edit_AZI_ANG(self, AZI_ANG_array):
        """
        Edit the azimuth angle of each averaging point 
            needed to reconstruct the FOV (when NAV > 1)
        @param AZI_ANG_array: 2D array
            Azimuth angle of each averaging point needed 
            to reconstruct the FOV (when NAV > 1)
        """
        AZI_ANG_array = np.array(AZI_ANG_array)
        
        assert AZI_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'AZI_ANG should be NGEOM by NAV.'

        self.AZI_ANG = AZI_ANG_array

    def edit_TANHE(self, TANHE_array):
        """
        Edit the tangent height of each averaging point 
            needed to reconstruct the FOV (when NAV > 1)
            for limb or solar occultation observations
        @param AZI_ANG_array: 2D array
            Tangent height of each averaging point needed 
            to reconstruct the FOV (when NAV > 1) for
            limb or solar occultation observations
        """
        TANHE_array = np.array(TANHE_array)
        
        assert TANHE_array.shape == (self.NGEOM, self.NAV.max()),\
            'TANHE should be NGEOM by NAV.'

        self.TANHE = TANHE_array

    def edit_WGEOM(self, WGEOM_array):
        """
        Edit the weights of each point for the averaging 
            of the FOV (when NAV > 1)
        @param AZI_ANG_array: 2D array
            Weights of each point for the averaging of 
            the FOV (when NAV > 1)
        """
        WGEOM_array = np.array(WGEOM_array)
        
        assert WGEOM_array.shape == (self.NGEOM, self.NAV.max()),\
            'WGEOM should be NGEOM by NAV.'

        self.WGEOM = WGEOM_array

    def calc_MeasurementVector(self):
        """
        Calculate the measurement vector based on the other parameters
        defined in this class
        """

        self.NY = np.sum(self.NCONV)
        y1 = np.zeros(self.NY)
        se1 = np.zeros(self.NY)
        ix = 0
        for i in range(self.NGEOM):
            y1[ix:ix+self.NCONV[i]] = self.MEAS[0:self.NCONV[i],i]
            se1[ix:ix+self.NCONV[i]] = self.ERRMEAS[0:self.NCONV[i],i]
            ix = ix + self.NCONV[i]

        self.Y = y1
        se = np.zeros([self.NY,self.NY])
        for i in range(self.NY):
            se[i,i] = se1[i]**2.

        self.SE = se


    def read_spx_SO(self,runname,MakePlot=False):
    
        """
        Fill the attribute and parameters of the Measurement class for a retrieval
        of solar occultation or limb observations

        @param Runname: string
            Name of the Nemesis run 
        """

        #Opening file
        f = open(runname+'.spx','r')
    
        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=4,dtype='float')
        inst_fwhm = float(tmp[0])
        xlat = float(tmp[1])
        xlon = float(tmp[2])
        ngeom = int(tmp[3])
    
        #Defining variables
        nav = 1 #it needs to be generalized to read more than one NAV per observation geometry
        nconv = np.zeros([ngeom],dtype='int')
        flat = np.zeros([ngeom,nav])
        flon = np.zeros([ngeom,nav])
        tanhe = np.zeros([ngeom,nav])
        wgeom = np.zeros([ngeom,nav])
        nconvmax = 20000
        wavetmp = np.zeros([nconvmax,ngeom])
        meastmp = np.zeros([nconvmax,ngeom])
        errmeastmp = np.zeros([nconvmax,ngeom])
        for i in range(ngeom):
            nconv[i] = int(f.readline().strip())
            for j in range(nav):
                navsel = int(f.readline().strip())
                tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
                flat[i,j] = float(tmp[0])
                flon[i,j] = float(tmp[1])
                tanhe[i,j] = float(tmp[2])
                wgeom[i,j] = float(tmp[5])
            for iconv in range(nconv[i]):
                tmp = np.fromfile(f,sep=' ',count=3,dtype='float')
                wavetmp[iconv,i] = float(tmp[0])
                meastmp[iconv,i] = float(tmp[1])
                errmeastmp[iconv,i] = float(tmp[2])


        #Making final arrays for the measured spectra
        nconvmax2 = max(nconv)
        wave = np.zeros([nconvmax2,ngeom])
        meas = np.zeros([nconvmax2,ngeom])
        errmeas = np.zeros([nconvmax2,ngeom])
        for i in range(ngeom):
            wave[0:nconv[i],:] = wavetmp[0:nconv[i],:]
            meas[0:nconv[i],:] = meastmp[0:nconv[i],:]
            errmeas[0:nconv[i],:] = errmeastmp[0:nconv[i],:]

        self.NGEOM = ngeom
        self.FWHM = inst_fwhm
        self.LATITUDE = xlat
        self.LONGITUDE = xlon
        self.NCONV = nconv
        self.NAV = np.ones(ngeom,dtype='int32')

        self.edit_WGEOM(wgeom)
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_VCONV(wave)
        self.edit_MEAS(meas)
        self.edit_ERRMEAS(errmeas)
        self.edit_TANHE(tanhe)

        self.calc_MeasurementVector()

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(13,4))

            colormap = 'nipy_spectral'
            norm = matplotlib.colors.Normalize(vmin=0.,vmax=self.TANHE.max())
            c_m = plt.cm.get_cmap(colormap,360)
            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])

            for igeom in range(self.NGEOM):
                ax1.plot(self.VCONV[0:self.NCONV[i],igeom],self.MEAS[0:self.NCONV[i],igeom],c=s_m.to_rgba([self.TANHE[igeom,0]]))

            ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax1.set_ylabel('Transmission')
            ax1.grid()

            cax = plt.axes([0.92, 0.15, 0.02, 0.7])   #Bottom
            cbar2 = plt.colorbar(s_m,cax=cax,orientation='vertical')
            cbar2.set_label('Altitude (km)')


    def read_spx(self,runname,MakePlot=False):
    
        """
        Fill the attribute and parameters of the Measurement class for a retrieval

        @param Runname: string
            Name of the Nemesis run 
        """

        #Opening file
        f = open(runname+'.spx','r')

        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=4,dtype='float')
        inst_fwhm = float(tmp[0])
        xlat = float(tmp[1])
        xlon = float(tmp[2])
        ngeom = int(tmp[3])

        #Defining variables
        navmax = 100
        nconvmax = 15000
        nconv = np.zeros([ngeom],dtype='int')
        nav = np.zeros([ngeom],dtype='int')
        flattmp = np.zeros([ngeom,navmax])
        flontmp = np.zeros([ngeom,navmax])
        sol_angtmp = np.zeros([ngeom,navmax])
        emiss_angtmp = np.zeros([ngeom,navmax])
        azi_angtmp = np.zeros([ngeom,navmax])
        wgeomtmp = np.zeros([ngeom,navmax])
        wavetmp = np.zeros([nconvmax,ngeom,navmax])
        meastmp = np.zeros([nconvmax,ngeom,navmax])
        errmeastmp = np.zeros([nconvmax,ngeom,navmax])
        for i in range(ngeom):
            nconv[i] = int(f.readline().strip())
            nav[i] = int(f.readline().strip())
            for j in range(nav[i]):
                tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
                flattmp[i,j] = float(tmp[0])
                flontmp[i,j] = float(tmp[1])
                sol_angtmp[i,j] = float(tmp[2])
                emiss_angtmp[i,j] = float(tmp[3])
                azi_angtmp[i,j] = float(tmp[4])
                wgeomtmp[i,j] = float(tmp[5])
                for iconv in range(nconv[i]):
                    tmp = np.fromfile(f,sep=' ',count=3,dtype='float')
                    wavetmp[iconv,i,j] = float(tmp[0])
                    meastmp[iconv,i,j] = float(tmp[1])
                    errmeastmp[iconv,i,j] = float(tmp[2])

        #Making final arrays for the measured spectra
        nconvmax2 = max(nconv)
        navmax2 = max(nav)
        wave = np.zeros([nconvmax2,ngeom])
        meas = np.zeros([nconvmax2,ngeom])
        errmeas = np.zeros([nconvmax2,ngeom])
        flat = np.zeros([ngeom,navmax2])
        flon = np.zeros([ngeom,navmax2])
        sol_ang = np.zeros([ngeom,navmax2])
        emiss_ang = np.zeros([ngeom,navmax2])
        azi_ang = np.zeros([ngeom,navmax2])
        wgeom = np.zeros([ngeom,navmax2])
        for i in range(ngeom):
            wave[0:nconv[i],i] = wavetmp[0:nconv[i],i,0]
            meas[0:nconv[i],i] = meastmp[0:nconv[i],i,0]
            errmeas[0:nconv[i],i] = errmeastmp[0:nconv[i],i,0]  
            flat[i,0:nav[i]] = flattmp[i,0:nav[i]]
            flon[i,0:nav[i]] = flontmp[i,0:nav[i]]
            sol_ang[i,0:nav[i]] = sol_angtmp[i,0:nav[i]]
            emiss_ang[i,0:nav[i]] = emiss_angtmp[i,0:nav[i]]
            azi_ang[i,0:nav[i]] = azi_angtmp[i,0:nav[i]]
            wgeom[i,0:nav[i]] = wgeomtmp[i,0:nav[i]]

        self.FWHM = inst_fwhm
        self.LATITUDE = xlat
        self.LONGITUDE = xlon
        self.NGEOM = ngeom
        self.NCONV = nconv
        self.NAV = nav
        self.edit_VCONV(wave)
        self.edit_MEAS(meas)
        self.edit_ERRMEAS(errmeas)
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_WGEOM(wgeom)
        self.edit_SOL_ANG(sol_ang)
        self.edit_EMISS_ANG(emiss_ang)
        self.edit_AZI_ANG(azi_ang)

        self.calc_MeasurementVector()

        #Make plot if keyword is specified
        if (MakePlot == True):
            
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(10,6))
            wavemin = wave.min()
            wavemax = wave.max()
            ax1.set_xlim(wavemin,wavemax)
            ax1.ticklabel_format(useOffset=False)
            ax2.set_xlim(wavemin,wavemax)
            ax2.ticklabel_format(useOffset=False)
            ax2.set_yscale('log')

            ax2.set_xlabel('Wavenumber/Wavelength')
            ax1.set_ylabel('Radiance')  
            ax2.set_ylabel('Radiance')

            for i in range(ngeom):
                im = ax1.plot(wave[0:nconv[i],i],meas[0:nconv[i],i])
                ax1.fill_between(wave[0:nconv[i],i],meas[0:nconv[i],i]-errmeas[0:nconv[i],i],meas[0:nconv[i],i]+errmeas[0:nconv[i],i],alpha=0.4)

            for i in range(ngeom):
                im = ax2.plot(wave[0:nconv[i],i],meas[0:nconv[i],i]) 
                ax2.fill_between(wave[0:nconv[i],i],meas[0:nconv[i],i]-errmeas[0:nconv[i],i],meas[0:nconv[i],i]+errmeas[0:nconv[i],i],alpha=0.4)
        
            ax1.grid()
            ax2.grid()
            plt.tight_layout()
            plt.show()


    def read_sha(self,runname):
    
        """
        Read the .sha file to see what the Instrument Lineshape is:
            (0) Square lineshape
            (1) Triangular
            (2) Gaussian
            (3) Hamming
            (4) Hanning 

        @param runname: string
            Name of the Nemesis run 
        """

        #Opening file
        f = open(runname+'.sha','r')
        s = f.readline().split()
        lineshape = int(s[0])

        self.ISHAPE = lineshape

    def read_fil(self,runname,MakePlot=False):
    
        """
        Read the .fil file to see what the Instrument Lineshape for each convolution wavenumber 
        (Only valid if FWHM<0.0)

        @param runname: string
            Name of the Nemesis run 
        """

        #Opening file
        f = open(runname+'.fil','r')
    
        #Reading first and second lines
        nconv = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
        wave = np.zeros([nconv],dtype='d')
        nfil = np.zeros([nconv],dtype='int')
        nfilmax = 100000
        vfil1 = np.zeros([nfilmax,nconv],dtype='d')
        afil1 = np.zeros([nfilmax,nconv],dtype='d')
        for i in range(nconv):
            wave[i] = np.fromfile(f,sep=' ',count=1,dtype='d')
            nfil[i] = np.fromfile(f,sep=' ',count=1,dtype='int')
            for j in range(nfil[i]):
                tmp = np.fromfile(f,sep=' ',count=2,dtype='d')
                vfil1[j,i] = tmp[0]
                afil1[j,i] = tmp[1]

        nfil1 = nfil.max()
        vfil = np.zeros([nfil1,nconv],dtype='d')
        afil = np.zeros([nfil1,nconv],dtype='d')
        for i in range(nconv):
            vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
            afil[0:nfil[i],i] = afil1[0:nfil[i],i]
    
        if self.NCONV[0]!=nconv:
            sys.exit('error :: Number of convolution wavelengths in .fil and .spx files must be the same')

        self.NFIL = nfil
        self.VFIL = vfil
        self.AFIL = afil

        if MakePlot==True:
            fsize = 11
            axis_font = {'size':str(fsize)}
            fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))
        
            ix = 0  #First wavenumber
            ax1.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax1.set_ylabel(r'f($\nu$)',**axis_font)
            ax1.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()
        
            ix = int(nconv/2)-1  #Centre wavenumber
            ax2.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax2.set_ylabel(r'f($\nu$)',**axis_font)
            ax2.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()
        
            ix = nconv-1  #Last wavenumber
            ax3.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax3.set_ylabel(r'f($\nu$)',**axis_font)
            ax3.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax3.ticklabel_format(useOffset=False)
            ax3.grid()
        
            plt.tight_layout()
            plt.show()

    def write_fil(self,runname,MakePlot=False,IGEOM=0):
    
        """
        Write the .fil file to see what the Instrument Lineshape for each convolution wavenumber 
        (Only valid if FWHM<0.0)

        @param runname: string
            Name of the Nemesis run 
        """

        f = open(runname+'.fil','w')
        f.write("%i \n" %  (self.NCONV[IGEOM]))

        #Running for each spectral point
        for i in range(self.NCONV[IGEOM]):
            f.write("%10.7f\n" % self.VCONV[i,IGEOM])

            f.write("%i \n" %  (self.NFIL[i]))
            for j in range(self.NFIL[i]):
                f.write("%10.10f %10.10e\n" % (self.VFIL[j,i], self.AFIL[j,i]) )
        f.close()



    def wavesetc(self,Spectroscopy,IGEOM=0):
        """
        Subroutine to calculate which 'calculation' wavelengths are needed to 
        cover the required 'convolution wavelengths' (In case of line-by-line calculation).

        @param Spectroscopy: Python class object
            Spectroscopy class indicating the grid of calculation wavelengths 
        """

        if self.FWHM>0.0:

            if self.ISHAPE==0:
                dv = 0.5*self.FWHM
            elif self.ISHAPE==1:
                dv = self.FWHM
            elif self.ISHAPE==2:
                dv = 3.* 0.5 * self.FWHM / np.sqrt(np.log(2.0))
            else:
                dv = 3.*self.FWHM

            wavemin = self.VCONV[0,IGEOM] - dv
            wavemax = self.VCONV[self.NCONV[IGEOM]-1,IGEOM] + dv

            if (wavemin<Spectroscopy.WAVE.min() or wavemax>Spectroscopy.WAVE.max()):
                sys.exit('error from wavesetc :: Channel wavelengths not covered by lbl-tables')

        elif self.FWHM<=0.0:

            wavemin = 1.0e10
            wavemax = 0.0
            for i in range(self.NCONV[IGEOM]):
                vminx = self.VFIL[0,i]
                vmaxx = self.VFIL[self.NFIL[i]-1,i]
                if vminx<wavemin:
                    wavemin = vminx
                if vmaxx>wavemax:
                    wavemax= vmaxx

            if (wavemin<Spectroscopy.WAVE.min() or wavemax>Spectroscopy.WAVE.max()):
                sys.exit('error from wavesetc :: Channel wavelengths not covered by lbl-tables')

        #Selecting the necessary wavenumbers
        iwave = np.where( (Spectroscopy.WAVE>=wavemin) & (Spectroscopy.WAVE<=wavemax) )
        iwave = iwave[0]
        self.WAVE = Spectroscopy.WAVE[iwave]
        self.NWAVE = len(self.WAVE)



    def wavesetb(self,Spectroscopy,IGEOM=0):
    
        """
        Subroutine to calculate which 'calculation' wavelengths are needed to 
        cover the required 'convolution wavelengths' (In case of correlated-k calculation).

        @param Spectroscopy: Python class object
            Spectroscopy class indicating the grid of calculation wavelengths 
        """

        #if (vkstep < 0.0 or fwhm == 0.0):
        if self.FWHM==0:

            wave = np.zeros(self.NCONV[IGEOM])
            wave[:] = self.VCONV[0:self.NCONV[IGEOM],IGEOM]
            self.WAVE = wave
            self.NWAVE = self.NCONV[IGEOM]

        elif self.FWHM<0.0:

            wavemin = 1.0e10
            wavemax = 0.0
            for i in range(self.NCONV[IGEOM]):
                vminx = self.VFIL[0,i]
                vmaxx = self.VFIL[self.NFIL[i]-1,i]
                if vminx<wavemin:
                    wavemin = vminx
                if vmaxx>wavemax:
                    wavemax= vmaxx

            """
            #The ILS and FWHM are specified in the .fil file
            nconv1,vconv1,nfil,vfil,afil = read_fil_nemesis(runname)
            if self.NCONV[IGEOM] != nconv1:
                sys.exit('error :: onvolution wavenumbers must be the same in .spx and .fil files')

            for i in range(nconv1):
                vcentral = vconv1[i]
                wavemin = 1.0e6
                wavemax = 0.0
                for j in range(self.NCONV[IGEOM]):
                    dv = abs(vcentral-self.VCONV[j,IGEOM])
                    if dv < 0.0001:
                        vminx = vfil[0,i]
                        vmaxx = vfil[nfil[i]-1,i]
                        if vminx<wavemin:
                            wavemin = vminx
                        if vmaxx>wavemax:
                            wavemax= vmaxx
                    else:
                        print('warning from wavesetb :: Convolution wavenumbers in .spx and .fil do not coincide')
            """

            if (wavemin<Spectroscopy.WAVE.min() or wavemax>Spectroscopy.WAVE.max()):
                sys.exit('error from wavesetc :: Channel wavelengths not covered by k-tables')

            #Selecting the necessary wavenumbers
            iwave = np.where( (Spectroscopy.WAVE>=wavemin) & (Spectroscopy.WAVE<=wavemax) )
            iwave = iwave[0]
            self.WAVE = Spectroscopy.WAVE[iwave]
            self.NWAVE = len(self.WAVE)

        elif self.FWHM>0.0:

            dv = self.FWHM * 0.5
            wavemin = self.VCONV[0,IGEOM] - dv
            wavemax = self.VCONV[self.NCONV[IGEOM]-1,IGEOM] + dv

            if (wavemin<Spectroscopy.WAVE.min() or wavemax>Spectroscopy.WAVE.max()):
                sys.exit('error from wavesetc :: Channel wavelengths not covered by k-tables')

            iwave = np.where( (Spectroscopy.WAVE>=wavemin) & (Spectroscopy.WAVE<=wavemax) )
            iwave = iwave[0]
            self.WAVE = Spectroscopy.WAVE[iwave]
            self.NWAVE = len(self.WAVE)

        else:
            sys.exit('error :: Measurement FWHM is not defined')

    def lblconv(self,ModSpec,IGEOM='All'):
    
        """
        Subroutine to convolve the Modelled spectrum with the Instrument Line Shape 

        @param ModSpec: 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum
        @param IGEOM: int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)
        """

        if IGEOM=='All':

            #It is assumed all geometries cover the same spectral range
            IG = 0 
            yout = np.zeros([self.NCONV[IG],self.NGEOM])
            ynor = np.zeros([self.NCONV[IG],self.NGEOM])

            if self.FWHM>0.0:
                #Set total width of Hamming/Hanning function window in terms of
                #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
                nfw = 3.
                for j in range(self.NCONV[IG]):
                    yfwhm = self.FWHM
                    vcen = self.VCONV[j,IG]
                    if self.ISHAPE==0:
                        v1 = vcen-0.5*yfwhm
                        v2 = v1 + yfwhm
                    elif self.ISHAPE==1:
                        v1 = vcen-yfwhm
                        v2 = vcen+yfwhm
                    elif self.ISHAPE==2:
                        sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
                        v1 = vcen - 3.*sig
                        v2 = vcen + 3.*sig
                    else:
                        v1 = vcen - nfw*yfwhm
                        v2 = vcen + nfw*yfwhm

                    #Find relevant points in tabulated files
                    inwave1 = np.where( (self.WAVE>=v1) & (self.WAVE<=v2) )
                    inwave = inwave1[0]

                    np1 = len(inwave)
                    for i in range(np1):
                        f1=0.0
                        if self.ISHAPE==0:
                            #Square instrument lineshape
                            f1=1.0
                        elif self.ISHAPE==1:
                            #Triangular instrument shape
                            f1=1.0 - abs(self.WAVE[inwave[i]] - vcen)/yfwhm
                        elif self.ISHAPE==2:
                            #Gaussian instrument shape
                            f1 = np.exp(-((self.WAVE[inwave[i]]-vcen)/sig)**2.0)
                        else:
                            sys.exit('lblconv :: ishape not included yet in function')

                        if f1>0.0:
                            yout[j,:] = yout[j,:] + f1*ModSpec[inwave[i],:]
                            ynor[j,:] = ynor[j,:] + f1

                    yout[j,:] = yout[j,:]/ynor[j,:]



            elif self.FWHM<0.0:

                #Line shape for each convolution number in each case is read from .fil file
                for j in range(self.NCONV[IG]):
                    v1 = self.VFIL[0,j]
                    v2 = self.VFIL[self.NFIL[j]-1,j]
                    #Find relevant points in tabulated files
                    inwave1 = np.where( (self.WAVE>=v1) & (self.WAVE<=v2) )
                    inwave = inwave1[0]

                    np1 = len(inwave)
                    xp = np.zeros([self.NFIL[j]])
                    yp = np.zeros([self.NFIL[j]])
                    xp[:] = self.VFIL[0:self.NFIL[j],j]
                    yp[:] = self.AFIL[0:self.NFIL[j],j]

                    for i in range(np1):
                        #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                        f1 = np.interp(self.WAVE[inwave[i]],xp,yp)
                        if f1>0.0:
                            yout[j,:] = yout[j,:] + f1*ModSpec[inwave[i],:]
                            ynor[j,:] = ynor[j,:] + f1

                    yout[j,:] = yout[j,:]/ynor[j,:]

        else:
            sys.exit('error in lblconv :: Must implement the case IGEOM!=-1')

        return yout

    def lblconvg(self,ModSpec,ModGrad,IGEOM='All'):
    
        """
        Subroutine to convolve the Modelled spectrum and the gradients with the Instrument Line Shape 

        @param ModSpec: 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum
        @param ModGrad: 1D or 2D array (NWAVE,NGEOM,NX)
            
        @param IGEOM: int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)
        """

        if IGEOM=='All':

            #It is assumed all geometries cover the same spectral range
            IG = 0 
            NX = len(ModGrad[0,0,:])
            yout = np.zeros((self.NCONV[IG],self.NGEOM))
            ynor = np.zeros((self.NCONV[IG],self.NGEOM))
            gradout = np.zeros((self.NCONV[IG],self.NGEOM,NX))
            gradnorm = np.zeros((self.NCONV[IG],self.NGEOM,NX))

            if self.FWHM>0.0:
                #Set total width of Hamming/Hanning function window in terms of
                #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
                nfw = 3.
                for j in range(self.NCONV[IG]):
                    yfwhm = self.FWHM
                    vcen = self.VCONV[j,IG]
                    if self.ISHAPE==0:
                        v1 = vcen-0.5*yfwhm
                        v2 = v1 + yfwhm
                    elif self.ISHAPE==1:
                        v1 = vcen-yfwhm
                        v2 = vcen+yfwhm
                    elif self.ISHAPE==2:
                        sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
                        v1 = vcen - 3.*sig
                        v2 = vcen + 3.*sig
                    else:
                        v1 = vcen - nfw*yfwhm
                        v2 = vcen + nfw*yfwhm

                    #Find relevant points in tabulated files
                    inwave1 = np.where( (self.WAVE>=v1) & (self.WAVE<=v2) )
                    inwave = inwave1[0]

                    np1 = len(inwave)
                    for i in range(np1):
                        f1=0.0
                        if self.ISHAPE==0:
                            #Square instrument lineshape
                            f1=1.0
                        elif self.ISHAPE==1:
                            #Triangular instrument shape
                            f1=1.0 - abs(self.WAVE[inwave[i]] - vcen)/yfwhm
                        elif self.ISHAPE==2:
                            #Gaussian instrument shape
                            f1 = np.exp(-((self.WAVE[inwave[i]]-vcen)/sig)**2.0)
                        else:
                            sys.exit('lblconv :: ishape not included yet in function')

                        if f1>0.0:
                            yout[j,:] = yout[j,:] + f1*ModSpec[inwave[i],:]
                            ynor[j,:] = ynor[j,:] + f1
                            gradout[j,:,:] = gradout[j,:,:] + f1*ModGrad[inwave[i],:,:]
                            gradnorm[j,:,:] = gradnorm[j,:,:] + f1

                    yout[j,:] = yout[j,:]/ynor[j,:]
                    gradout[j,:,:] = gradout[j,:,:]/gradnorm[j,:,:]

            elif self.FWHM<0.0:

                #Line shape for each convolution number in each case is read from .fil file
                for j in range(self.NCONV[IG]):
                    v1 = self.VFIL[0,j]
                    v2 = self.VFIL[self.NFIL[j]-1,j]
                    #Find relevant points in tabulated files
                    inwave1 = np.where( (self.WAVE>=v1) & (self.WAVE<=v2) )
                    inwave = inwave1[0]

                    np1 = len(inwave)
                    xp = np.zeros([self.NFIL[j]])
                    yp = np.zeros([self.NFIL[j]])
                    xp[:] = self.VFIL[0:self.NFIL[j],j]
                    yp[:] = self.AFIL[0:self.NFIL[j],j]

                    for i in range(np1):
                        #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                        f1 = np.interp(self.WAVE[inwave[i]],xp,yp)
                        if f1>0.0:
                            yout[j,:] = yout[j,:] + f1*ModSpec[inwave[i],:]
                            ynor[j,:] = ynor[j,:] + f1
                            gradout[j,:,:] = gradout[j,:,:] + f1*ModGrad[inwave[i],:,:]
                            gradnorm[j,:,:] = gradnorm[j,:,:] + f1

                    yout[j,:] = yout[j,:]/ynor[j,:]
                    gradout[j,:,:] = gradout[j,:,:]/gradnorm[j,:,:]

        else:
            sys.exit('error in lblconv :: Must implement the case IGEOM!=-1')

        return yout,gradout


    def conv(self,ModSpec,IGEOM='All',FWHMEXIST=''):
    
        """
        Subroutine to convolve the Modelled spectrum with the Instrument Line Shape 

        @param ModSpec: 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum
        @param IGEOM: int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)
        @param FWHMEXIST: str
            If != '', then FWHMEXIST indicates that the .fwhm exists (that includes the variation of FWHM for each wave) and 
            FWHMEXIST is expected to be the name of the Nemesis run
        """

        import os.path
        from scipy import interpolate

        nstep = 20

        if IGEOM=='All':

            sys.exit("error in conv :: Must implement the case IGEOM=All")

        else:

            yout = np.zeros(self.NCONV[IGEOM])
            ynor = np.zeros(self.NCONV[IGEOM])

            if self.FWHM>0.0:

                nwave1 = self.NWAVE
                wave1 = np.zeros(nwave+2)
                y1 = np.zeros(nwave+2)
                wave1[1:nwave+1] = self.WAVE
                y1[1:nwave+1] = ModSpec[0:self.NWAVE]

                #Extrapolating the last wavenumber
                iup = 0
                if(self.VCONV[self.NCONV[IGEOM],IGEOM]>(self.WAVE.max()-self.FWHM/2.)):
                    nwave1 = nwave1 +1
                    wave1[nwave1-1] = self.VCONV[self.NCONV[IGEOM],IGEOM] + self.FWHM
                    frac = (ModSpec[self.NWAVE-1]-ModSpec[self.NWAVE-2])/(self.WAVE[self.NWAVE-1]-self.WAVE[self.NWAVE-2])
                    y1[nwave-1] = ModSpec[Measurement.NWAVE-1] + frac * (wave1[nwave1-1]-self.WAVE[self.NWAVE-1])
                    iup=1

                #Extrapolating the first wavenumber
                idown = 0
                if(self.VCONV[0,IGEOM]<(self.WAVE.min()+self.FWHM/2.)):
                    nwave1 = nwave1 + 1
                    wave1[0] = self.VCONV[0,IGEOM] - self.FWHM
                    frac = (ModSpec[1] - ModSpec[2])/(self.WAVE[1]-self.WAVE[0])
                    y1[0] = ModSpec[0] + frac * (wave1[0] - self.WAVE[0])
                    idown = 1

                #Re-shaping the spectrum
                nwave = nwave1 + iup + idown
                wave = np.zeros(nwave)
                y = np.zeros(nwave)
                if((idown==1) & (iup==1)):
                    wave[:] = wave1[:]
                    y[:] = y1[:]
                elif((idown==1) & (iup==0)):
                    wave[0:nwave] = wave1[0:nwave1-1]
                    y[0:nwave] = y1[0:nwave1-1]
                elif((idown==0) & (iup==1)):
                    wave[0:nwave] = wave1[1:nwave1]
                    y[0:nwave] = y1[1:nwave1]
                else:
                    wave[0:nwave] = wave1[1:nwave1-1]
                    y[0:nwave] = y1[1:nwave1-1]

                #Checking if .fwh file exists (indicating that FWHM varies with wavelength)
                ifwhm = 0
                if os.path.exists(FWHMEXIST+'.fwh')==True:

                    #Reading file
                    f = open(FWHMEXIST+'.fwh')
                    s = f.readline().split()
                    nfwhm = int(s[0])
                    vfwhm = np.zeros(nfwhm)
                    xfwhm = np.zeros(nfwhm)
                    for ifwhm in range(nfwhm):
                        s = f.readline().split()
                        vfwhm[i] = float(s[0])
                        xfwhm[i] = float(s[1])
                    f.close()

                    ffwhm = interpolate.interp1d(vfwhm,xfwhm)
                    ifwhm==1

                fy = interpolate.CubicSpline(wave,y)
                for ICONV in range(self.NCONV[IGEOM]):
                    
                    if ifwhm==1:
                        yfwhm = ffwhm(self.VCONV[ICONV,IGEOM])
                    else:
                        yfwhm = self.FWHM

                    x1 = self.VCONV[ICONV,IGEOM] - yfwhm/2.
                    x2 = self.VCONV[ICONV,IGEOM] + yfwhm/2.
                    delx = (x2-x1)/(nstep-1)
                    xi = np.linspace(x1,x2,nstep)
                    yi = fy(xi)
                    for j in range(nstep):
                        if j==0:
                            sum1 = 0.0 
                        else:
                            sum1 = sum1 + (yi[j] - yold) * delx/2.
                        yold = yi[j]

                    yout[ICONV] = sum1 / yfwhm

            elif self.FWHM==0.0:

                #Channel Integrator mode where the k-tables have been previously
                #tabulated INCLUDING the filter profile. In which case all we
                #need do is just transfer the outputs
                yout[:] = ModSpec[:]

            elif self.FWHM<0.0:

                #Channel Integrator Mode: Slightly more advanced than previous

                #In this case the filter function for each convolution wave is defined in the .fil file
                #This file has been previously read and its variables are stored in NFIL,VFIL,AFIL

                for ICONV in range(self.NCONV[IGEOM]):

                    v1 = self.VFIL[0,ICONV]
                    v2 = self.VFIL[self.NFIL[ICONV]-1,ICONV]
                    #Find relevant points in tabulated files
                    inwave1 = np.where( (self.WAVE>=v1) & (self.WAVE<=v2) )
                    inwave = inwave1[0]

                    np1 = len(inwave)
                    xp = np.zeros([self.NFIL[ICONV]])
                    yp = np.zeros([self.NFIL[ICONV]])
                    xp[:] = self.VFIL[0:self.NFIL[ICONV],ICONV]
                    yp[:] = self.AFIL[0:self.NFIL[ICONV],ICONV]

                    for i in range(np1):
                        #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                        f1 = np.interp(self.WAVE[inwave[i]],xp,yp)
                        if f1>0.0:
                            yout[ICONV] = yout[ICONV] + f1*ModSpec[inwave[i]]
                            ynor[ICONV] = ynor[ICONV] + f1

                    yout[ICONV] = yout[ICONV]/ynor[ICONV]
                
        return yout

                