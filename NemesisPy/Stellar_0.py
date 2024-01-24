from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Stellar_0:

    def __init__(self, SOLEXIST=True, DIST=None, RADIUS=None, ISPACE=None, NCONV=None):

        """
        Inputs
        ------
        @param SOLEXIST: log,
            Flag indicating whether sunlight needs to be included (SOLEXIST=True) or not (SOLEXIST=False)
        @param DIST: float,
            Distance between star and planet (AU) 
        @param RADIUS: float,
            Radius of the star (km)       
        @param ISPACE: int,
            Spectral units in which the solar spectrum is defined (0) Wavenumber (1) Wavelength              
        @param NCONV: int,
            Number of spectral points in which the stellar spectrum is defined
        
        Attributes
        ----------
        @attribute VCONV: 1D array
            Wavelength array at which the stellar file is defined
        @attribute SOLSPEC: 1D array
            Stellar luminosity spectrum (W/(cm-1) or W/um)
        @attribute SOLFLUX: 1D array
            Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)
        @attribute STELLARDATA: str
            String indicating where the STELLAR data files are stored

        Methods
        -------
        Stellar_0.edit_SOLSPEC
        """

        from NemesisPy import Nemesis_Path

        #Input parameters
        self.SOLEXIST = SOLEXIST
        self.DIST = DIST
        self.RADIUS = RADIUS
        self.ISPACE = ISPACE
        self.NCONV = NCONV

        # Input the following profiles using the edit_ methods.
        self.VCONV = None # np.zeros(NCONV)
        self.SOLSPEC = None # np.zeros(NCONV)
        self.SOLFLUX = None #np.zeros(NCONV)

        self.STELLARDATA = Nemesis_Path()+'NemesisPy/Data/stellar/'


    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        if self.SOLEXIST==True:

            #Checking some common parameters to all cases
            assert np.issubdtype(type(self.ISPACE), np.integer) == True , \
                'ISPACE must be int'
            assert self.ISPACE >= 0 , \
                'ISPACE must be >=0 and <=1'
            assert self.ISPACE <= 1 , \
                'ISPACE must be >=0 and <=1'

            #Checking some common parameters to all cases
            assert np.issubdtype(type(self.DIST), np.float64) == True , \
                'DIST must be float'

            #Checking some common parameters to all cases
            assert np.issubdtype(type(self.RADIUS), np.float64) == True , \
                'RADIUS must be float'

            #Checking some common parameters to all cases
            assert np.issubdtype(type(self.NCONV), np.integer) == True , \
                'NCONV must be int'
            assert self.NCONV >= 0 , \
                'NCONV must be >=0'
            
            assert len(self.VCONV) == self.NCONV , \
                'VCONV must have size (NCONV)'
            
            assert len(self.SOLSPEC) == self.NCONV , \
                'SOLSPEC must have size (NCONV)'
            

    def edit_VCONV(self, VCONV):
        """
        Edit the solar spectrum 
        @param VCONV: 1D array
            Array defining the wavelengths at which the solar spectrum is defined
        """
        VCONV_array = np.array(VCONV)
        assert len(VCONV_array) == self.NCONV, 'VCONV should have NCONV elements'
        self.VCONV = VCONV_array

    def edit_SOLSPEC(self, SOLSPEC):
        """
        Edit the solar spectrum 
        @param SOLSPEC: 1D array
            Array defining the solar spectrum
        """
        SOLSPEC_array = np.array(SOLSPEC)
        assert len(SOLSPEC_array) == self.NCONV, 'SOLSPEC should have NCONV elements'
        self.SOLSPEC = SOLSPEC_array

    def write_hdf5(self,runname,solfile=None):
        """
        Write the information about the solar spectrum in the HDF5 file

        If the optional input solfile is defined, then the information is read from the
        specified file (assumed to be stored in the Data/stellar directory).

        If solfile is not defined, then the information about the solar spectrum is assumed
        to be defined in the class
        """

        import h5py
        from NemesisPy.Utils import file_lines

        if solfile is not None:

            #Reading the solar spectrum file

            nlines = file_lines(self.STELLARDATA+solfile)

            #Reading buffer
            ibuff = 0
            with open(self.STELLARDATA+solfile,'r') as fsol:
                for curline in fsol:
                    if curline.startswith("#"):
                        ibuff = ibuff + 1
                    else:
                        break

            nvsol = nlines - ibuff - 2
            
            #Reading file
            fsol = open(self.STELLARDATA+solfile,'r')
            for i in range(ibuff):
                s = fsol.readline().split()
        
            s = fsol.readline().split()
            ispace = int(s[0])
            s = fsol.readline().split()
            solrad = float(s[0])
            vsol = np.zeros(nvsol)
            rad = np.zeros(nvsol)
            for i in range(nvsol):
                s = fsol.readline().split()
                vsol[i] = float(s[0])
                rad[i] = float(s[1])
        
            fsol.close()


            self.RADIUS = solrad
            self.ISPACE = ispace
            self.NCONV = nvsol
            self.edit_VCONV(vsol)
            self.edit_SOLSPEC(rad)


        self.assess()

        #Writing the information into the HDF5 file
        f = h5py.File(runname+'.h5','a')
        #Checking if Stellar already exists
        if ('/Stellar' in f)==True:
            del f['Stellar']   #Deleting the Stellar information that was previously written in the file

        if self.SOLEXIST==True:

            grp = f.create_group("Stellar")

            #Writing the spectral units
            dset = grp.create_dataset('ISPACE',data=self.ISPACE)
            dset.attrs['title'] = "Spectral units"
            if self.ISPACE==0:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==1:
                dset.attrs['units'] = 'Wavelength / um'

            #Writing the Planet-Star distance
            dset = grp.create_dataset('DIST',data=self.DIST)
            dset.attrs['title'] = "Planet-Star distance"
            dset.attrs['units'] = 'Astronomical Units'

            #Writing the Star radius
            dset = grp.create_dataset('RADIUS',data=self.RADIUS)
            dset.attrs['title'] = "Star radius"
            dset.attrs['units'] = 'km'

            #Writing the number of points in stellar spectrum
            dset = grp.create_dataset('NCONV',data=self.NCONV)
            dset.attrs['title'] = "Number of spectral points in stellar spectrum"

            #Writing the spectral array
            dset = grp.create_dataset('VCONV',data=self.VCONV)
            dset.attrs['title'] = "Spectral array"
            if self.ISPACE==0:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==1:
                dset.attrs['units'] = 'Wavelength / um' 

            #Writing the solar spectrum
            dset = grp.create_dataset('SOLSPEC',data=self.SOLSPEC)
            dset.attrs['title'] = "Stellar power spectrum"
            if self.ISPACE==0:
                dset.attrs['units'] = 'W (cm-1)-1'
            elif self.ISPACE==1:
                dset.attrs['units'] = 'W um-1'     

        f.close()

    def read_hdf5(self,runname):
        """
        Read the Stellar properties from an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','r')

        #Checking if Surface exists
        e = "/Stellar" in f
        if e==False:
            self.SOLEXIST = False
        else:
            self.SOLEXIST = True
            self.ISPACE = np.int32(f.get('Stellar/ISPACE'))
            self.DIST = np.float64(f.get('Stellar/DIST'))
            self.RADIUS = np.float64(f.get('Stellar/RADIUS'))
            self.NCONV = np.int32(f.get('Stellar/NCONV'))
            self.VCONV = np.array(f.get('Stellar/VCONV'))
            self.SOLSPEC = np.array(f.get('Stellar/SOLSPEC'))

        f.close()

    def read_sol(self, runname, MakePlot=False):
        """
        Read the solar spectrum from the .sol file. There are two options for this file:

            - The only line in the file is the name of another file including the solar power spectrum,
              assumed to be stored in the Data/stellar/ directory

            - The first line is equal to -1. Then the stellar spectrum is read from the solar file.

        @param runname: str
            Name of the NEMESIS run
        """

        from NemesisPy import file_lines
        
        #Opening file
        f = open(runname+'.sol','r')
        s = f.readline().split()
        solname = s[0]
        f.close()

        if solname=='-1':
            #Information about stellar spectrum is stored in this same file
            nlines = file_lines(runname+'.sol')
            nvsol = nlines - 3


            f = open(runname+'.sol','r')
            s = f.readline().split()
            solname = s[0]

            s = f.readline().split()
            ispace = int(s[0])
            s = f.readline().split()
            solrad = float(s[0])
            vsol = np.zeros(nvsol)
            rad = np.zeros(nvsol)
            for i in range(nvsol):
                s = f.readline().split()
                vsol[i] = float(s[0])
                rad[i] = float(s[1])
            f.close()

        else:

            nlines = file_lines(self.STELLARDATA+solname)

            #Reading buffer
            ibuff = 0
            with open(self.STELLARDATA+solname,'r') as fsol:
                for curline in fsol:
                    if curline.startswith("#"):
                        ibuff = ibuff + 1
                    else:
                        break

            nvsol = nlines - ibuff - 2
            
            #Reading file
            fsol = open(self.STELLARDATA+solname,'r')
            for i in range(ibuff):
                s = fsol.readline().split()
        
            s = fsol.readline().split()
            ispace = int(s[0])
            s = fsol.readline().split()
            solrad = float(s[0])
            vsol = np.zeros(nvsol)
            rad = np.zeros(nvsol)
            for i in range(nvsol):
                s = fsol.readline().split()
                vsol[i] = float(s[0])
                rad[i] = float(s[1])
        
            fsol.close()

        self.RADIUS = solrad
        self.ISPACE = ispace
        self.NCONV = nvsol
        self.edit_VCONV(vsol)
        self.edit_SOLSPEC(rad)

        if MakePlot==True:
            fig,ax1=plt.subplots(1,1,figsize=(8,3))
            ax1.plot(vsol,rad)
            #ax1.set_yscale('log')
            plt.tight_layout()
            plt.show()

    def write_sol(self,runname):
        """
        Write the solar power into a .sol file with the format required by NEMESISpy, in the case
        that the solar power spectrum is stored directy in the .sol file
        """

        f = open(runname+'.sol','w')

        #Defining errors while writing file
        if self.ISPACE is None:
            sys.exit('error :: ISPACE must be defined in Stellar class to write Stellar power to file')

        if self.RADIUS is None:
            sys.exit('error :: RADIUS must be defined in Stellar class to write Stellar power to file')

        if self.NCONV is None:
            sys.exit('error :: NCONV must be defined in Stellar class to write Stellar power to file')

        if self.VCONV is None:
            sys.exit('error :: VCONV must be defined in Stellar class to write Stellar power to file')

        if self.SOLSPEC is None:
            sys.exit('error :: SOLSPEC Must be defined in Stellar class to write Stellar power to file')


        header = '-1'
        f.write(header+' \n')
        f.write('\t %i \n' % (self.ISPACE))
        f.write('\t %7.3e \n' % (self.RADIUS))
        for i in range(self.NCONV):
            f.write('\t %7.6f \t %7.5e \n' % (self.VCONV[i],self.SOLSPEC[i]))
        f.close()


    def calc_solar_flux(self):
        """
        Calculate the stellar flux at the planet's distance
        """

        AU = 1.49598e11
        area = 4.*np.pi*(self.DIST * AU * 100. )**2.
        self.SOLFLUX = self.SOLSPEC / area   #W cm-2 (cm-1)-1 or W cm-2 um-1


    def calc_solar_power(self):
        """
        Calculate the stellar power based on the solar flux measured at a given distance
        """

        AU = 1.49598e11
        area = 4.*np.pi*(self.DIST * AU * 100. )**2.
        self.SOLSPEC = self.SOLFLUX * area   #W (cm-1)-1 or W um-1


    def write_solar_file(self,filename,header=None):
        """
        Write the solar power into a file with the format required by NEMESIS
        """

        f = open(filename,'w')

        #Defining errors while writing file
        if self.ISPACE is None:
            sys.exit('error :: ISPACE must be defined in Stellar class to write Stellar power to file')

        if self.RADIUS is None:
            sys.exit('error :: RADIUS must be defined in Stellar class to write Stellar power to file')

        if self.NCONV is None:
            sys.exit('error :: NCONV must be defined in Stellar class to write Stellar power to file')

        if self.VCONV is None:
            sys.exit('error :: VCONV must be defined in Stellar class to write Stellar power to file')

        if self.SOLSPEC is None:
            sys.exit('error :: SOLSPEC Must be defined in Stellar class to write Stellar power to file')


        if header==None:
            if self.ISPACE==0:
                header = '# Stellar power in W (cm-1)-1'
            elif self.ISPACE==1:
                header = '# Stellar power in W um-1' 
        else:
            if header[0]!='#':
                header = '#'+header
        
        f.write(header+' \n')
        f.write('\t %i \n' % (self.ISPACE))
        f.write('\t %7.3e \n' % (self.RADIUS))
        for i in range(self.NCONV):
            f.write('\t %7.6f \t %7.5e \n' % (self.VCONV[i],self.SOLSPEC[i]))

        f.close()