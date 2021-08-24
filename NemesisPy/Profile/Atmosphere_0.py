#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:27:12 2021

@author: jingxuanyang

Basic Atmosphere Class.
"""
import numpy as np
from scipy.special import legendre
from NemesisPy.Data import *
from NemesisPy.Utils.Utils import *

class Atmosphere_0:
    """
    Clear atmosphere. Simplest possible profile.
    """
    def __init__(self, runname='wasp43b', NP=10, NVMR=6, ID=[0,0,0,0,0,0],
                ISO=[0,0,0,0,0,0], LATITUDE=0.0, IPLANET=1, AMFORM=1, RADIUS=0.0):
        """
        Set up an atmosphere profile with NP points and NVMR gases.
        Use the class methods to edit Height, Pressure, Temperature and
        gas Volume Mixing Ratios after creating an Atmosphere_0 object, e.g.
            atm = Atmosphere_0('Jupiter',100,6,[1,2,4,5,39,40],
                      [0,0,0,0,0,0],0,1,1)
            atm.edit_H(your_height_profile)
            atm.edit_P(your_pressure_profile)
            atm.edit_T(your_temperature_profile)
            atm.edit_VMR(your_VMR_profile)
        Can write to a *.prf file in Nemesis format.

        Inputs
        ------
        @param runname: str
            Name of this particular profile (no space)
        @param NP: int,
            Number of points defined in the profile
        @param NVMR: int
            Number of gases in the profile (expressed as volume mixing ratios)
        @param ID: 1D array
            Gas ID for each gas to be defined in the profile
        @param ISO: 1D array
            Isotope ID for each gas, default 0 for all
            isotopes in terrestrial relative abundance
        @param IPLANET: int
            Planet ID
        @param LATITUDE: real
            Planetocentric latitude
        @param AMFORM: int,
            Format of the atmospheric profile, default AMFORM=1:
            assume that at each level the VMRs add up to 1.0.

        Attributes
        ----------
        @attribute H: 1D array
            Height in m of each points above reference level:
            ground for terrestrial planets,
            usually 1-bar level for gas giants
        @attribute P: 1D array
            Pressure in Pa of each point defined in the profile
        @attribute T: 1D array
            Temperature in K at each point defined in the profile
        @attribute VMR: 2D array
            VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
            the column j corresponds to the gas with RADTRANS ID ID[j]
            and RADTRANS isotope ID ISO[j]

        Methods
        -------
        Atmosphere_0.edit_H
        Atmosphere_0.edit_P
        Atmosphere_0.edit_T
        Atmosphere_0.edit_VMR
        Atmosphere_0.write_to_file
        Atmosphere_0.check
        """

        assert type(runname) == str and len(runname) <= 100,\
            'runname should be a string <= 100 char'
        assert type(NP) == int and type(NVMR) == int,\
            'NP and NVMR should be integers'
        try:
            assert type(ID) == np.ndarray
        except:
            assert type(ID) == list, 'ID should be an array'
            ID = np.array(ID)
        try:
            assert type(ISO) == np.ndarray
        except:
            assert type(ISO) == list, 'ISO should be an array'
            ISO = np.array(ISO)
        assert len(ID) == NVMR, "len(ID) should be equal to NVMR"
        for i in range(NVMR):
            assert type(ID[i]) == np.int64,\
                'RADTRANS gas IDs should be integers'
            assert type(ISO[i]) == np.int64,\
                'RADTRANS isotope IDs should be integers'
        assert len(ID) == len(ISO),\
            "len(ID) should be equal to len(ISO)"
        assert type(IPLANET) == int and type(AMFORM) == int,\
            'IPLANET and AMFORM should be integers'
        assert abs(LATITUDE) <= 90, 'LATITUDE should be <= 90 deg'

        self.runname = runname
        self.NP = NP
        self.NVMR = NVMR
        self.ID = ID
        self.ISO = ISO
        self.IPLANET = IPLANET
        self.LATITUDE = LATITUDE
        self.AMFORM = AMFORM

        # Input the following profiles using the edit_ methods.
        self.H = None # np.zeros(NP)
        self.P = None # np.zeros(NP)
        self.T =  None # np.zeros(NP)
        self.MOLWT = None #np.zeros(NP)
        self.GRAV = None #np.zeros(NP)
        self.VMR = None # np.zeros((NP, NVMR))

    def edit_H(self, H_array):
        """
        Edit the Height profile.
        @param H_array: 1D array
            Heights of the vertical points in m
        """
        H_array = np.array(H_array)
        assert len(H_array) == self.NP, 'H should have NP elements'
        assert ((H_array[1:]-H_array[:-1])>0).all(),\
            'H should be strictly increasing'
        self.H = H_array

    def edit_P(self, P_array):
        """
        Edit the Pressure profile.
        @param P_array: 1D array
            Pressures of the vertical points in Pa
        """
        P_array = np.array(P_array)
        assert len(P_array) == self.NP, 'P should have NP elements'
        #assert (P_array>0).all() == True, 'P should be positive'
        #assert ((P_array[1:]-P_array[:-1])<0).all(),\
        #    'P should be strictly decreasing'
        self.P = P_array

    def edit_T(self, T_array):
        """
        Edit the Temperature profile.
        @param T_array: 1D array
            Temperature of the vertical points in K
        """
        T_array = np.array(T_array)
        assert len(T_array) == self.NP, 'T should have NP elements'
        assert (T_array>0).all() == True, 'T should be positive'
        self.T = T_array

    def edit_VMR(self, VMR_array):
        """
        Edit the gas Volume Mixing Ratio profile.
        @param VMR_array: 2D array
            NP by NVMR array containing the Volume Mixing Ratio of gases.
            VMR_array[i,j] is the volume mixing ratio of gas j at point i.
        """
        VMR_array = np.array(VMR_array)
        try:
            assert VMR_array.shape == (self.NP, self.NVMR),\
                'VMR should be NP by NVMR.'
        except:
            assert VMR_array.shape == (self.NP,) and self.NVMR==1,\
                'VMR should be NP by NVMR.'
        assert (VMR_array>=0).all() == True,\
            'VMR should be non-negative'
        self.VMR = VMR_array

    def adjust_VMR(self, ISCALE=[1,1,1,1,1,1]):

        """
        Subroutine to adjust the vmrs at a particular level to add up to 1.0.
        ISCALE :: Flag to indicate if gas vmr can be scaled(1) or not (0).
        """ 

        ISCALE = np.array(ISCALE)
        jvmr1 = np.where(ISCALE==1)
        jvmr2 = np.where(ISCALE==0)

        vmr = np.zeros([self.NP,self.NVMR])
        vmr[:,:] = self.VMR
        for ipro in range(self.NP):

            sumtot = np.sum(self.VMR[ipro,:])
            sum1 = np.sum(self.VMR[ipro,jvmr2]) 

            if sumtot!=1.0:
                #Need to adjust the VMRs of those gases that can be scaled to 
                #bring the total sum to 1.0
                xfac = (1.0-sum1)/(sumtot-sum1)
                vmr[ipro,jvmr1] = self.VMR[ipro,jvmr1] * xfac
                #self.VMR[ipro,jvmr1] = self.VMR[ipro,jvmr1] * xfac

        self.edit_VMR(vmr)

    def calc_molwt(self):
        """
        Subroutine to calculate the molecular weight of the atmosphere (kg/mol)
        """      

        molwt = np.zeros(self.NP)
        vmrtot = np.zeros(self.NP)
        for i in range(self.NVMR):
            if self.ISO[i]==0:
                molwt1 = gas_info[str(self.ID[i])]['mmw']
            else:
                molwt1 = gas_info[str(self.ID[i])]['isotope'][str(self.ISO[i])]['mass']

            vmrtot[:] = vmrtot[:] + self.VMR[:,i]
            molwt[:] = molwt[:] + self.VMR[:,i] * molwt1

        molwt = molwt / vmrtot
        self.MOLWT = molwt / 1000.

    def calc_rho(self):
        """
        Subroutine to calculate the atmospheric density (kg/m3) at each level
        """      
        R = const["R"]
        rho = self.P * self.MOLWT / R / self.T

        return rho


    def calc_radius(self):
        """
        Subroutine to calculate the radius of the planet at the required latitude
        """  

        #Getting the information about the planet
        data = planet_info[str(self.IPLANET)]
        xradius = data["radius"] * 1.0e5   #cm

        #Calculating some values to account for the latitude dependence
        lat = 2 * np.pi * self.LATITUDE/360.      #Latitude in rad
        latc = np.arctan(np.tan(lat)/xellip**2.)   #Converts planetographic latitude to planetocentric
        slatc = np.sin(latc)
        clatc = np.cos(latc)
        Rr = np.sqrt(clatc**2 + (xellip**2. * slatc**2.))  #ratio of radius at equator to radius at current latitude
        r = (xradius+self.H*1.0e2)/Rr    #Radial distance of each altitude point to centre of planet (cm)
        radius = (xradius/Rr)*1.0e-5     #Radius of the planet at the given distance (km)

        self.RADIUS = radius * 1.0e3     #Metres

    def calc_grav(self):
        """
        Subroutine to calculate the gravity at each level following the method
        of Lindal et al., 1986, Astr. J., 90 (6), 1136-1146
        """   

        #Reading data and calculating some parameters   
        Grav = const["G"]
        data = planet_info[str(self.IPLANET)]
        xgm = data["mass"] * Grav * 1.0e24 * 1.0e6
        xomega = 2.*np.pi / (data["rotation"]*24.*3600.)
        xellip=1.0/(1.0-data["flatten"])
        Jcoeff = data["Jcoeff"]
        xcoeff = np.zeros(3)
        xcoeff[0] = Jcoeff[0] / 1.0e3
        xcoeff[1] = Jcoeff[1] / 1.0e6
        xcoeff[2] = Jcoeff[2] / 1.0e8
        xradius = data["radius"] * 1.0e5   #cm
        isurf = data["isurf"]
        name = data["name"]


        #Calculating some values to account for the latitude dependence
        lat = 2 * np.pi * self.LATITUDE/360.      #Latitude in rad
        latc = np.arctan(np.tan(lat)/xellip**2.)   #Converts planetographic latitude to planetocentric
        slatc = np.sin(latc)
        clatc = np.cos(latc)
        Rr = np.sqrt(clatc**2 + (xellip**2. * slatc**2.))  #ratio of radius at equator to radius at current latitude
        r = (xradius+self.H*1.0e2)/Rr    #Radial distance of each altitude point to centre of planet (cm)
        radius = (xradius/Rr)*1.0e-5     #Radius of the planet at the given distance (km)

        self.RADIUS = radius * 1.0e3

        #Calculating Legendre polynomials
        pol = np.zeros(6)
        for i in range(6):
            Pn = legendre(i+1)
            pol[i] = Pn(slatc)

        #Evaluate radial contribution from summation 
        # for first three terms,
        #then subtract centrifugal effect.
        g = 1.
        for i in range(3):
            ix = i + 1
            g = g - ((2*ix+1) * Rr**(2 * ix) * xcoeff[ix-1] * pol[2*ix-1])

        gradial = (g * xgm/r**2.) - (r * xomega**2. * clatc**2.)

        #Evaluate latitudinal contribution for 
        # first three terms, then add centrifugal effects

        gtheta1 = 0.
        for i in range(3):
            ix = i + 1
            gtheta1 = gtheta1 - (4. * ix**2 * Rr**(2 * ix) * xcoeff[ix-1] * (pol[2*ix-1-1] - slatc * pol[2*ix-1])/clatc)

        gtheta = (gtheta1 * xgm/r**2) + (r * xomega**2 * clatc * slatc)

        #Combine the two components and write the result
        gtot = np.sqrt(gradial**2. + gtheta**2.)*0.01   #m/s2

        self.GRAV = gtot

    def adjust_hydrostatP(self,htan,ptan):
        """
        Subroutine to rescale the pressures of a H/P/T profile according to
        the hydrostatic equation above and below a specified altitude
        given the pressure at that altitude
            htan :: specified altitude (m)
            ptan :: Pressure at specified altitude (Pa) 
        """   

        #First find the level below the reference altitude
        alt0,ialt = find_nearest(self.H,htan)
        if ( (alt0>htan) & (ialt>0)):
            ialt = ialt -1

        #Calculating the gravity at each altitude level
        self.calc_grav()

        #Calculate the scaling factor
        R = const["R"]
        scale = R * self.T / (self.MOLWT * self.GRAV)   #scale height (m)

        sh =  0.5*(scale[ialt]+scale[ialt+1])
        delh = self.H[ialt+1]-htan 
        p = np.zeros(self.NP)
        p[ialt+1] = ptan*np.exp(-delh/sh)
        delh = self.H[ialt]-htan
        p[ialt] = ptan*np.exp(-delh/sh)

        for i in range(ialt+2,self.NP):
            sh =  0.5*(scale[i-1]+scale[i])
            delh = self.H[i]-self.H[i-1]
            p[i] = p[i-1]*np.exp(-delh/sh)

        for i in range(ialt-1,-1,-1):
            sh =  0.5*(scale[i+1]+scale[i])
            delh = self.H[i]-self.H[i+1]
            p[i] = p[i+1]*np.exp(-delh/sh)

        self.edit_P(p)


    def adjust_hydrostatH(self):
        """
        Subroutine to rescale the heights of a H/P/T profile according to
        the hydrostatic equation above and below the level where height=0.
        """   

        #First find the level closest to the 0m altitude
        alt0,ialt = find_nearest(self.H,0.0)
        if ( (alt0>0.0) & (ialt>0)):
            ialt = ialt -1


        xdepth = 100.
        while xdepth>1:  

            h = np.zeros(self.NP)
            p = np.zeros(self.NP)
            h[:] = self.H
            p[:] = self.P
        
            #Calculating the atmospheric depth
            atdepth = h[self.NP-1] - h[0]

            #Calculate the gravity at each altitude level
            self.calc_grav()

            #Calculate the scale height
            R = const["R"]
            scale = R * self.T / (self.MOLWT * self.GRAV)   #scale height (m)

            p[:] = self.P
            if ((ialt>0) & (ialt<self.NP-1)):
                h[ialt] = 0.0

            nupper = self.NP - ialt - 1
            for i in range(ialt+1,self.NP):
                sh = 0.5 * (scale[i-1] + scale[i])
                #self.H[i] = self.H[i-1] - sh * np.log(self.P[i]/self.P[i-1])
                h[i] = h[i-1] - sh * np.log(p[i]/p[i-1])

            for i in range(ialt-1,-1,-1):
                sh = 0.5 * (scale[i+1] + scale[i])
                #self.H[i] = self.H[i+1] - sh * np.log(self.P[i]/self.P[i+1])  
                h[i] = h[i+1] - sh * np.log(p[i]/p[i+1]) 

            #atdepth1 = self.H[self.NP-1] - self.H[0]
            atdepth1 = h[self.NP-1] - h[0]

            xdepth = 100.*abs((atdepth1-atdepth)/atdepth)

            self.H = h[:]

            #Re-Calculate the gravity at each altitude level
            self.calc_grav()  


    def add_gas(self,gasID,isoID,vmr):
        """
        Subroutine to add a gas into the reference atmosphere
            gasID :: Radtran ID of the gas
            isoID :: Radtran isotopologue ID of the gas
            vmr(NP) :: Volume mixing ratio of the gas at each altitude level
        """  

        ngas = self.NVMR + 1
        if len(vmr)!=self.NP:
            sys.exit('error in Atmosphere.add_gas() :: Number of altitude levels in vmr must be the same as in Atmosphere')
        else:
            vmr1 = np.zeros([self.NP,ngas])
            gasID1 = np.zeros(ngas,dtype='int32')
            isoID1 = np.zeros(ngas,dtype='int32')
            vmr1[:,0:self.NVMR] = self.VMR
            vmr1[:,ngas-1] = vmr[:]
            gasID1[0:self.NVMR] = self.ID
            isoID1[0:self.NVMR] = self.ISO
            gasID1[ngas-1] = gasID
            isoID1[ngas-1] = isoID
            self.NVMR = ngas
            self.ID = gasID1
            self.ISO = isoID1
            self.edit_VMR(vmr1)


    def remove_gas(self,gasID,isoID):
        """
        Subroutine to remove a gas from the reference atmosphere
            gasID :: Radtran ID of the gas
            isoID :: Radtran isotopologue ID of the gas
        """  

        igas = np.where( (self.ID==gasID) & (self.ISO==isoID) )
        igas = igas[0]
        if len(igas)==0:
            sys.exit('error in Atmosphere.remove_gas() :: Gas ID and Iso ID not found in reference atmosphere')

        ngas = self.NVMR - 1
        vmr1 = np.zeros([self.NP,ngas])
        gasID1 = np.zeros(ngas,dtype='int32')
        isoID1 = np.zeros(ngas,dtype='int32')
        ix = 0
        for i in range(self.NVMR):
            if i==igas:
                pass
            else:
                vmr1[:,ix] = self.VMR[:,i]
                gasID1[ix] = self.ID[i]
                isoID1[ix] = self.ISO[i]
                ix = ix + 1

        self.NVMR = ngas
        self.ID = gasID1
        self.ISO = isoID1
        self.edit_VMR(vmr1)

    def update_gas(self,gasID,isoID,vmr):
        """
        Subroutine to update a gas into the reference atmosphere
            gasID :: Radtran ID of the gas
            isoID :: Radtran isotopologue ID of the gas
            vmr(NP) :: Volume mixing ratio of the gas at each altitude level
        """ 

        igas = np.where( (self.ID==gasID) & (self.ISO==isoID) )
        igas = igas[0]
        if len(igas)==0:
            sys.exit('error in Atmosphere.update_gas() :: Gas ID and Iso ID not found in reference atmosphere')

        if len(vmr)!=self.NP:
            sys.exit('error in Atmosphere.update_gas() :: Number of altitude levels in vmr must be the same as in Atmosphere')

        vmr1 = np.zeros([self.NP,self.VMR])
        vmr1[:,:] = self.VMR
        vmr1[:,igas] = vmr[:]
        self.edit_VMR(vmr1)

    def write_to_file(self):
        """
        Write the current profile to a runname.prf file in Nemesis format.
        """
        self.check()
        f = open('{}.prf'.format(self.runname),'w')
        f.write('{}\n'.format(self.AMFORM))
        f.write('{:<10} {:<10} {:<10} {:<10}\n'
                .format(self.IPLANET, self.LATITUDE, self.NP, self.NVMR))
        for i in range(self.NVMR):
            f.write('{:<5} {:<5}\n'.format(self.ID[i], self.ISO[i]))
        f.write('{:<15} {:<15} {:<15} '
                .format('H(km)', 'press(atm)', 'temp(K)'))
        for i in range(self.NVMR):
            f.write('{:<15} '.format('VMR gas{}'.format(i+1)))
        for i in range(self.NP):
            f.write('\n{:<15.3f} {:<15.5E} {:<15.3f} '
                    .format(self.H[i]*1e-3, self.P[i]/101325, self.T[i]))
            for j in range(self.NVMR):
                f.write('{:<15.5E} '.format(self.VMR[i][j]))
        f.close()


    def check(self):
        assert isinstance(self.H, np.ndarray),\
            'Need to input height profile'
        assert isinstance(self.P, np.ndarray),\
            'Need to input pressure profile'
        assert isinstance(self.T, np.ndarray),\
            'Need to input temperature profile'
        assert isinstance(self.VMR, np.ndarray),\
            'Need to input VMR profile'
        return True


    def read_ref(self,runname):
        """
        Fills the parameters of the Atmospheric class by reading the .ref file
        """

        #Opening file
        f = open(runname+'.ref','r')
    
        #Reading first and second lines
        
        tmp = np.fromfile(f,sep=' ',count=1,dtype='int')
        amform = int(tmp[0])
        tmp = np.fromfile(f,sep=' ',count=1,dtype='int')
    
        #Reading third line
        tmp = f.readline().split()
        nplanet = int(tmp[0])
        xlat = float(tmp[1])
        npro = int(tmp[2])
        ngas = int(tmp[3])
        if amform==0:
            molwt = float(tmp[4])

        #Reading gases
        gasID = np.zeros(ngas,dtype='int')
        isoID = np.zeros(ngas,dtype='int')
        for i in range(ngas):
            tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
            gasID[i] = int(tmp[0])
            isoID[i] = int(tmp[1])
    
        #Reading profiles
        height = np.zeros(npro)
        press = np.zeros(npro)
        temp = np.zeros(npro)
        vmr = np.zeros([npro,ngas])
        s = f.readline().split()
        for i in range(npro):
            tmp = np.fromfile(f,sep=' ',count=ngas+3,dtype='float')
            height[i] = float(tmp[0])
            press[i] = float(tmp[1])
            temp[i] = float(tmp[2])
            for j in range(ngas):
                vmr[i,j] = float(tmp[3+j])

        #Storing the results into the atmospheric class
        self.NP = npro
        self.NVMR = ngas
        self.ID = gasID
        self.ISO = isoID
        self.IPLANET = nplanet
        self.LATITUDE = xlat
        self.AMFORM = amform
        self.edit_H(height*1.0e3)
        self.edit_P(press*101325.)
        self.edit_T(temp)
        self.edit_VMR(vmr)
        self.runname = runname

        if ( (self.AMFORM==1) or (self.AMFORM==2) ):
            self.calc_molwt()
        else:
            molwt1 = np.zeros(npro)
            molwt1[:] = molwt
            self.MOLWT = molwt1 / 1000.   #kg/m3

        self.calc_grav()


    def write_ref(self,runname):
        """
        Write the current atmospheric profiles into the .ref file
        """

        fref = open(runname+'.ref','w')
        fref.write('\t %i \n' % (self.AMFORM)) 
        nlat = 1    #Would need to be updated to include more latitudes
        fref.write('\t %i \n' % (nlat))
    
        if self.AMFORM==0:
            fref.write('\t %i \t %7.4f \t %i \t %i \t %7.4f \n' % (self.IPLANET,self.LATITUDE,self.NP,self.NVMR,self.MOLWT[0]))
        else:
            fref.write('\t %i \t %7.4f \t %i \t %i \n' % (self.IPLANET,self.LATITUDE,self.NP,self.NVMR))

        gasname = [''] * self.NVMR
        header = [''] * (3+self.NVMR)
        header[0] = 'height(km)'
        header[1] = 'press(atm)'
        header[2] = 'temp(K)  '
        str1 = header[0]+'\t'+header[1]+'\t'+header[2]
        for i in range(self.NVMR):
            fref.write('\t %i \t %i\n' % (self.ID[i],self.ISO[i]))
            strgas = 'GAS'+str(i+1)+'_vmr'
            str1 = str1+'\t'+strgas

        fref.write(str1+'\n')

        for i in range(self.NP):
            str1 = str('{0:7.3f}'.format(self.H[i]/1.0e3))+'\t'+str('{0:7.6e}'.format(self.P[i]/101325.))+'\t'+str('{0:7.4f}'.format(self.T[i]))
            for j in range(self.NVMR):
                str1 = str1+'\t'+str('{0:7.6e}'.format(self.VMR[i,j])) 
            fref.write(str1+'\n')
 
        fref.close()

    def plot_Atm(self):

        """
        Makes a summary plot of the current atmospheric profiles
        """

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(10,4))

        ax1.semilogx(self.P/101325.,self.H/1.0e3,c='black')
        ax2.plot(self.T,self.H/1.0e3,c='black')
        for i in range(self.NVMR):
            label1 = gas_info[str(self.ID[i])]['name']
            if self.ISO[i]!=0:
                label1 = label1+' ('+str(self.ISO[i])+')'
            ax3.semilogx(self.VMR[:,i],self.H/1.0e3,label=label1)
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax3.set_xlabel('Volume mixing ratio')
        plt.subplots_adjust(left=0.08,bottom=0.12,right=0.88,top=0.96,wspace=0.16,hspace=0.20)
        legend = ax3.legend(bbox_to_anchor=(1.01, 1.02))
        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.show()