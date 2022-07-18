from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import os,sys

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

Model variables Class.
"""

class Variables_0:

    def __init__(self, NVAR=2, NPARAM=10, NX=10, JPRE=-1, JTAN=-1, JSURF=-1, JALB=-1, JXSC=-1, JRAD=-1, JLOGG=-1, JFRAC=-1):

        """
        Inputs
        ------
        @param NVAR: int,
            Number of model variables to be included
        @param NPARAM: int,
            Number of extra parameters needed to implement the different models       
        @param NX: int,
            Number of points in the state vector
        @param JPRE: int,
            Position of ref. tangent pressure in state vector (if included)
        @param JTAN: int,
            Position of tangent altitude correction in state vector (if included)
        @param JSURF: int,
            Position of surface temperature in state vector (if included)
        @param JALB: int,
            Position of start of surface albedo spectrum in state vector (if included)
        @param JXSC: int,
            Position of start of x-section spectrum in state vector (if included)
        @param JRAD: int,
            Position of radius of the planet in state vector (if included)
        @param JLOGG: int,
            Position of surface log_10(g) of planet in state vector (if included)     
         @param JFRAC: int,
            Position of fractional coverage in state vector (if included)  

        Attributes
        ----------
        @attribute VARIDENT: 2D array
            Variable ID
        @attribute VARPARAM: 2D array
            Extra parameters needed to implement the parameterisation
        @attribute NXVAR: 1D array
            Number of points in state vector associated with each variable
        @attribute XA: 1D array
            A priori State vector
        @attribute SA: 2D array
            A priori Covariance matrix of the state vector
        @attribute XN: 1D array
            State vector
        @attribute SX: 2D array
            Covariance matrix of the state vector
        @attribute LX: 1D array
            Flag indicating whether the elements of the state vector are carried in log-scale
        @attribute FIX: 1D array
            Flag indicating whether the elements of the state vector must be fixed
        @attribute NUM: 1D array
            Flag indicating how the gradients with respect to a particular element of the state vector must be computed
            (0) Gradients are computed analytically inside CIRSradg (Atmospheric gradients or Surface temperature) or subspecretg (Others)
            (1) Gradients are computed numerically 
        @attribute DSTEP: 1D array
            For the elements of the state vector whose derivative is being calculated numerically, this array indicates
            the step in the function to be used to calculate the numerical derivative (f' = (f(x+h) - f(x)) / h ). 
            If not specified, this step is assumed to be 5% of the value (h = 0.05*x)

        Methods
        -------
        Variables_0.edit_VARIDENT()
        Variables_0.edit_VARPARAM()
        Variables_0.calc_NXVAR()
        Variables_0.edit_XA()
        Variables_0.edit_XN()
        Variables_0.edit_LX()
        Variables_0.calc_FIX()
        Variables_0.edit_SA()
        Variables_0.edit_SX()
        """

        #Input parameters
        self.NVAR = NVAR
        self.NPARAM = NPARAM
        self.NX = NX
        self.JPRE = JPRE
        self.JTAN = JTAN
        self.JSURF = JSURF
        self.JALB = JALB
        self.JXSC = JXSC
        self.JRAD = JRAD
        self.JLOGG = JLOGG
        self.JFRAC = JFRAC

        # Input the following profiles using the edit_ methods.
        self.VARIDENT = None # np.zeros(NVAR,3)
        self.VARPARAM = None # np.zeros(NVAR,NPARAM)
        self.NXVAR =  None # np.zeros(NX)
        self.XN = None # np.zeros(NX)
        self.LX = None # np.zeros(NX)
        self.FIX =  None # np.zeros(NX)
        self.SX = None # np.zeros((NX, NX))
        self.NUM = None #np.zeros(NX)
        self.DSTEP = None #np.zeros(NX)

    def edit_VARIDENT(self, VARIDENT_array):
        """
        Edit the Variable IDs
        @param VARIDENT_array: 2D array
            Parameter IDs defining the parameterisation
        """
        VARIDENT_array = np.array(VARIDENT_array)
        #assert len(VARIDENT_array[:,0]) == self.NVAR, 'VARIDENT should have (NVAR,3) elements'
        #assert len(VARIDENT_array[0,:]) == 3, 'VARIDENT should have (NVAR,3) elements'
        self.VARIDENT = VARIDENT_array

    def edit_VARPARAM(self, VARPARAM_array):
        """
        Edit the extra parameters needed to implement the parameterisations
        @param VARPARAM_array: 2D array
            Extra parameters defining the model
        """
        VARPARAM_array = np.array(VARPARAM_array)
        #assert len(VARPARAM_array[:,0]) == self.NVAR, 'VARPARAM should have (NVAR,NPARAM) elements'
        #assert len(VARPARAM_array[0,:]) == self.NPARAM, 'VARPARAM should have (NVAR,NPARAM) elements'
        self.VARPARAM = VARPARAM_array

    def calc_NXVAR(self, NPRO):
        """
        Calculate the array defining the number of parameters in the state 
        vector associated with each model
        @param NXVAR_array: 1D array
            Number of parameters in the state vector associated with each model
        """

        nxvar = np.zeros(self.NVAR,dtype='int32')

        if self.NVAR==1:
            if len(self.VARIDENT.shape)==1:
                imod = self.VARIDENT[2]
                ipar = self.VARPARAM[0]
                ipar2 = self.VARPARAM[1]
            else:
                imod = self.VARIDENT[0,2]
                ipar = self.VARPARAM[0,0]
                ipar2 = self.VARPARAM[0,1]

        for i in range(self.NVAR):

            if self.NVAR>1:
                imod = self.VARIDENT[i,2]
                ipar = self.VARPARAM[i,0]
                ipar2 = self.VARPARAM[i,1]

            if imod == -1:
                nxvar[i] = NPRO
            elif imod == 0:
                nxvar[i] = NPRO
            elif imod == 1:
                nxvar[i] = 2
            elif imod == 2:
                nxvar[i] = 1
            elif imod == 3:
                nxvar[i] = 1
            elif imod == 4:
                nxvar[i] = 3
            elif imod == 5:
                nxvar[i] = 1
            elif imod == 6:
                nxvar[i] = 2
            elif imod == 7:
                nxvar[i] = 2
            elif imod == 8:
                nxvar[i] = 3
            elif imod == 9:
                nxvar[i] = 3
            elif imod == 10:
                nxvar[i] = 4
            elif imod == 11:
                nxvar[i] = 2
            elif imod == 12:
                nxvar[i] = 3
            elif imod == 13:
                nxvar[i] = 3
            elif imod == 14:
                nxvar[i] = 3
            elif imod == 15:
                nxvar[i] = 3
            elif imod == 16:
                nxvar[i] = 4
            elif imod == 17:
                nxvar[i] = 2
            elif imod == 18:
                nxvar[i] = 2
            elif imod == 19:
                nxvar[i] = 4
            elif imod == 20:
                nxvar[i] = 2
            elif imod == 21:
                nxvar[i] = 2
            elif imod == 22:
                nxvar[i] = 5
            elif imod == 23:
                nxvar[i] = 4
            elif imod == 24:
                nxvar[i] = 3
            elif imod == 25:
                nxvar[i] = int(ipar)
            elif imod == 26:
                nxvar[i] = 4
            elif imod == 27:
                nxvar[i] = 3
            elif imod == 28:
                nxvar[i] = 1
            elif imod == 49:
                nxvar[i] = NPRO
            elif imod == 50:
                nxvar[i] = NPRO
            elif imod == 228:
                nxvar[i] = 8
            elif imod == 229:
                nxvar[i] = 7
            elif imod == 230:
                nxvar[i] = 7*int(ipar)
            elif imod == 231:
                nxvar[i] = int(ipar)*int(ipar2+1)
            elif imod == 232:
                nxvar[i] = 2*int(ipar)
            elif imod == 233:
                nxvar[i] = 3*int(ipar)
            elif imod == 444:
                nxvar[i] = 1 + 1 + int(ipar)
            elif imod == 446:
                nxvar[i] = 2*int(ipar)
            elif imod == 666:
                nxvar[i] = 1
            elif imod == 667:
                nxvar[i] = 1
            elif imod == 887:
                nxvar[i] = int(ipar)
            elif imod == 998:
                nxvar[i] = int(ipar)
            elif imod == 999:
                nxvar[i] = 1
            else:
                sys.exit('error :: varID not included in calc_NXVAR()')

        self.NXVAR = nxvar

    def calc_DSTEP(self):
        """
        Calculate the step size to be used for the calculation of the numerical derivatives
        f' =  ( f(x+h) - f(x) ) / h
        """

        #Generally, we use h = 0.05*x
        dstep = np.zeros(self.NX)
        dstep[:] = self.XN * 0.05

        #Changing the step size for certain parameterisations
        ix = 0
        for i in range(self.NVAR):

            if self.NVAR>1:
                imod = self.VARIDENT[i,2]
                ipar = self.VARPARAM[i,0]
            else:
                imod = self.VARIDENT[0,2]
                ipar = self.VARPARAM[0,0]

            if imod == 228:
                
                #V0,C0,C1,C2,P0,P1,P2,P3
                dstep[ix] = 0.5 * np.sqrt( self.SA[ix,ix] )
                dstep[ix+1] = 0.5 * np.sqrt( self.SA[ix+1,ix+1] )
                dstep[ix+2] = 0.5 * np.sqrt( self.SA[ix+2,ix+2] )
                dstep[ix+3] = 0.5 * np.sqrt( self.SA[ix+3,ix+3] )

            ix = ix + self.NXVAR[i]

        self.DSTEP = dstep


    def edit_XA(self, XA_array):
        """
        Edit the State Vector.
        @param XA_array: 1D array
            Parameters defining the a priori state vector
        """
        XA_array = np.array(XA_array)
        assert len(XA_array) == self.NX, 'XA should have NX elements'
        self.XA = XA_array

    def edit_XN(self, XN_array):
        """
        Edit the State Vector.
        @param XN_array: 1D array
            Parameters defining the state vector
        """
        XN_array = np.array(XN_array)
        assert len(XN_array) == self.NX, 'XN should have NX elements'
        self.XN = XN_array

    def edit_LX(self, LX_array):
        """
        Edit the the flag indicating if the elements are in log-scale
        @param LX_array: 1D array
            Flag indicating whether a particular element of the state 
            vector is in log-scale (1) or not (0)
        """
        LX_array = np.array(LX_array,dtype='int32')
        assert len(LX_array) == self.NX, 'LX should have NX elements'
        self.LX = LX_array  

    def edit_FIX(self, FIX_array):
        """
        Edit the the flag indicating if the elements are to be fixed
        @param FIX_array: 1D array
            Flag indicating whether a particular element of the state 
            vector is fixed (1) or not (0)
        """
        FIX_array = np.array(FIX_array,dtype='int32')
        assert len(FIX_array) == self.NX, 'FIX should have NX elements'
        self.FIX = FIX_array 

    def edit_SA(self, SA_array):
        """
        Edit the a priori covariance matrix
        @param SA_array: 2D array
            A priori covariance matrix
        """
        SA_array = np.array(SA_array)
        assert len(SA_array[:,0]) == self.NX, 'SA should have (NX,NX) elements'
        assert len(SA_array[0,:]) == self.NX, 'SA should have (NX,NX) elements'
        self.SA = SA_array 

    def edit_SX(self, SX_array):
        """
        Edit the state vector covariance matrix
        @param SX_array: 2D array
            State vector covariance matrix
        """
        SX_array = np.array(SX_array)
        assert len(SX_array[:,0]) == self.NX, 'SX should have (NX,NX) elements'
        assert len(SX_array[0,:]) == self.NX, 'SX should have (NX,NX) elements'
        self.SX = SX_array 

    def calc_FIX(self):
        """
        Check if the fractional error on any of the state vector parameters is so small 
        that it must be kept constant in the retrieval
        @param FIX: 1D array
            Flag indicating the elements of the state vector that need to be fixed
        """

        minferr = 1.0e-6  #minimum fractional error to fix variable.

        ifix = np.zeros(self.NX,dtype='int32')    
        for ix in range(self.NX):
            xa1 = self.XA[ix]
            ea1 = np.sqrt(abs(self.SA[ix,ix]))

            if self.LX[ix]==1:
                xa1 = np.exp(xa1)
                ea1 = xa1*ea1

            ferr = abs(ea1/xa1)
            if ferr<=minferr:
                ifix[ix] = 1
                
        self.FIX = ifix

    def read_apr(self,runname,npro):
        """
        Read the .apr file, which contains information about the variables and
        parametrisations that are to be retrieved, as well as their a priori values.
        These parameters are then included in the Variables class.
        
        N.B. In this code, the apriori and retrieved vectors x are usually
        converted to logs, all except for temperature and fractional scale heights
        This is done to reduce instabilities when different parts of the
        vectors and matrices hold vastly different sized properties. e.g.
        cloud x-section and base height.

        @param runname: str
            Name of the Nemesis run
        @param NPRO: int
            Number of altitude levels in the reference atmosphere
        """

        from NemesisPy import Scatter_0

        #Open file
        f = open(runname+'.apr','r')
    
        #Reading header
        s = f.readline().split()
    
        #Reading first line
        s = f.readline().split()
        nvar = int(s[0])
    
        #Initialise some variables
        jsurf = -1
        jalb = -1
        jxsc = -1
        jtan = -1
        jpre = -1
        jrad = -1
        jlogg = -1
        jfrac = -1
        sxminfac = 0.001
        mparam = 200        #Giving big sizes but they will be re-sized
        mx = 2000
        varident = np.zeros([nvar,3],dtype='int')
        varparam = np.zeros([nvar,mparam])
        lx = np.zeros([mx],dtype='int')
        x0 = np.zeros([mx])
        sx = np.zeros([mx,mx])
        inum = np.zeros([mx],dtype='int')

        #Reading data
        ix = 0
    
        for i in range(nvar):
            s = f.readline().split()
            for j in range(3):
                varident[i,j] = int(s[j])

            #Starting different cases
            if varident[i,2] <= 100:    #Parameter must be an atmospheric one

                if varident[i,2] == 0:
#               ********* continuous profile ************************
                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
                    nlevel = int(tmp[0])
                    if nlevel != npro:
                        sys.exit('profiles must be listed on same grid as .prf')
                    clen = float(tmp[1])
                    pref = np.zeros([nlevel])
                    ref = np.zeros([nlevel])
                    eref = np.zeros([nlevel])
                    for j in range(nlevel):
                        tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                        pref[j] = float(tmp[0])
                        ref[j] = float(tmp[1])
                        eref[j] = float(tmp[2])
                    f1.close()

                    if varident[i,0] == 0:  # *** temperature, leave alone ****
                        x0[ix:ix+nlevel] = ref[:]
                        for j in range(nlevel):
                            sx[ix+j,ix+j] = eref[j]**2.
                            if varident[i,1] == -1: #Gradients computed numerically
                                inum[ix+j] = 1

                    else:                   #**** vmr, cloud, para-H2 , fcloud, take logs ***
                        for j in range(nlevel):
                            lx[ix+j] = 1
                            x0[ix+j] = np.log(ref[j])
                            sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

                    #Calculating correlation between levels in continuous profile
                    for j in range(nlevel):
                        for k in range(nlevel):
                            if pref[j] < 0.0:
                                sys.exit('Error in read_apr_nemesis().  A priori file must be on pressure grid')
                        
                            delp = np.log(pref[k])-np.log(pref[j])
                            arg = abs(delp/clen)
                            xfac = np.exp(-arg)
                            if xfac >= sxminfac:
                                sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                                sx[ix+k,ix+j] = sx[ix+j,ix+k]
                        
                    ix = ix + nlevel


                elif varident[i,2] == -1:
#               * continuous cloud, but cloud retrieved as particles/cm3 rather than
#               * particles per gram to decouple it from pressure.
#               ********* continuous particles/cm3 profile ************************
                    if varident[i,0] >= 0:
                        sys.exit('error in read_apr_nemesis :: model -1 type is only for use with aerosols')
        
                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
                    nlevel = int(tmp[0])
                    if nlevel != npro:
                        sys.exit('profiles must be listed on same grid as .prf')
                    clen = float(tmp[1])
                    pref = np.zeros([nlevel])
                    ref = np.zeros([nlevel])
                    eref = np.zeros([nlevel])
                    for j in range(nlevel):
                        tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                        pref[j] = float(tmp[0])
                        ref[j] = float(tmp[1])
                        eref[j] = float(tmp[2])
                    
                        lx[ix+j] = 1
                        x0[ix+j] = np.log(ref[j])
                        sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.
                
                    f1.close()

                    #Calculating correlation between levels in continuous profile
                    for j in range(nlevel):
                        for k in range(nlevel):
                            if pref[j] < 0.0:
                                sys.exit('Error in read_apr_nemesis().  A priori file must be on pressure grid')
                
                            delp = np.log(pref[k])-np.log(pref[j])
                            arg = abs(delp/clen)
                            xfac = np.exp(-arg)
                            if xfac >= sxminfac:
                                sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                                sx[ix+k,ix+j] = sx[ix+j,ix+k]
                
                    ix = ix + nlevel

                elif varident[i,2] == 1:
#               ******** profile held as deep amount, fsh and knee pressure **
#               Read in xdeep,fsh,pknee
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    pknee = float(tmp[0])
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xdeep = float(tmp[0])
                    edeep = float(tmp[1])
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xfsh = float(tmp[0])
                    efsh = float(tmp[1])

                    varparam[i,0] = pknee
    
                    if varident[i,0] == 0:  #Temperature, leave alone
                        x0[ix] = xdeep
                        sx[ix,ix] = edeep**2.
                    else:
                        x0[ix] = np.log(xdeep)
                        sx[ix,ix] = ( edeep/xdeep )**2.
                        lx[ix] = 1
        
                    ix = ix + 1

                    if xfsh > 0.0:
                        x0[ix] = np.log(xfsh)
                        lx[ix] = 1
                        sx[ix,ix] = ( efsh/xfsh  )**2.
                    else:
                        sys.exit('Error in read_apr_nemesis().  xfsh must be > 0')
                
                    ix = ix + 1

                elif varident[i,2] == 2:
#               **** Simple scaling factor of reference profile *******
#               Read in scaling factor

                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = (float(tmp[1]))**2.

                    ix = ix + 1

                elif varident[i,2] == 3:
#               **** Exponential scaling factor of reference profile *******
#               Read in scaling factor
        
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xfac = float(tmp[0])
                    err = float(tmp[1])
        
                    if xfac > 0.0:
                        x0[ix] = np.log(xfac)
                        lx[ix] = 1
                        sx[ix,ix] = ( err/xfac ) **2.
                    else:
                        sys.exit('Error in read_apr_nemesis().  xfac must be > 0')
            
                    ix = ix + 1

                elif varident[i,2] == 4:
#               ******** profile held as deep amount, fsh and VARIABLE knee press
#               Read in xdeep,fsh,pknee
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    pknee = float(tmp[0])
                    eknee = float(tmp[1])
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xdeep = float(tmp[0])
                    edeep = float(tmp[1])
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xfsh = float(tmp[0])
                    efsh = float(tmp[1])

                    if varident[i,0] == 0:  #Temperature, leave alone
                        x0[ix] = xdeep
                        sx[ix,ix] = edeep**2.
                    else:
                        x0[ix] = np.log(xdeep)
                        sx[ix,ix] = ( edeep/xdeep )**2.
                        lx[ix] = 1
                        ix = ix + 1
                
                    if xfsh > 0.0:
                        x0[ix] = np.log(xfsh)
                        lx[ix] = 1
                        sx[ix,ix] = ( efsh/xfsh  )**2.
                    else:
                        sys.exit('Error in read_apr_nemesis().  xfsh must be > 0')
                    ix = ix + 1
                
                    x0[ix] = np.log(pknee)
                    lx[ix] = 1
                    sx[ix,ix] = (eknee/pknee)**2
                    ix = ix + 1


                elif varident[i,2] == 9:
#               ******** cloud profile held as total optical depth plus
#               ******** base height and fractional scale height. Below the knee
#               ******** pressure the profile is set to zero - a simple
#               ******** cloud in other words!
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    hknee = tmp[0]
                    eknee = tmp[1]
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xdeep = tmp[0]
                    edeep = tmp[1]
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xfsh = tmp[0]
                    efsh = tmp[1]

                    if xdeep>0.0:
                        x0[ix] = np.log(xdeep)
                        lx[ix] = 1
                        #inum[ix] = 1
                    else:
                        sys.exit('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

                    err = edeep/xdeep
                    sx[ix,ix] = err**2.

                    ix = ix + 1

                    if xfsh>0.0:
                        x0[ix] = np.log(xfsh)
                        lx[ix] = 1
                        #inum[ix] = 1
                    else:
                        sys.exit('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

                    err = efsh/xfsh
                    sx[ix,ix] = err**2.

                    ix = ix + 1

                    x0[ix] = hknee
                    #inum[ix] = 1
                    sx[ix,ix] = eknee**2.

                    ix = ix + 1

                elif varident[i,2] == 49:
#               ********* continuous profile in linear scale ************************
                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
                    nlevel = int(tmp[0])
                    if nlevel != npro:
                        sys.exit('profiles must be listed on same grid as .prf')
                    clen = float(tmp[1])
                    pref = np.zeros([nlevel])
                    ref = np.zeros([nlevel])
                    eref = np.zeros([nlevel])
                    for j in range(nlevel):
                        tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                        pref[j] = float(tmp[0])
                        ref[j] = float(tmp[1])
                        eref[j] = float(tmp[2])
                    f1.close()

                    #inum[ix:ix+nlevel] = 1
                    x0[ix:ix+nlevel] = ref[:]
                    for j in range(nlevel):
                        sx[ix+j,ix+j] = eref[j]**2.

                    #Calculating correlation between levels in continuous profile
                    for j in range(nlevel):
                        for k in range(nlevel):
                            if pref[j] < 0.0:
                                sys.exit('Error in read_apr_nemesis().  A priori file must be on pressure grid')
                        
                            delp = np.log(pref[k])-np.log(pref[j])
                            arg = abs(delp/clen)
                            xfac = np.exp(-arg)
                            if xfac >= sxminfac:
                                sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                                sx[ix+k,ix+j] = sx[ix+j,ix+k]
                        
                    ix = ix + nlevel



                elif varident[i,2] == 50:
#               ********* continuous profile of a scaling factor ************************
                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
                    nlevel = int(tmp[0])
                    if nlevel != npro:
                        sys.exit('profiles must be listed on same grid as .prf')
                    clen = float(tmp[1])
                    pref = np.zeros([nlevel])
                    ref = np.zeros([nlevel])
                    eref = np.zeros([nlevel])
                    for j in range(nlevel):
                        tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                        pref[j] = float(tmp[0])
                        ref[j] = float(tmp[1])
                        eref[j] = float(tmp[2])
                    f1.close()

                    x0[ix:ix+nlevel] = ref[:]
                    for j in range(nlevel):
                        sx[ix+j,ix+j] = eref[j]**2.

                    #Calculating correlation between levels in continuous profile
                    for j in range(nlevel):
                        for k in range(nlevel):
                            if pref[j] < 0.0:
                                sys.exit('Error in read_apr_nemesis().  A priori file must be on pressure grid')
                        
                            delp = np.log(pref[k])-np.log(pref[j])
                            arg = abs(delp/clen)
                            xfac = np.exp(-arg)
                            if xfac >= sxminfac:
                                sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                                sx[ix+k,ix+j] = sx[ix+j,ix+k]
                        
                    ix = ix + nlevel



            
                else:
                    sys.exit('error in read_apr() :: Variable ID not included in this function')

            else:

                if varident[i,2] == 228:
#               ******** model for retrieving the ILS and Wavelength calibration in ACS MIR solar occultation observations
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #V0
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C0
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C1
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C2
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1

                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P0 - Offset of the second gaussian with respect to the first one (assumed spectrally constant)
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1

                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P1 - FWHM of the main gaussian (assumed to be constant in wavelength units)
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P2 - Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P3 - Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 1
                    ix = ix + 1

                elif varident[i,2] == 229:
#               ******** model for retrieving the ILS in ACS MIR solar occultation observations

                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at lowest wavenumber
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 0
                    ix = ix + 1
        
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at wavenumber in the middle
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 0
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at highest wavenumber
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 0
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Offset of the second gaussian with respect to the first one (assumed spectrally constant)
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 0
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #FWHM of the main gaussian (assumed to be constant in wavelength units)
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 0
                    ix = ix + 1

                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 0
                    ix = ix + 1
                
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
                    x0[ix] = float(tmp[0])
                    sx[ix,ix] = float(tmp[1])**2.
                    lx[ix] = 0
                    inum[ix] = 0
                    ix = ix + 1


                elif varident[i,2] == 230:
#               ******** model for retrieving multiple ILS (different spectral windows) in ACS MIR solar occultation observations

                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    s = f1.readline().split()
                    nwindows = int(s[0])
                    varparam[i,0] = nwindows
                    liml = np.zeros(nwindows)
                    limh = np.zeros(nwindows)
                    for iwin in range(nwindows):
                        s = f1.readline().split()
                        liml[iwin] = float(s[0])
                        limh[iwin] = float(s[1])
                        varparam[i,2*iwin+1] = liml[iwin]
                        varparam[i,2*iwin+2] = limh[iwin]

                    par = np.zeros((7,nwindows))
                    parerr = np.zeros((7,nwindows))
                    for iw in range(nwindows):
                        for j in range(7):
                            s = f1.readline().split()
                            par[j,iw] = float(s[0])
                            parerr[j,iw] = float(s[1])
                            x0[ix] = par[j,iw]
                            sx[ix,ix] = (parerr[j,iw])**2.
                            inum[ix] = 0
                            ix = ix + 1

                elif varident[i,2] == 231:
#               ******** Continuum addition to transmission spectra using a varying scaling factor (following polynomial of degree N)

                    #The computed spectra is multiplied by R = R0 * (T0 + POL)
                    #Where the polynomial function POL depends on the wavelength given by:
                    # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    tmp = np.fromfile(f1,sep=' ',count=2,dtype='int')
                    nlevel = int(tmp[0])
                    ndegree = int(tmp[1])
                    varparam[i,0] = nlevel
                    varparam[i,1] = ndegree
                    for ilevel in range(nlevel):
                        tmp = np.fromfile(f1,sep=' ',count=2*(ndegree+1),dtype='float')
                        for ic in range(ndegree+1):
                            r0 = float(tmp[2*ic])
                            err0 = float(tmp[2*ic+1])
                            x0[ix] = r0
                            sx[ix,ix] = (err0)**2.
                            inum[ix] = 0
                            ix = ix + 1

                elif varident[i,2] == 232:
#               ******** Continuum addition to transmission spectra using the Angstrom coefficient

                    #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
                    #Where the parameters to fit are TAU0 and ALPHA

                    s = f.readline().split()
                    wavenorm = float(s[0])                    

                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
                    nlevel = int(tmp[0])
                    varparam[i,0] = nlevel
                    varparam[i,1] = wavenorm
                    for ilevel in range(nlevel):
                        tmp = np.fromfile(f1,sep=' ',count=4,dtype='float')
                        r0 = float(tmp[0])   #Opacity level at wavenorm
                        err0 = float(tmp[1])
                        r1 = float(tmp[2])   #Angstrom coefficient
                        err1 = float(tmp[3])
                        x0[ix] = r0
                        sx[ix,ix] = (err0)**2.
                        x0[ix+1] = r1
                        sx[ix+1,ix+1] = err1**2.
                        inum[ix] = 0
                        inum[ix+1] = 0                        
                        ix = ix + 2

                elif varident[i,2] == 233:
#               ******** Aerosol opacity modelled with a variable angstrom coefficient. Applicable to transmission spectra.

                    #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
                    #Where the aerosol opacity is modelled following

                    # np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

                    #The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
                    #233 converges to model 232 when a2=0.                  

                    #Reading the file where the a priori parameters are stored
                    s = f.readline().split()
                    f1 = open(s[0],'r')
                    tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
                    nlevel = int(tmp[0])
                    varparam[i,0] = nlevel
                    for ilevel in range(nlevel):
                        tmp = np.fromfile(f1,sep=' ',count=6,dtype='float')
                        a0 = float(tmp[0])   #A0
                        err0 = float(tmp[1])
                        a1 = float(tmp[2])   #A1
                        err1 = float(tmp[3])
                        a2 = float(tmp[4])   #A2
                        err2 = float(tmp[5])
                        x0[ix] = a0
                        sx[ix,ix] = (err0)**2.
                        x0[ix+1] = a1
                        sx[ix+1,ix+1] = err1**2.
                        x0[ix+2] = a2
                        sx[ix+2,ix+2] = err2**2.
                        inum[ix] = 0
                        inum[ix+1] = 0    
                        inum[ix+2] = 0                  
                        ix = ix + 3

                elif varident[i,2] == 447:
#               ******** model for retrieving an aerosol opacity profile + Angstrom coefficient profile

                    #Read the number of altitude levels (and therefore number of aerosol populations)
                    #Wavelength at which the opacity must be normalised
                    s = f.readline().split()
                    nlevel = int(s[0])
                    WaveNorm = float(s[2])

                    varparam[i,0] = nlevel
                    varparam[i,2] = WaveNorm

                    if nlevel!=npro:
                        sys.exit('error in model 446 : The number of aerosol levels in the input file must be the same as in the .ref file')

                    


                elif varident[i,2] == 446:
#               ******** model for retrieving an aerosol density profile + aerosol particle size (log-normal distribution)

                    #This model requires one aerosol population for altitude level in order to have the correct extinction
                    #coefficients and single scattering albedo at all altitudes

                    #Read the number of altitude levels (and therefore number of aerosol populations)
                    #Read the aerosol id (from the aerosol dictionary) from which to take the refractive index
                    #Read the sigma of the lognormal distribution (assumed to be fixed)
                    #Read the 
                    s = f.readline().split()
                    nlevel = int(s[0])
                    aerosol_id = int(s[1])
                    sigma_dist = float(s[2])
                    idust0 = int(s[3])
                    WaveNorm = float(s[4])
                    
                    varparam[i,0] = nlevel
                    varparam[i,1] = aerosol_id
                    varparam[i,2] = sigma_dist
                    varparam[i,3] = idust0
                    varparam[i,4] = WaveNorm

                    if nlevel!=npro:
                        sys.exit('error in model 446 : The number of aerosol levels in the input file must be the same as in the .ref file')

                    #Read the name of the file where the a priori vertical profiles are defined
                    s = f.readline().split()
                    fnamex = s[0]

                    #Reading the file and the profiles
                    faero = open(fnamex,'r')
                    s = faero.readline().split()
                    clen_dens = float(s[0])
                    clen_rsize = float(s[1])

                    pref = np.zeros(nlevel)
                    aerodens = np.zeros(nlevel)
                    aerodenserr = np.zeros(nlevel)
                    rsize = np.zeros(nlevel)
                    rsizeerr = np.zeros(nlevel)
                    for ih in range(nlevel):
                        s = faero.readline().split()
                        pref[ih] = float(s[0])
                        aerodens[ih] = float(s[1])
                        aerodenserr[ih] = float(s[2])
                        rsize[ih] = float(s[3])
                        rsizeerr[ih] = float(s[4])
                    
                    faero.close()

                    #Filling the state vector and a priori covariance matrix with the dust abundance
                    for ih in range(nlevel):
                        lx[ix+ih] = 1
                        x0[ix+ih] = np.log(aerodens[ih])
                        sx[ix+ih,ix+ih] = ( aerodenserr[ih]/aerodens[ih]  )**2.

                    for j in range(nlevel):
                        for k in range(nlevel):
                            if pref[j] < 0.0:
                                sys.exit('Error in read_apr_nemesis().  A priori file must be on pressure grid')
                        
                            delp = np.log(pref[k])-np.log(pref[j])
                            arg = abs(delp/clen_dens)
                            xfac = np.exp(-arg)
                            if xfac >= sxminfac:
                                sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                                sx[ix+k,ix+j] = sx[ix+j,ix+k]

                    ix = ix + nlevel

                    #Filling the state vector and a priori covariance matrix with the dust particle size
                    for ih in range(nlevel):
                        #x0[ix+ih] = rsize[ih]
                        lx[ix+ih] = 1
                        x0[ix+ih] = np.log(rsize[ih])
                        inum[ix+ih] = 1
                        sx[ix+ih,ix+ih] = ( rsizeerr[ih]/rsize[ih]  )**2.
                        #sx[ix+ih,ix+ih] = (rsizeerr[ih])**2.

                    for j in range(nlevel):
                        for k in range(nlevel):
                            if pref[j] < 0.0:
                                sys.exit('Error in read_apr_nemesis().  A priori file must be on pressure grid')
                        
                            delp = np.log(pref[k])-np.log(pref[j])
                            arg = abs(delp/clen_rsize)
                            xfac = np.exp(-arg)
                            if xfac >= sxminfac:
                                sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                                sx[ix+k,ix+j] = sx[ix+j,ix+k]

                    ix = ix + nlevel

                elif varident[i,2] == 666:
#               ******** pressure at given altitude
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    htan = float(tmp[0])
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    ptan = float(tmp[0])
                    ptanerr = float(tmp[1])
                    varparam[i,0] = htan
                    if ptan>0.0:
                        x0[ix] = np.log(ptan)
                        lx[ix] = 1
                        inum[ix] = 1
                    else:
                        sys.exit('error in read_apr_nemesis() :: pressure must be > 0')
                
                    sx[ix,ix] = (ptanerr/ptan)**2.
                    jpre = ix
                
                    ix = ix + 1

                elif varident[i,2] == 667:
#               ******** dilution factor to account for thermal gradients thorughout exoplanet
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                    xfac = float(tmp[0])
                    xfacerr = float(tmp[1])
                    x0[ix] = xfac
                    inum[ix] = 0 
                    sx[ix,ix] = xfacerr**2.
                    ix = ix + 1

                elif varident[i,2] == 887:
#               ******** Cloud x-section spectrum

                    #Read in number of points, cloud id, and correlation between elements.
                    s = f.readline().split()
                    nwv = int(s[0]) #number of spectral points (must be the same as in .xsc)
                    icloud = int(s[1])  #aerosol ID
                    clen = float(s[2])  #Correlation length (in wavelengths/wavenumbers)

                    varparam[i,0] = nwv
                    varparam[i,1] = icloud

                    #Read the wavelengths and the extinction cross-section value and error
                    wv = np.zeros(nwv)
                    xsc = np.zeros(nwv)
                    err = np.zeros(nwv)
                    for iw in range(nwv):
                        s = f.readline().split()
                        wv[iw] = float(s[0])
                        xsc[iw] = float(s[1])
                        err[iw] = float(s[2])
                        if xsc[iw]<=0.0:
                            sys.exit('error in read_apr :: Cross-section in model 887 must be greater than 0')

                    #It is important to check that the wavelengths in .apr and in .xsc are the same
                    Aero0 = Scatter_0()
                    Aero0.read_xsc(runname)
                    for iw in range(Aero0.NWAVE):
                        if (wv[iw]-Aero0.WAVE[iw])>0.01:
                            sys.exit('error in read_apr :: Number of wavelengths in model 887 must be the same as in .xsc')

                    #Including the parameters in state vector and covariance matrix
                    for j in range(nwv):
                        x0[ix+j] = np.log(xsc[j])
                        lx[ix+j] = 1
                        inum[ix+j] = 1
                        sx[ix+j,ix+j] = (err[j]/xsc[j])**2.

                    for j in range(nwv):
                        for k in range(nwv):
                            delv = wv[j] - wv[k]
                            arg = abs(delv/clen)
                            xfac = np.exp(-arg)
                            if xfac>0.001:
                                sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                                sx[ix+k,ix+j] = sx[ix+j,ix+k]

                    jxsc = ix

                    ix = ix + nwv

                elif varident[i,2] == 998:
#               ******** map of surface temperatures 
                    ipfile = f.readline().split()
                    ipfile = ipfile[0]
                    ftsurf = open(ipfile,'r')
                    s = ftsurf.readline().split()
                    ntsurf = int(s[0])
                    varparam[i,0] = ntsurf

                    iparam = 1
                    for itsurf in range(ntsurf):
                        s = ftsurf.readline().split()
                        latsurf = float(s[0])
                        lonsurf = float(s[1])
                        varparam[i,iparam] = latsurf
                        varparam[i,iparam+1] = lonsurf
                        iparam = iparam + 1
                        s = ftsurf.readline().split()
                        r0 = float(s[0])
                        err = float(s[1])
                        x0[ix] = r0
                        sx[ix,ix] = err**2.0
                        inum[ix] = 1
                        ix = ix + 1

                elif varident[i,2] == 999:
#               ******** surface temperature
                    s = f.readline().split()
                    tsurf = float(s[0])
                    esurf = float(s[1])
                    x0[ix] = tsurf
                    sx[ix,ix] = esurf**2.
                    inum[ix] = 0
                    jsurf = ix
            
                    ix = ix + 1



        f.close()

        nx = ix
        lx1 = np.zeros(nx,dtype='int32')
        inum1 = np.zeros(nx,dtype='int32')
        xa = np.zeros(nx)
        sa = np.zeros([nx,nx])
        lx1[0:nx] = lx[0:nx]
        inum1[0:nx] = inum[0:nx]
        xa[0:nx] = x0[0:nx]
        sa[0:nx,0:nx] = sx[0:nx,0:nx]

        self.NVAR=nvar
        self.NPARAM=mparam
        self.edit_VARIDENT(varident)
        self.edit_VARPARAM(varparam)
        self.calc_NXVAR(npro)
        self.JPRE, self.JTAN, self.JSURF, self.JALB, self.JXSC, self.JLOGG, self.JFRAC = jpre, jtan, jsurf, jalb, jxsc, jlogg, jfrac
        self.NX = nx
        self.edit_XA(xa)
        self.edit_XN(xa)
        self.edit_SA(sa)
        self.edit_LX(lx1)
        self.NUM = inum1
        self.calc_DSTEP()
        self.calc_FIX()