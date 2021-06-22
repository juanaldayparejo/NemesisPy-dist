# NAME:
#       files.py (nemesislib)
#
# DESCRIPTION:
#
#	This library contains functions to read and write files that are formatted as 
#	required by the NEMESIS radiative transfer code         
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
import nemesislib.utils as utils
import nemesislib.spectroscopy as spec
from NemesisPy.Profile import *
from NemesisPy.Data import *
from NemesisPy.Models.Models import *

###############################################################################################

def read_mre(runname,MakePlot=False):

    """
    FUNCTION NAME : read_mre()

    DESCRIPTION : Reads the .mre file from a Nemesis run

    INPUTS :
    
        runname :: Name of the Nemesis run

    OPTIONAL INPUTS:
    
        MakePlot : If True, a summary plot is made
            
    OUTPUTS : 

        lat :: Latitude (degrees)
        lon :: Longitude (degrees)
        ngeom :: Number of geometries in the observation
        nconv :: Number of points in the measurement vector for each geometry (assuming they all have the same number of points)
        wave(nconv,ngeom) :: Wavelength/wavenumber of each point in the measurement vector
        specret(nconv,ngeom) :: Retrieved spectrum for each of the geometries
        specmeas(nconv,ngeom) :: Measured spectrum for each of the geometries
        specerrmeas(nconv,ngeom) :: Error in the measured spectrum for each of the geometries
        nx :: Number of points in the state vector
        varident(nvar,3) :: Retrieved variable ID, as defined in Nemesis manual
        nxvar :: Number of points in the state vector associated with each retrieved variable
        varparam(nvar,5) :: Extra parameters containing information about how to read the retrieved variables
        aprprof(nx,nvar) :: A priori profile for each variable in the state vector
        aprerr(nx,nvar) :: Error in the a priori profile for each variable in the state vector
        retprof(nx,nvar) :: Retrieved profile for each variable in the state vector
        reterr(nx,nvar) :: Error in the retrieved profile for each variable in the state vector

    CALLING SEQUENCE:

        lat,lon,ngeom,ny,wave,specret,specmeas,specerrmeas,nx,Var,aprprof,aprerr,retprof,reterr = read_mre(runname)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)

    """

    #Opening .ref file for getting number of altitude levels
    Atm = read_ref(runname)
    
    #Opening file
    f = open(runname+'.mre','r')

    #Reading first three lines
    tmp = np.fromfile(f,sep=' ',count=1,dtype='int')
    s = f.readline().split()
    nspec = int(tmp[0])
    tmp = np.fromfile(f,sep=' ',count=5,dtype='float')
    s = f.readline().split()
    ispec = int(tmp[0])
    ngeom = int(tmp[1])
    ny2 = int(tmp[2])
    ny = int(ny2 / ngeom)
    nx = int(tmp[3])
    tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
    s = f.readline().split()
    lat = float(tmp[0])
    lon = float(tmp[1])
    
    #Reading spectra
    s = f.readline().split()
    s = f.readline().split()
    wave = np.zeros([ny,ngeom])
    specret = np.zeros([ny,ngeom])
    specmeas = np.zeros([ny,ngeom])
    specerrmeas = np.zeros([ny,ngeom])
    for i in range(ngeom):
        for j in range(ny):
            tmp = np.fromfile(f,sep=' ',count=7,dtype='float')
            wave[j,i] = float(tmp[1])
            specret[j,i] = float(tmp[5])
            specmeas[j,i] = float(tmp[2])
            specerrmeas[j,i] = float(tmp[3])

    #Reading the retrieved state vector
    s = f.readline().split()
    nvar = int(s[2])
    nxvar = np.zeros([nvar],dtype='int')
    Var = Variables()
    Var.NVAR = nvar
    aprprof1 = np.zeros([nx,nvar])
    aprerr1 = np.zeros([nx,nvar])
    retprof1 = np.zeros([nx,nvar])
    reterr1 = np.zeros([nx,nvar])
    varident = np.zeros([nvar,3],dtype='int')
    varparam = np.zeros([nvar,5])
    for i in range(nvar):
        s = f.readline().split()
        tmp = np.fromfile(f,sep=' ',count=3,dtype='int')
        varident[i,:] = tmp[:]
        tmp = np.fromfile(f,sep=' ',count=5,dtype='float')
        varparam[i,:] = tmp[:]
        s = f.readline().split()
        Var1 = Variables()
        Var1.NVAR = 1
        Var1.edit_VARIDENT(varident[i,:])
        Var1.edit_VARPARAM(varparam[i,:])
        Var1.calc_NXVAR(Atm.NP)
        for j in range(Var1.NXVAR[0]):
            tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
            aprprof1[j,i] = float(tmp[2])
            aprerr1[j,i] = float(tmp[3])
            retprof1[j,i] = float(tmp[4])
            reterr1[j,i] = float(tmp[5])

    Var.edit_VARIDENT(varident)
    Var.edit_VARPARAM(varparam)
    Var.calc_NXVAR(Atm.NP)

    aprprof = np.zeros([nxvar.max(),nvar])
    aprerr = np.zeros([nxvar.max(),nvar])
    retprof = np.zeros([nxvar.max(),nvar])
    reterr = np.zeros([nxvar.max(),nvar])

    for i in range(nvar):
        aprprof[0:nxvar[i],i] = aprprof1[0:nxvar[i],i]
        aprerr[0:nxvar[i],i] = aprerr1[0:nxvar[i],i]
        retprof[0:nxvar[i],i] = retprof1[0:nxvar[i],i]
        reterr[0:nxvar[i],i] = reterr1[0:nxvar[i],i]

 
    return lat,lon,ngeom,ny,wave,specret,specmeas,specerrmeas,nx,Var,aprprof,aprerr,retprof,reterr


###############################################################################################

def read_ref(runname, Atm=None, MakePlot=False, SavePlot=False):
    
    """
        FUNCTION NAME : read_ref()
        
        DESCRIPTION : Reads the .ref file from a Nemesis run
        
        INPUTS :
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS:

            Atm :: If None, a new atmospheric class is created to store the parameters. 
                   If this is filled with another class, then this will be updated with the
                   parameters in the file
            MakePlot : If True, a summary plot is made
        
        OUTPUTS :
        
            amform :: if amform =1 then assumed that all VMR sum up 1.
            nplanet :: Planet ID (Mercury=1, Venus=2, Earth=3, Mars=4...)
            xlat :: Planetocentric latitude
            npro :: Number of points in the profile
            ngas :: Number of gases whose volume mixing ratios are included in the file
            molwt :: Mean molecular weight of the atmosphere in grams
            gasID(ngas) :: HITRAN ID of the gas that need to be included
            isoID(ngas) :: ID Number of the isotopologue to include (0 for all)
            height(npro) :: height profile in km
            press(npro) :: pressure profile in atm
            temp(npro) :: temperature profiles in K
            vmr(npro,ngas) :: volume mixing ratio of the different
        
        CALLING SEQUENCE:
        
            amform,nplanet,xlat,npro,ngas,molwt,gasID,isoID,height,press,temp,vmr = read_ref(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    #Opening file
    f = open(runname+'.ref','r')
    
    #Reading first and second lines
    tmp = np.fromfile(f,sep=' ',count=1,dtype='int')
    amform = int(tmp[0])
    tmp = np.fromfile(f,sep=' ',count=1,dtype='int')
    
    #Reading third line
    tmp = np.fromfile(f,sep=' ',count=5,dtype='float')
    nplanet = int(tmp[0])
    xlat = float(tmp[1])
    npro = int(tmp[2])
    ngas = int(tmp[3])
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
    if Atm==None:
        Atm = Atmosphere_1()
    
    Atm.NP = npro
    Atm.NVMR = ngas
    Atm.ID = gasID
    Atm.ISO = isoID
    Atm.IPLANET = nplanet
    Atm.LATITUDE = xlat
    Atm.AMFORM = amform
    Atm.edit_H(height*1.0e3)
    Atm.edit_P(press*101325.)
    Atm.edit_T(temp)
    Atm.edit_VMR(vmr)
    Atm.runname = runname

    if ( (Atm.AMFORM==1) or (Atm.AMFORM==2) ):
        Atm.calc_molwt()
    else:
        molwt1 = np.zeros(npro)
        molwt1[:] = molwt
        Atm.MOLWT = molwt1 / 1000.   #kg/m3

    Atm.calc_grav()

    #Make plot if keyword is specified
    if MakePlot == True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(10,4))

        ax1.semilogx(Atm.P/101325.,Atm.H/1.0e3,c='black')
        ax2.plot(Atm.T,Atm.H/1.0e3,c='black')
        for i in range(Atm.NVMR):
            label1 = gas_info[str(Atm.ID[i])]['name']
            if Atm.ISO[i]!=0:
                label1 = label1+' ('+str(Atm.ISO[i])+')'
            ax3.semilogx(Atm.VMR[:,i],Atm.H/1.0e3,label=label1)
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Temperature (K)')
        ax3.set_xlabel('Volume mixing ratio')
        plt.subplots_adjust(left=0.08,bottom=0.12,right=0.88,top=0.96,wspace=0.16,hspace=0.20)
        legend = ax3.legend(bbox_to_anchor=(1.01, 1.02))
        ax1.grid()
        ax2.grid()
        ax3.grid()
        if SavePlot==True:
            fig.savefig(runname+'refatm.png',dpi=200)

        plt.show()

    return Atm

###############################################################################################

def read_aerosol(Atm=None, MakePlot=False, SavePlot=False):

    """

        FUNCTION NAME : read_aerosol()

        DESCRIPTION : Reads the aerosol.ref file from a Nemesis run

        INPUTS : none

        OPTIONAL INPUTS:

            Atm :: If None, a new atmospheric class is created to store the parameters. 
                   If this is filled with another class, then this will be updated with the
                   parameters in the file
            MakePlot : If True, a summary plot is made
            
        OUTPUTS : 

            npro :: Number of points in the profile
            naero :: Number of aerosol types
            height(npro) :: Altitude (km)
            aerodens(npro,naero) :: Aerosol density of each particle type (particles per gram of air)
  
        CALLING SEQUENCE:

            Atm = read_aerosol_nemesis()

        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    #Opening file
    f = open('aerosol.ref','r')

    #Reading header
    s = f.readline().split()

    #Reading first line
    tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
    npro = tmp[0]
    naero = tmp[1]

    #Reading data
    height = np.zeros([npro])
    aerodens = np.zeros([npro,naero])
    for i in range(npro):
        tmp = np.fromfile(f,sep=' ',count=naero+1,dtype='float')
        height[i] = tmp[0]
        for j in range(naero):
            aerodens[i,j] = tmp[j+1]

    #Storing the results into the atmospheric class
    if Atm==None:
        Atm = Atmosphere_1()
    else:
        if npro!=Atm.NP:
            sys.exit('Number of altitude points in aerosol.ref must be equal to NP')
    
    Atm.NP = npro
    Atm.NDUST = naero
    Atm.edit_H(height*1.0e3)   #m
    Atm.edit_DUST(aerodens)

    #Make plot if keyword is specified
    if MakePlot == True:
        fig,ax1 = plt.subplots(1,1,figsize=(4,7))
        ax1.set_xlabel('Aerosol density (part. per gram of air)')
        ax1.set_ylabel('Altitude (km)')
        for i in range(Atm.NDUST):
                im = ax1.plot(Atm.DUST[:,i],Atm.H/1.0e3)
        plt.grid()
        plt.show()
        if SavePlot == True:
                fig.savefig(runname+'_aerosol.png',dpi=200)

    return Atm

###############################################################################################

def read_sur(runname,MakePlot=False):
    
    """
        FUNCTION NAME : read_sur()
        
        DESCRIPTION : Read the .sur file (surface emissivity spectrum)
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is made
        
        OUTPUTS :
        
            nem :: Number of spectral points in surface emissitivity spectrum
            vem(nem) :: Wavenumber array (cm-1)
            emissivity(nem) :: Surface emissivity
        
        CALLING SEQUENCE:
        
            nem,vem,emissivity = read_sur(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    #Opening file
    f = open(runname+'.sur','r')
    nem = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
    
    vem = np.zeros([nem])
    emissivity = np.zeros([nem])
    for i in range(nem):
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        vem[i] = tmp[0]
        emissivity[i] = tmp[1]

    if MakePlot==True:
        fig,ax1=plt.subplots(1,1,figsize=(8,3))
        ax1.plot(vem,emissivity)
        plt.tight_layout()
        plt.show()
    
    return nem,vem,emissivity

###############################################################################################

def read_xsc(runname):
    
    """
        FUNCTION NAME : read_xsc()
        
        DESCRIPTION : This function reads the .xsc file, which contains information about the extinction cross
        section and the single scattering albedo of aerosols.
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            naero :: Number of aerosol populations
            nwave :: Number of wavelengths
            wave(nwave) :: Wavenumber/wavelength array (cm-1/um)
            ext_coeff(nwave,naero) :: Extinction coefficient (cm2)
            sglalb(nwave,naero) :: Single scattering albedo
        
        CALLING SEQUENCE:
        
            naero,nwave,wave,ext_coeff,sglalb = read_xsc(runname)
        
        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        
    """
    
    #reading number of lines in file
    nlines = utils.file_lines(runname+'.xsc')
    nwave = int((nlines-1)/ 2)
    
    #Reading file
    f = open(runname+'.xsc','r')
    
    s = f.readline().split()
    naero = int(s[0])
    
    wave = np.zeros([nwave])
    ext_coeff = np.zeros([nwave,naero])
    sglalb = np.zeros([nwave,naero])
    for i in range(nwave):
        s = f.readline().split()
        wave[i] = float(s[0])
        for j in range(naero):
            ext_coeff[i,j] = float(s[j+1])
        s = f.readline().split()
        for j in range(naero):
            sglalb[i,j] = float(s[j])

    f.close()

    return nwave,naero,wave,ext_coeff,sglalb

###############################################################################################

def read_inp(runname):
    
    """
        FUNCTION NAME : read_inp()
        
        DESCRIPTION : Read the .inp file for a Nemesis run
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            ispace :: (0) Wavenumber in cm-1 (1) Wavelength in um
            iscat :: (0) Thermal emission calculation
                     (1) Multiple scattering required
                     (2) Internal scattered radiation field is calculated first (required for limb-
                         scattering calculations)
                     (3) Single scattering plane-parallel atmosphere calculation
                     (4) Single scattering spherical atmosphere calculation
            ilbl :: (0) Pre-tabulated correlated-k calculation
                    (1) Line by line calculation
                    (2) Pre-tabulated line by line calculation
            woff :: Wavenumber/wavelength calibration offset error to be added to the synthetic spectra
            niter :: Number of iterations of the retrieval model required
            philimit :: Percentage convergence limit. If the percentage reduction of the cost function phi
                        is less than philimit then the retrieval is deemed to have converged.
            nspec :: Number of retrievals to perform (for measurements contained in the .spx file)
            ioff :: Index of the first spectrum to fit (in case that nspec > 1).
            lin :: Integer indicating whether the results from previous retrievals are to be used to set any
                    of the atmospheric profiles. (Look Nemesis manual)
        
        CALLING SEQUENCE:
        
            ispace,iscat,ilbl,woff,niter,philimit,nspec,ioff,lin = read_inp(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
        """
    
    #Opening file
    f = open(runname+'.inp','r')
    tmp = f.readline().split()
    ispace = int(tmp[0])
    iscat = int(tmp[1])
    ilbl = int(tmp[2])
    
    tmp = f.readline().split()
    woff = float(tmp[0])
    fmerrname = str(f.readline().split())
    tmp = f.readline().split()
    niter = int(tmp[0])
    tmp = f.readline().split()
    philimit = float(tmp[0])
    
    tmp = f.readline().split()
    nspec = int(tmp[0])
    ioff = int(tmp[1])
    
    tmp = f.readline().split()
    lin = int(tmp[0])
    
    return  ispace,iscat,ilbl,woff,niter,philimit,nspec,ioff,lin

###############################################################################################

def read_apr(runname,npro):
    
    """
        FUNCTION NAME : read_apr()
        
        DESCRIPTION :
        
            Reads the .apr file, which contains information about the variables and
            parametrisations that are to be retrieved, as well as their a priori values
        
            N.B. In this code, the apriori and retrieved vectors x are usually
            converted to logs, all except for temperature and fractional scale
            heights
            This is done to reduce instabilities when different parts of the
            vectors and matrices hold vastly different sized properties. e.g.
            cloud x-section and base height.
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            npro :: Number of elements in atmospheric profiles
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Var :: Python class defining the variables and parameterisations
            Xn :: Python class defining the state vector
            nvar :: Number of variables
            varident(nvar,3) :: Variable ID as presented in the Nemesis manual
            varparam(nvar,nparam) :: Additional parameters constraining the profile
            jsurf :: Position of surface temperature element (if included)
            jalb :: Position of start of surface albedo spectrum (if included)
            jxsc :: Position of start of x-section spectrum (if included)
            jtan :: Position of tangent altitude correction (if included)
            jpre :: Position of ref. tangent  pressure (if included)
            jrad :: Position of radius of planet (if included)
            jlogg :: Position of surface log_10(g) of planet (if included)
            jfrac :: Position of fractional coverage
            nx :: Number of elements in state vector
            xa(nx) :: A priori state vector
            sa(nx,nx) :: A priori covariance matrix
            lx(nx) :: Log flag. 0 if real number 1 if log number
            csx(nvar) :: Ratio volume of shell/total volume of particle for Maltmieser
                         coated sphere model. For homogeneous sphere model, csx(ivar)=-1
        
        CALLING SEQUENCE:
        
            nvar,varident,varparam,jsurf,jalb,jxsc,jtan,jpre,jrad,jlogg,jfrac,nx,xa,sa,lx = read_apr(runname,npro)
            Var,Xn = read_apr(runname,npro)
        
        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        
    """

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
    mx = 1000
    varident = np.zeros([nvar,3],dtype='int')
    varparam = np.zeros([nvar,mparam])
    lx = np.zeros([mx],dtype='int')
    x0 = np.zeros([mx])
    sx = np.zeros([mx,mx])

    #Reading data
    ix = 0
    
    for i in range(nvar):
        s = f.readline().split()
        for j in range(3):
            varident[i,j] = int(s[j])
        
        print('Reading variable :: ',varident[i,:])
        
        #Starting different cases
        if varident[i,2] <= 100:    #Parameter must be an atmospheric one
            
            if varident[i,2] == 0:
                #           ********* continuous profile ************************
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
#           * continuous cloud, but cloud retrieved as particles/cm3 rather than
#           * particles per gram to decouple it from pressure.
#           ********* continuous particles/cm3 profile ************************
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
#           ******** profile held as deep amount, fsh and knee pressure **
#           Read in xdeep,fsh,pknee
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
#           **** Simple scaling factor of reference profile *******
#           Read in scaling factor

                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                x0[ix] = float(tmp[0])
                sx[ix,ix] = (float(tmp[1]))**2.

                ix = ix + 1
    
            elif varident[i,2] == 3:
#           **** Exponential scaling factor of reference profile *******
#           Read in scaling factor
        
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
#           ******** profile held as deep amount, fsh and VARIABLE knee press
#           Read in xdeep,fsh,pknee
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
#           ******** cloud profile held as total optical depth plus
#           ******** base height and fractional scale height. Below the knee
#           ******** pressure the profile is set to zero - a simple
#           ******** cloud in other words!
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
                else:
                    print('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

                err = edeep/xdeep
                sx[ix,ix] = err**2.

                ix = ix + 1

                if xfsh>0.0:
                    x0[ix] = np.log(xfsh)
                    lx[ix] = 1
                else:
                    print('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

                err = efsh/xfsh
                sx[ix,ix] = err**2.

                ix = ix + 1

                x0[ix] = hknee
                sx[ix,ix] = eknee**2.

                ix = ix + 1
            
            else:
                sys.exit('error in read_apr() :: Variable ID not included in this function')

        else:

            if varident[i,2] == 228:
#           ******** model for retrieving the ILS in ACS MIR solar occultation observations
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at lowest wavenumber
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at highest wavenumber
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Offset of the second gaussian with respect to the first one (assumed spectrally constant)
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #FWHM of the main gaussian at the lowest wavenumber
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #FWHM of the main gaussian at the highest wavenumber (Assumed linear variation)
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1

            if varident[i,2] == 229:
#           ******** model for retrieving the ILS in ACS MIR solar occultation observations

                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at lowest wavenumber
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
        
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at wavenumber in the middle
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at highest wavenumber
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Offset of the second gaussian with respect to the first one (assumed spectrally constant)
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #FWHM of the main gaussian (assumed to be constant in wavelength units)
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1
                
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
                x0[ix] = float(tmp[0])
                sx[ix,ix] = float(tmp[1])**2.
                lx[ix] = 0
                ix = ix + 1

            if varident[i,2] == 230:
#           ******** Aerosol opacity using Angstrom coefficient
#           !!! this model is only valid for the python version of nemesisSO

                s = f.readline().split()
                f1 = open(s[0],'r')
                tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
                nlevel = int(tmp[0])
                varparam[i,0] = nlevel
                for ilevel in range(nlevel):
                    tmp = np.fromfile(f1,sep=' ',count=4,dtype='float')
                    r0 = float(tmp[0])    #Opaity at the first wavenumber
                    err0 = float(tmp[1])
                    r1 = float(tmp[2])    #Angstrom coefficient
                    err1 = float(tmp[3])
                    x0[ix] = np.log(r0)
                    sx[ix,ix] = (err0/r0)**2.
                    lx[ix] = 1
                    x0[ix+1] = r1
                    sx[ix+1,ix+1] = err1**2.
                    lx[ix] = 0
                    ix = ix + 2

            if varident[i,2] == 231:
#           ******** This model multiplies the computed transmission spectra by TRANS = TRANS0 * (T0 + T1*(WAVE-WAVE0))
#           !!! this model is only valid for the python version of nemesisSO

                s = f.readline().split()
                f1 = open(s[0],'r')
                tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
                nlevel = int(tmp[0])
                varparam[i,0] = nlevel
                for ilevel in range(nlevel):
                    tmp = np.fromfile(f1,sep=' ',count=4,dtype='float')
                    r0 = float(tmp[0])   #Transmission level at the first wavenumber
                    err0 = float(tmp[1])
                    r1 = float(tmp[2])   #Slope ofthe transmission with wavenumber
                    err1 = float(tmp[3])
                    x0[ix] = r0
                    sx[ix,ix] = (err0)**2.
                    x0[ix+1] = r1
                    sx[ix+1,ix+1] = err1**2.
                    lx[ix] = 0
                    ix = ix + 2
            
            if varident[i,2] == 666:
#           ******** pressure at given altitude
                tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                htan = float(tmp[0])
                tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
                ptan = float(tmp[0])
                ptanerr = float(tmp[1])
                varparam[i,0] = htan
                if ptan>0.0:
                    x0[ix] = np.log(ptan)
                    lx[ix] = 1
                else:
                    sys.exit('error in read_apr_nemesis() :: pressure must be > 0')
                
                sx[ix,ix] = (ptanerr/ptan)**2.
                jpre = ix
                
                ix = ix + 1

            if varident[i,2] == 998:
#           ******** map of surface temperatures 
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
                    ix = ix + 1

            if varident[i,2] == 999:
#           ******** surface temperature
                s = f.readline().split()
                tsurf = float(s[0])
                esurf = float(s[1])
                x0[ix] = tsurf
                sx[ix,ix] = esurf**2.
                jsurf = ix
            
                ix = ix + 1

    f.close()

    nx = ix
    lx1 = np.zeros([nx],dtype='int32')
    xa = np.zeros([nx])
    sa = np.zeros([nx,nx])
    lx1[0:nx] = lx[0:nx]
    xa[0:nx] = x0[0:nx]
    sa[0:nx,0:nx] = sx[0:nx,0:nx]
                        
    Var = Variables()
    Var.NVAR=nvar
    Var.NPARAM=mparam
    Var.edit_VARIDENT(varident)
    Var.edit_VARPARAM(varparam)
    Var.calc_NXVAR(npro)

    Xn = StateVector()
    Xn.NX = nx
    Xn.JPRE, Xn.JTAN, Xn.JSURF, Xn.JALB, Xn.JXSC, Xn.JLOGG, Xn.JFRAC = jpre, jtan, jsurf, jalb, jxsc, jlogg, jfrac
    Xn.edit_XN(xa)
    Xn.edit_SX(sa)
    Xn.edit_LX(lx1)
                        
    return Var,Xn

###############################################################################################

def write_fcloud(npro,naero,height,frac,icloud, MakePlot=False):
    
    """
        FUNCTION NAME : write_fcloud()
        
        DESCRIPTION : Writes the fcloud.ref file, which specifies if the cloud is in the form of
                      a uniform thin haze or is instead arranged in thicker clouds covering a certain
                      fraction of the mean area.
        
        INPUTS :
        
            npro :: Number of altitude profiles in reference atmosphere
            naero :: Number of aerosol populations in the atmosphere
            height(npro) :: Altitude (km)
            frac(npro) :: Fractional cloud cover at each level
            icloud(npro,naero) :: Flag indicating which aerosol types contribute to the broken cloud
                                  which has a fractional cloud cover of frac
        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            fcloud.ref file
        
        CALLING SEQUENCE:
        
            ll = write_fcloud(npro,naero,height,frac,icloud)
        
        MODIFICATION HISTORY : Juan Alday (16/03/2021)
        
    """

    f = open('fcloud.ref','w')

    f.write('%i \t %i \n' % (npro,naero))
    
    for i in range(npro):
        str1 = str('{0:7.6f}'.format(height[i]))+'\t'+str('{0:7.3f}'.format(frac[i]))
        for j in range(naero):
            str1 = str1+'\t'+str('{0:d}'.format(icloud[i,j]))
            f.write(str1+'\n')

    f.close()

    dummy = 1
    return dummy

###############################################################################################

def write_ref(runname,amform,nplanet,xlat,npro,ngas,molwt,gasID,isoID,height,press,temp,vmr):
    
    """
        FUNCTION NAME : write_ref()
        
        DESCRIPTION : Writes the .ref file from a Nemesis run
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            amform :: if amform =1 then assumed that all VMR sum up 1.
            nplanet :: Planet ID (Mercury=1, Venus=2, Earth=3, Mars=4...)
            xlat :: Planetocentric latitude
            npro :: Number of points in the profile
            ngas :: Number of gases whose volume mixing ratios are included in the file
            molwt :: Mean molecular weight of the atmosphere in grams
            gasID(ngas) :: HITRAN ID of the gas that need to be included
            isoID(ngas) :: ID Number of the isotopologue to include (0 for all)
            height(npro) :: height profile in km
            press(npro) :: pressure profile in atm
            temp(npro) :: temperature profiles in K
            vmr(npro,ngas) :: volume mixing ratio of the different
        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            Nemesis .ref file
        
        CALLING SEQUENCE:
        
            ll = write_ref(runname,amform,nplanet,xlat,npro,ngas,molwt,gasID,isoID,height,press,temp,vmr)
        
        MODIFICATION HISTORY : Juan Alday (16/03/2021)
        
    """
    
    fref = open(runname+'.ref','w')
    fref.write('\t %i \n' % (amform))
    nlat = 1    #Would need to be updated to include more latitudes
    fref.write('\t %i \n' % (nlat))
    
    fref.write('\t %i \t %7.4f \t %i \t %i \t %7.4f \n' % (nplanet,xlat,npro,ngas,molwt))
    
    gasname = [''] * ngas
    header = [''] * (3+ngas)
    header[0] = 'height(km)'
    header[1] = 'press(atm)'
    header[2] = 'temp(K)  '
    str1 = header[0]+'\t'+header[1]+'\t'+header[2]
    for i in range(ngas):
        fref.write('\t %i \t %i\n' % (gasID[i],isoID[i]))
        strgas = 'GAS'+str(i+1)+'_vmr'
        str1 = str1+'\t'+strgas
    
    fref.write(str1+'\n')

    for i in range(npro):
        str1 = str('{0:7.6f}'.format(height[i]))+'\t'+str('{0:7.6e}'.format(press[i]))+'\t'+str('{0:7.4f}'.format(temp[i]))
        for j in range(ngas):
            str1 = str1+'\t'+str('{0:7.6e}'.format(vmr[i,j]))
            fref.write(str1+'\n')
    
    fref.close()
    dummy = 1
    return dummy

###############################################################################################

def write_fla(runname,inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv):
    
    """
        FUNCTION NAME : write_fla()
        
        DESCRIPTION : Write the .fla file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            inormal :: ortho/para-H2 ratio is in equilibrium (0) or normal 3:1 (1)
            iray :: (0) Rayleigh scattering optical depth not included
                    (1) Rayleigh optical depths for gas giant atmosphere
                    (2) Rayleigh optical depth suitable for CO2-dominated atmosphere
                    (>2) Rayleigh optical depth suitable for a N2-O2 atmosphere
            ih2o :: Additional H2O continuum off (0) or on (1)
            ich4 :: Additional CH4 continuum off (0) or on (1)
            io3 :: Additional O3 continuum off (0) or on (1)
            inh3 :: Additional NH3 continuum off (0) or on (1)
            iptf :: Normal partition function calculation (0) or high-temperature partition
                    function for CH4 for Hot Jupiters
            imie :: Only relevant for scattering calculations.
                    (0) Phase function is computed from the associated Henyey-Greenstein hgphase*.dat files.
                    (1) Phase function computed from the Mie-Theory calculated PHASEN.DAT
            iuv :: Additional flag for including UV cross sections off (0) or on (1)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemesis .fla file
        
        CALLING SEQUENCE:
        
            ll = write_fla(runname,inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv)
        
        MODIFICATION HISTORY : Juan Alday (16/03/2021)
        
    """
    
    f = open(runname+'.fla','w')
    f.write('%i \n' % (inormal))
    f.write('%i \n' % (iray))
    f.write('%i \n' % (ih2o))
    f.write('%i \n' % (ich4))
    f.write('%i \n' % (io3))
    f.write('%i \n' % (inh3))
    f.write('%i \n' % (iptf))
    f.write('%i \n' % (imie))
    f.write('%i \n' % (iuv))
    f.close()
    
    dummy = 1
    return dummy

###############################################################################################

def read_cov(runname,MakePlot=False):
    
    
    """
        FUNCTION NAME : read_cov()
        
        DESCRIPTION :
        
            Reads the the .cov file with the standard Nemesis format
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            npro :: Number of points in atmospheric profiles
            nvar :: Number of retrieved variables
            varident(nvar,3) :: Variable ID
            varparam(nvar,mparam) :: Extra parameters for describing the retrieved variable
            nx :: Number of elements in state vector
            ny :: Number of elements in measurement vector
            sa(nx,nx) :: A priori covariance matric
            sm(nx,nx) :: Final measurement covariance matrix
            sn(nx,nx) :: Final smoothing error covariance matrix
            st(nx,nx) :: Final full covariance matrix
            se(ny,ny) :: Measurement error covariance matrix
            aa(nx,nx) :: Averaging kernels
            dd(nx,ny) :: Gain matrix
            kk(ny,nx) :: Jacobian matrix
        
        CALLING SEQUENCE:
        
            npro,nvar,varident,varparam,nx,ny,sa,sm,sn,st,se,aa,dd,kk = read_cov(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    import matplotlib as matplotlib
    from matplotlib import gridspec
    from matplotlib import ticker
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    
    #Open file
    f = open(runname+'.cov','r')
    
    #Reading variables that were retrieved
    tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
    npro = int(tmp[0])
    nvar = int(tmp[1])
    
    varident = np.zeros([nvar,3],dtype='int')
    varparam = np.zeros([nvar,5],dtype='int')
    for i in range(nvar):
        tmp = np.fromfile(f,sep=' ',count=3,dtype='int')
        varident[i,:] = tmp[:]
        
        tmp = np.fromfile(f,sep=' ',count=5,dtype='float')
        varparam[i,:] = tmp[:]
    
    
    #Reading optimal estimation matrices
    tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
    nx = int(tmp[0])
    ny = int(tmp[1])


    sa = np.zeros([nx,nx])
    sm = np.zeros([nx,nx])
    sn = np.zeros([nx,nx])
    st = np.zeros([nx,nx])
    aa = np.zeros([nx,nx])
    dd = np.zeros([nx,ny])
    kk = np.zeros([ny,nx])
    se = np.zeros([ny,ny])
    for i in range(nx):
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            sa[i,j] = tmp[0]
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            sm[i,j] = tmp[0]
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            sn[i,j] = tmp[0]
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            st[i,j] = tmp[0]

    for i in range(nx):
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            aa[i,j] = tmp[0]
    
    
    for i in range(nx):
        for j in range(ny):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            dd[i,j] = tmp[0]

    for i in range(ny):
        for j in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
            kk[i,j] = tmp[0]
    
    for i in range(ny):
        tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
        se[i,i] = tmp[0]

    f.close()

    return npro,nvar,varident,varparam,nx,ny,sa,sm,sn,st,se,aa,dd,kk

###############################################################################################

def read_drv(runname,MakePlot=False):
    
    """
        FUNCTION NAME : read_drv()
        
        DESCRIPTION : Read the .drv file, which contains all the required information for
                      calculating the observation paths
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS:
        
            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            iconv :: Spectral model code
            flagh2p :: Flag for para-H2
            ncont :: Number of aerosol populations
            flagc :: Flag for
            nlayer :: Number of atmospheric layers
            npath :: Number of observation paths
            ngas :: Number of gases in atmosphere
            gasID(ngas) :: RADTRAN gas ID
            isoID(ngas) :: RADTRAN isotopologue ID (0 for all isotopes)
            iproc(ngas) :: Process parameter
            baseH(nlayer) :: Altitude of the base of each layer (km)
            delH(nlayer) :: Altitude covered by the layer (km)
            baseP(nlayer) :: Pressure at the base of each layer (atm)
            baseT(nlayer) :: Temperature at the base of each layer (K)
            totam(nlayer) :: Vertical total column density in atmosphere (cm-2)
            press(nlayer) :: Effective pressure of each layer (atm)
            temp(nlayer) :: Effective temperature of each layer (K)
            doppler(nlayer) ::
            par_coldens(nlayer,ngas) :: Vertical total column density for each gas in atmosphere (cm-2)
            par_press(nlayer,ngas) :: Partial pressure of each gas (atm)
            cont_coldens(nlayer,ncont) :: Aerosol column density for each aerosol population in atmosphere (particles per gram of atm)
            hfp(nlayer) ::
            hfc(nlayer,ncont) ::
            nlayin(npath) :: Number of layers seen in each path
            imod(npath) :: Path model
            errlim(npath) ::
            layinc(npath,2*nlayer) :: Layer indices seen in each path
            emtemp(npath,2*nlayer) :: Emission temperature of each layer in path
            scale(npath,2*nlayer) :: Factor to be applied to the vertical column density to calculate the line-of-sight column density
            nfilt :: Number of profile filter points
            filt(nfilt) :: Filter points
            vfilt(nfilt) ::
            ncalc :: Number of calculations
            itype(ncalc) :: Calculation type
            nintp(ncalc) ::
            nrealp(ncalc) ::
            nchp(ncalc) ::
            icald(ncalc,10) ::
            rcald(ncalc,10) ::
            
        CALLING SEQUENCE:
            
            iconv,flagh2p,ncont,flagc,nlayer,npath,ngas,gasID,isoID,iproc,\
            baseH,delH,baseP,baseT,totam,press,temp,doppler,par_coldens,par_press,cont_coldens,hfp,hfc,\
            nlayin,imod,errlim,layinc,emtemp,scale,\
            nfilt,filt,vfilt,ncalc,itype,nintp,nrealp,nchp,icald,rcald = read_drv(runname)
            
        MODIFICATION HISTORY : Juan Alday (29/09/2019)
            
    """

    f = open(runname+'.drv','r')
    
    #Reading header
    header = f.readline().split()
    var1 = f.readline().split()
    var2 = f.readline().split()
    linkey = f.readline().split()
    
    #Reading flags
    ###############
    flags = f.readline().split()
    iconv = int(flags[0])
    flagh2p = int(flags[1])
    ncont = int(flags[2])
    flagc = int(flags[3])
    
    #Reading name of .xsc file
    xscname1 = f.readline().split()
    
    #Reading variables
    ###################
    var1 = f.readline().split()
    nlayer = int(var1[0])
    npath = int(var1[1])
    ngas = int(var1[2])
    
    gasID = np.zeros([ngas],dtype='int32')
    isoID = np.zeros([ngas],dtype='int32')
    iproc = np.zeros([ngas],dtype='int32')
    for i in range(ngas):
        var1 = f.readline().split()
        var2 = f.readline().split()
        gasID[i] = int(var1[0])
        isoID[i] = int(var2[0])
        iproc[i] = int(var2[1])

    #Reading parameters of each layer
    ##################################
    header = f.readline().split()
    header = f.readline().split()
    header = f.readline().split()
    header = f.readline().split()
    baseH = np.zeros([nlayer])
    delH = np.zeros([nlayer])
    baseP = np.zeros([nlayer])
    baseT = np.zeros([nlayer])
    totam = np.zeros([nlayer])
    press = np.zeros([nlayer])
    temp = np.zeros([nlayer])
    doppler = np.zeros([nlayer])
    par_coldens = np.zeros([nlayer,ngas])
    par_press = np.zeros([nlayer,ngas])
    cont_coldens = np.zeros([nlayer,ncont])
    hfp = np.zeros([nlayer])
    hfc = np.zeros([nlayer,ncont])
    for i in range(nlayer):
        #Reading layers
        var1 = f.readline().split()
        baseH[i] = float(var1[1])
        delH[i] = float(var1[2])
        baseP[i] = float(var1[3])
        baseT[i] = float(var1[4])
        totam[i] = float(var1[5])
        press[i] = float(var1[6])
        temp[i] = float(var1[7])
        doppler[i] = float(var1[8])

        #Reading partial pressures and densities of gases in each layer
        nlines = ngas*2./6.
        if nlines-int(nlines)>0.0:
            nlines = int(nlines)+1
        else:
            nlines = int(nlines)

        ix = 0
        var = np.zeros([ngas*2])
        for il in range(nlines):
            var1 = f.readline().split()
            for j in range(len(var1)):
                var[ix] = var1[j]
                ix = ix + 1

        ix = 0
        for il in range(ngas):
            par_coldens[i,il] = var[ix]
            par_press[i,il] = var[ix+1]
            ix = ix + 2
        
        #Reading amount of aerosols in each layer
        nlines = ncont/6.
        if nlines-int(nlines)>0.0:
            nlines = int(nlines)+1
        else:
            nlines = int(nlines)
        var = np.zeros([ncont])
        ix = 0
        for il in range(nlines):
            var1 = f.readline().split()
            for j in range(len(var1)):
                var[ix] = var1[j]
                ix = ix + 1

        ix = 0
        for il in range(ncont):
            cont_coldens[i,il] = var[ix]
            ix = ix + 1
        
        #Reading if FLAGH2P is set
        if flagh2p==1:
            var1 = f.readline().split()
            hfp[i] = float(var1[0])


        #Reading if FLAGC is set
        if flagc==1:
            var = np.zeros([ncont])
            ix = 0
            for il in range(ncont):
                var1 = f.readline().split()
                for j in range(len(var1)):
                    var[ix] = var1[j]
                    ix = ix + 1

            ix = 0
            for il in range(ncont):
                hfc[i,il] = var[ix]
                ix = ix + 1
                    
    #Reading the atmospheric paths
    #########################################
    nlayin = np.zeros([npath],dtype='int32')
    imod = np.zeros([npath])
    errlim = np.zeros([npath])
    layinc = np.zeros([npath,2*nlayer],dtype='int32')
    emtemp = np.zeros([npath,2*nlayer])
    scale = np.zeros([npath,2*nlayer])
    for i in range(npath):
        var1 = f.readline().split()
        nlayin[i] = int(var1[0])
        imod[i] = int(var1[1])
        errlim[i] = float(var1[2])
        for j in range(nlayin[i]):
            var1 = f.readline().split()
            layinc[i,j] = int(var1[1]) - 1   #-1 stands for the fact that arrays in python start in 0, and 1 in fortran
            emtemp[i,j] = float(var1[2])
            scale[i,j] = float(var1[3])

    #Reading number of filter profile points
    #########################################
    var1 = f.readline().split()
    nfilt = int(var1[0])
    filt = np.zeros([nfilt])
    vfilt = np.zeros([nfilt])
    for i in range(nfilt):
        var1 = f.readline().split()
        filt[i] = float(var1[0])
        vfilt[i] = float(var1[1])
                            
    outfile = f.readline().split()

    #Reading number of calculations
    ################################
    var1 = f.readline().split()
    ncalc = int(var1[0])
    itype = np.zeros([ncalc],dtype='int32')
    nintp = np.zeros([ncalc],dtype='int32')
    nrealp = np.zeros([ncalc],dtype='int32')
    nchp = np.zeros([ncalc],dtype='int32')
    icald = np.zeros([ncalc,10],dtype='int32')
    rcald = np.zeros([ncalc,10])
    for i in range(ncalc):
        var1 = f.readline().split()
        itype[i] = int(var1[0])
        nintp[i] = int(var1[1])
        nrealp[i] = int(var1[2])
        nchp[i] = int(var1[3])
        for j in range(nintp[i]):
            var1 = f.readline().split()
            icald[i,j] = int(var1[0])
        for j in range(nrealp[i]):
            var1 = f.readline().split()
            rcald[i,j] = float(var1[0])
        for j in range(nchp[i]):
            var1 = f.readline().split()
            #NOT FINISHED HERE!!!!!!

    f.close()

    if MakePlot==True:
        #Plotting the model for the atmospheric layers
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(15,7))
        ax1.semilogx(baseP,baseH)
        ax1.set_xlabel('Pressure (atm)')
        ax1.set_ylabel('Base altitude (km)')
        ax1.grid()
        
        ax2.plot(baseT,baseH)
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Base altitude (km)')
        ax2.grid()
        
        ax3.semilogx(totam,baseH)
        ax3.set_xlabel('Vertical column density in layer (cm$^{-2}$)')
        ax3.set_ylabel('Base altitude (km)')
        ax3.grid()
        
        for i in range(ngas):
            strgas = spec.read_gasname(gasID[i],isoID[i])
            ax4.semilogx(par_coldens[:,i],baseH,label=strgas)
    
        ax4.legend()
        ax4.set_xlabel('Vertical column density in layer (cm$^{-2}$)')
        ax4.set_ylabel('Base altitude (km)')
        ax4.grid()
        
        plt.tight_layout()
        
        plt.show()

    return iconv,flagh2p,ncont,flagc,nlayer,npath,ngas,gasID,isoID,iproc,\
            baseH,delH,baseP,baseT,totam,press,temp,doppler,par_coldens,par_press,cont_coldens,hfp,hfc,\
            nlayin,imod,errlim,layinc,emtemp,scale,\
            nfilt,filt,vfilt,ncalc,itype,nintp,nrealp,nchp,icald,rcald

###############################################################################################

def read_spx_SO(runname):
    
    """
        FUNCTION NAME : read_spx_nemesisl()
        
        DESCRIPTION : Reads the .spx file from a NemesisL run
        
        INPUTS :
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            inst_fwhm :: Instrument full-width at half maximum
            xlat :: Planetocentric latitude at centre of the field of view
            xlon :: Planetocentric longitude at centre of the field of view
            ngeom :: Number of different observation geometries under which the location is observed
            nav ::  For each geometry, nav how many individual spectral calculations need to be
                    performed in order to reconstruct the field of fiew
            wgeom(ngeom,nav) :: Integration weight
            nconv(ngeom) :: Number of wavenumbers/wavelengths in each spectrum
            flat(ngeom,nav) :: Integration point latitude (when nav > 1)
            flon(ngeom,nav) :: Integration point longitude (when nav > 1)
            tanhe(ngeom) :: Tangent height (km)
            wave(nconv,ngeom,nav) :: Wavenumbers/wavelengths
            meas(nconv,ngeom,nav) :: Measured spectrum
            errmeas(nconv,ngeom,nav) :: Measurement noise
        
        CALLING SEQUENCE:
        
            inst_fwhm,xlat,xlon,ngeom,nav,wgeom,nconv,flat,flon,tanhe,wave,meas,errmeas = read_spx_SO(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
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
    tanhe = np.zeros([ngeom])
    wgeom = np.zeros([ngeom,nav])
    nconvmax = 20000
    wavetmp = np.zeros([nconvmax,ngeom,nav])
    meastmp = np.zeros([nconvmax,ngeom,nav])
    errmeastmp = np.zeros([nconvmax,ngeom,nav])
    for i in range(ngeom):
        nconv[i] = int(f.readline().strip())
        for j in range(nav):
            navsel = int(f.readline().strip())
            tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
            flat[i,j] = float(tmp[0])
            flon[i,j] = float(tmp[1])
            tanhe[i] = float(tmp[2])
            wgeom[i,j] = float(tmp[5])
        for iconv in range(nconv[i]):
            tmp = np.fromfile(f,sep=' ',count=3,dtype='float')
            wavetmp[iconv,i,j] = float(tmp[0])
            meastmp[iconv,i,j] = float(tmp[1])
            errmeastmp[iconv,i,j] = float(tmp[2])


    #Making final arrays for the measured spectra
    nconvmax2 = max(nconv)
    wave = np.zeros([nconvmax2,ngeom,nav])
    meas = np.zeros([nconvmax2,ngeom,nav])
    errmeas = np.zeros([nconvmax2,ngeom,nav])
    for i in range(ngeom):
        wave[0:nconv[i],:,:] = wavetmp[0:nconv[i],:,:]
        meas[0:nconv[i],:,:] = meastmp[0:nconv[i],:,:]
        errmeas[0:nconv[i],:,:] = errmeastmp[0:nconv[i],:,:]


    return inst_fwhm,xlat,xlon,ngeom,nav,wgeom,nconv,flat,flon,tanhe,wave,meas,errmeas


###############################################################################################

def read_sha(runname):
    
    """
        FUNCTION NAME : read_sha_nemesis()
        
        DESCRIPTION : Read the .sha file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            lineshape :: Instrument lineshape as defined by Nemesis manual
                        (0) Square lineshape
                        (1) Triangular
                        (2) Gaussian
                        (3) Hamming
                        (4) Hanning
        
        CALLING SEQUENCE:
        
            lineshape = read_sha(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
        """
    
    #Opening file
    f = open(runname+'.sha','r')
    s = f.readline().split()
    lineshape = int(s[0])
    
    return lineshape

###############################################################################################

def read_lls(runname):
    
    """
        FUNCTION NAME : read_lls()
        
        DESCRIPTION : Read the .lls file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            ngasact :: Number of active gases
            strlta(ngasact) :: String containg the .lta lbl-tables
        
        CALLING SEQUENCE:
        
            ngasact,strlta = read_lls(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    ngasact = len(open(runname+'.lls').readlines(  ))
    
    #Opening file
    f = open(runname+'.lls','r')
    #strlta = np.chararray(ngasact,itemsize=1000)
    strlta = [''] * ngasact
    for i in range(ngasact):
        s = f.readline().split()
        strlta[i] = s[0]
    
    return ngasact,strlta

###############################################################################################

def read_kls(runname):
    
    """
        FUNCTION NAME : read_kls()
        
        DESCRIPTION : Read the .kls file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            ngasact :: Number of active gases
            strkta(ngasact) :: String containg the .kta lbl-tables
        
        CALLING SEQUENCE:
        
            ngasact,strkta = read_kls(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """
    
    ngasact = len(open(runname+'.kls').readlines(  ))
    
    #Opening file
    f = open(runname+'.kls','r')
    #strlta = np.chararray(ngasact,itemsize=1000)
    strkta = [''] * ngasact
    for i in range(ngasact):
        s = f.readline().split()
        strkta[i] = s[0]
    
    return ngasact,strkta

###############################################################################################

def read_fil(runname, MakePlot=False):
    
    """
        FUNCTION NAME : read_fil()
        
        DESCRIPTION : Read the .fil file and store the data into variables
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            nconv :: Number of convolution wavelengths
            wave(nconv) :: Wavenumber array of the spectrum (cm-1)
            nfil(nconv) :: Number of wavelengths used to describe the ILS for each
            spectral point
            vfil(nfil,nconv) :: Wavenumber array used for describing the ILS (cm-1)
            afil(nfil,nconv) :: Function describing the ILS for each spectral point
        
        CALLING SEQUENCE:
        
            nconv,wave,nfil,vfil,afil = read_fil(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
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
    
    
    return nconv,wave,nfil,vfil,afil
