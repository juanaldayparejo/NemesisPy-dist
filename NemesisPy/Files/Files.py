# NAME:
#       Files.py (NemesisPy)
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
import matplotlib as matplotlib
from NemesisPy.Utils import *
from NemesisPy.Profile import *
from NemesisPy.Data import *
from NemesisPy.Models import *
from NemesisPy.Path import *
from NemesisPy.Layer import *
from NemesisPy.Radtrans import *
from NemesisPy.Surface import *
from NemesisPy.Scatter import *

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
    Atmosphere = Atmosphere_1(runname=runname)
    Atmosphere.read_ref()
    
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
    if len(s)==2:
        nvar = int(s[1])
    else:
        nvar = int(s[2])
    nxvar = np.zeros([nvar],dtype='int')
    Var = Variables_0()
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
        Var1 = Variables_0()
        Var1.NVAR = 1
        Var1.edit_VARIDENT(varident[i,:])
        Var1.edit_VARPARAM(varparam[i,:])
        Var1.calc_NXVAR(Atmosphere.NP)
        for j in range(Var1.NXVAR[0]):
            tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
            aprprof1[j,i] = float(tmp[2])
            aprerr1[j,i] = float(tmp[3])
            retprof1[j,i] = float(tmp[4])
            reterr1[j,i] = float(tmp[5])

    Var.edit_VARIDENT(varident)
    Var.edit_VARPARAM(varparam)
    Var.calc_NXVAR(Atmosphere.NP)

    aprprof = np.zeros([Var.NXVAR.max(),nvar])
    aprerr = np.zeros([Var.NXVAR.max(),nvar])
    retprof = np.zeros([Var.NXVAR.max(),nvar])
    reterr = np.zeros([Var.NXVAR.max(),nvar])

    for i in range(Var.NVAR):
        aprprof[0:Var.NXVAR[i],i] = aprprof1[0:Var.NXVAR[i],i]
        aprerr[0:Var.NXVAR[i],i] = aprerr1[0:Var.NXVAR[i],i]
        retprof[0:Var.NXVAR[i],i] = retprof1[0:Var.NXVAR[i],i]
        reterr[0:Var.NXVAR[i],i] = reterr1[0:Var.NXVAR[i],i]

    return lat,lon,ngeom,ny,wave,specret,specmeas,specerrmeas,nx,Var,aprprof,aprerr,retprof,reterr

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
        
            write_fcloud(npro,naero,height,frac,icloud)
        
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
        
            write_fla(runname,inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv)
        
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
            var = np.zeros([ncont+1])
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
            #strgas = spec.read_gasname(gasID[i],isoID[i])
            strgas = 'CHANGE'
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

def read_spx(runname, MakePlot=False, SavePlot=False):

    """
    FUNCTION NAME : read_spx()

    DESCRIPTION : Reads the .spx file from a Nemesis run

    INPUTS :
 
        runname :: Name of the Nemesis run

    OPTIONAL INPUTS:

        MakePlot : If True, a summary plot is made
            
    OUTPUTS : 
        inst_fwhm :: Instrument full-width at half maximum
        xlat :: Planetocentric latitude at centre of the field of view
        xlon :: Planetocentric longitude at centre of the field of view
        ngeom :: Number of different observation geometries under which the location is observed
        nav(ngeom) ::  For each geometry, nav how many individual spectral calculations need to be
                performed in order to reconstruct the field of fiew
        nconv(ngeom) :: Number of wavenumbers/wavelengths in each spectrum
        flat(ngeom,nav) :: Integration point latitude (when nav > 1)
        flon(ngeom,nav) :: Integration point longitude (when nav > 1)
        sol_ang(ngeom,nav) :: Solar incident angle 
        emiss_ang(ngeom,nav) :: Emission angle
        azi_ang(ngeom,nav) :: Azimuth angle
        wgeom(ngeom,nav) :: Weights for the averaging of the FOV
        wave(nconv,ngeom,nav) :: Wavenumbers/wavelengths
        meas(nconv,ngeom,nav) :: Measured spectrum
        errmeas(nconv,ngeom,nav) :: Measurement noise

    CALLING SEQUENCE:

        inst_fwhm,xlat,xlon,ngeom,nav,nconv,flat,flon,sol_ang,emiss_ang,azi_ang,wgeom,wave,meas,errmeas = read_spx(runname)

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
    navmax = 1000
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


    #Make plot if keyword is specified
    if (MakePlot == True) or (SavePlot == True):
        axis_font = {'size':'13'}
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(20,8))
        wavemin = wave.min()
        wavemax = wave.max()
        ax1.set_xlim(wavemin,wavemax)
        ax1.tick_params(labelsize=13)
        ax1.ticklabel_format(useOffset=False)
        ax2.set_xlim(wavemin,wavemax)
        ax2.tick_params(labelsize=13)
        ax2.ticklabel_format(useOffset=False)
        ax2.set_yscale('log')

        ax2.set_xlabel('Wavenumber/Wavelength',**axis_font)
        ax1.set_ylabel('Radiance',**axis_font)  
        ax2.set_ylabel('Radiance',**axis_font)

        for i in range(ngeom):
            im = ax1.plot(wave[0:nconv[i],i,0],meas[0:nconv[i],i,0])
            ax1.fill_between(wave[0:nconv[i],i,0],meas[0:nconv[i],i,0]-errmeas[0:nconv[i],i,0],meas[0:nconv[i],i,0]+errmeas[0:nconv[i],i,0],alpha=0.4)

        for i in range(ngeom):
            im = ax2.plot(wave[0:nconv[i],i,0],meas[0:nconv[i],i,0]) 
            ax2.fill_between(wave[0:nconv[i],i,0],meas[0:nconv[i],i,0]-errmeas[0:nconv[i],i,0],meas[0:nconv[i],i,0]+errmeas[0:nconv[i],i,0],alpha=0.4)
        
        plt.grid()

        if MakePlot==True:
            plt.show()
        if SavePlot == True:
            fig.savefig(runname+'_spectra.png',dpi=300)


    return inst_fwhm,xlat,xlon,ngeom,nav,nconv,flat,flon,sol_ang,emiss_ang,azi_ang,wgeom,wave,meas,errmeas



###############################################################################################

def write_spx(runname,inst_fwhm,xlat,xlon,ngeom,nav,nconv,flat,flon,sol_ang,emiss_ang,azi_ang,wgeom,wave,meas,errmeas, MakePlot=False, SavePlot=False):

    """

    FUNCTION NAME : write_spx()

    DESCRIPTION : Writes the .spx file from a Nemesis run

    INPUTS :

        runname :: Name of the Nemesis run
        inst_fwhm :: Instrument full-width at half maximum
        xlat :: Planetocentric latitude at centre of the field of view
        xlon :: Planetocentric longitude at centre of the field of view
        ngeom :: Number of different observation geometries under which the location is observed
        nav(ngeom) ::  For each geometry, nav how many individual spectral calculations need to be
                performed in order to reconstruct the field of fiew
        nconv(ngeom) :: Number of wavenumbers/wavelengths in each spectrum
        flat(ngeom,nav) :: Integration point latitude (when nav > 1)
        flon(ngeom,nav) :: Integration point longitude (when nav > 1)
        sol_ang(ngeom,nav) :: Solar incident angle
        emiss_ang(ngeom,nav) :: Emission angle
        azi_ang(ngeom,nav) :: Azimuth angle
        wgeom(ngeom,nav) :: Weights for the averaging of the FOV
        wave(nconv,ngeom,nav) :: Wavenumbers/wavelengths
        meas(nconv,ngeom,nav) :: Measured spectrum
        errmeas(nconv,ngeom,nav) :: Measurement noise

    OPTIONAL INPUTS:

        MakePlot : If True, a summary plot is made

    OUTPUTS :
    
        Nemesis .spx file

    CALLING SEQUENCE:

        ll = write_spx(runname,inst_fwhm,xlat,xlon,ngeom,nav,nconv,flat,flon,sol_ang,emiss_ang,azi_ang,wgeom,wave,meas,errmeas)

    MODIFICATION HISTORY : Juan Alday (29/04/2021)

    """

    fspx = open(runname+'.spx','w')
    fspx.write('%7.5f \t %7.5f \t %7.5f \t %i \n' % (inst_fwhm,xlat,xlon,ngeom))

    for i in range(ngeom):
        fspx.write('\t %i \n' % (nconv))
        fspx.write('\t %i \n' % (nav[i]))
        for j in range(nav[i]):
            fspx.write('\t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t \n' % (flat[i,j],flon[i,j],sol_ang[i,j],emiss_ang[i,j],azi_ang[i,j],wgeom[i,j]))
            for k in range(nconv[i]):
                fspx.write('\t %10.5f \t %20.7f \t %20.7f \n' % (wave[k,i,j],meas[k,i,j],errmeas[k,i,j]))

    fspx.close()
    dummy = 1
    return dummy

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


###############################################################################################


def read_inp(runname,Measurement=None,Scatter=None,Spectroscopy=None):

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
        
            Measurement,Scatter,Spectroscopy,WOFF,fmerrname,NITER,PHILIMIT,NSPEC,IOFF,LIN = read_inp(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    from NemesisPy import file_lines

    #Getting number of lines 
    nlines = file_lines(runname+'.inp')
    if nlines==7:
        iiform=0
    if nlines==8:
        iiform=1

    #Opening file
    f = open(runname+'.inp','r')
    tmp = f.readline().split()
    ispace = int(tmp[0])
    iscat = int(tmp[1])
    ilbl = int(tmp[2])

    
    if Measurement==None:
        Measurement = Measurement_0()
    Measurement.ISPACE=ispace

    if Scatter==None:
        Scatter = Scatter_0
    Scatter.ISPACE = ispace
    Scatter.ISCAT = iscat

    if Spectroscopy==None:
        Spectroscopy = Spectroscopy_0(RUNNAME=runname)
    Spectroscopy.ILBL = ilbl
    
    tmp = f.readline().split()
    WOFF = float(tmp[0])
    fmerrname = str(f.readline().split())
    tmp = f.readline().split()
    NITER = int(tmp[0])
    tmp = f.readline().split()
    PHILIMIT = float(tmp[0])
    
    tmp = f.readline().split()
    NSPEC = int(tmp[0])
    IOFF = int(tmp[1])
    
    tmp = f.readline().split()
    LIN = int(tmp[0])

    if iiform==1:
        tmp = f.readline().split()
        iform = int(tmp[0])
        Measurement.IFORM=iform
    else:
        Measurement.IFORM=0
    
    return  Measurement,Scatter,Spectroscopy,WOFF,fmerrname,NITER,PHILIMIT,NSPEC,IOFF,LIN

###############################################################################################

def read_set(runname,Layer=None,Surface=None,Stellar=None,Scatter=None):
    
    """
        FUNCTION NAME : read_set()
        
        DESCRIPTION : Read the .set file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Scatter :: Python class defining the scattering calculations
            Stellar :: Python class defining the stellar properties
            Surface :: Python class defining the surface properties
            Layer :: Python class defining the layering scheme of the atmosphere
        
        CALLING SEQUENCE:
        
            Scatter,Stellar,Surface,Layer = read_set(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    #Opening file
    f = open(runname+'.set','r')
    dummy = f.readline().split()
    nmu1 = f.readline().split()
    nmu = int(nmu1[5])
    mu = np.zeros([nmu],dtype='d')
    wtmu = np.zeros([nmu],dtype='d')
    for i in range(nmu):
        tmp = np.fromfile(f,sep=' ',count=2,dtype='d')
        mu[i] = tmp[0]
        wtmu[i] = tmp[1]
    
    dummy = f.readline().split()
    nf = int(dummy[5])
    dummy = f.readline().split()
    nphi = int(dummy[8])
    dummy = f.readline().split()
    isol = int(dummy[5])
    dummy = f.readline().split()
    dist = float(dummy[5])
    dummy = f.readline().split()
    lowbc = int(dummy[6])
    dummy = f.readline().split()
    galb = float(dummy[3])
    dummy = f.readline().split()
    tsurf = float(dummy[3])

    dummy = f.readline().split()

    dummy = f.readline().split()
    layht = float(dummy[8])
    dummy = f.readline().split()
    nlayer = int(dummy[5])
    dummy = f.readline().split()
    laytp = int(dummy[3])
    dummy = f.readline().split()
    layint = int(dummy[3])

    #Creating or updating Scatter class
    if Scatter==None:
        Scatter = Scatter_0()
        Scatter.NMU = nmu
        Scatter.NF = nf
        Scatter.NPHI = nphi
        Scatter.calc_GAUSS_LOBATTO()
    else:
        Scatter.NMU = nmu
        Scatter.calc_GAUSS_LOBATTO()
        Scatter.NF = nf
        Scatter.NPHI = nphi

    #Creating or updating Stellar class
    if Stellar==None:
        Stellar = Stellar_0()
        Stellar.DIST = dist
        if isol==1:
            Stellar.SOLEXIST = True
            Stellar.read_sol(runname)
        elif isol==0:
            Stellar.SOLEXIST = False
        else:
            sys.exit('error reading .set file :: SOLEXIST must be either True or False')

    #Creating or updating Surface class
    if Surface==None:
        Surface = Surface_0()

    Surface.LOWBC = lowbc
    Surface.GALB = galb
    Surface.TSURF = tsurf

    #Creating or updating Layer class
    if Layer==None:
        Layer = Layer_0()
    
    Layer.LAYHT = layht*1.0e3
    Layer.LAYTYP = laytp
    Layer.LAYINT = layint
    Layer.NLAY = nlayer

    return Scatter,Stellar,Surface,Layer

###############################################################################################

def plot_itr(runname):
    
    """
        FUNCTION NAME : plot_itr()
        
        DESCRIPTION : Read the .itr file and make some summary plots
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS : none
        
        CALLING SEQUENCE:
        
            plot_itr(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    #Opening file
    f = open(runname+'.itr','r')

    tmp = np.fromfile(f,sep=' ',count=3,dtype='int')
    nx = int(tmp[0])
    ny = int(tmp[1])
    niter = int(tmp[2])
    nx = 156

    chisq = np.zeros([niter])
    nchisq = np.zeros([niter])
    phi = np.zeros([niter])
    xn = np.zeros([nx,niter])
    xa = np.zeros([nx,niter])
    y = np.zeros([ny,niter])
    se = np.zeros([ny,niter])
    yn = np.zeros([ny,niter])
    yn1 = np.zeros([ny,niter])
    kk = np.zeros([nx,ny,niter])

    for it in range(niter):
        tmp = np.fromfile(f,sep=' ',count=2)
        chisq[it] = float(tmp[0])
        phi[it] = float(tmp[1])
        nchisq[it] = chisq[it]/ny

        for ix in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1)
            xn[ix,it] = float(tmp[0])

        for ix in range(nx):
            tmp = np.fromfile(f,sep=' ',count=1)
            xa[ix,it] = float(tmp[0])

        for iy in range(ny):
            tmp = np.fromfile(f,sep=' ',count=1)
            y[iy,it] = float(tmp[0])

        for iy in range(ny):
            tmp = np.fromfile(f,sep=' ',count=1)
            se[iy,it] = float(tmp[0])

        for iy in range(ny):
            tmp = np.fromfile(f,sep=' ',count=1)
            yn1[iy,it] = float(tmp[0])

        for iy in range(ny):
            tmp = np.fromfile(f,sep=' ',count=1)
            yn[iy,it] = float(tmp[0])

        for ix in range(nx):
            for iy in range(ny):
                tmp = np.fromfile(f,sep=' ',count=1)
                kk[ix,iy,it] = float(tmp[0])

        print('Iteration '+str(it)+' - Normalized chi-squared :: '+str(nchisq[it]))

        #Plotting the measurement vector
        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,6))
        ax1.plot(np.linspace(0,ny-1,ny),y[:,it],c='black',label='Measured')
        ax1.plot(np.linspace(0,ny-1,ny),yn[:,it],c='tab:green',label='Modelled')
        ax1.legend()
        ax2.semilogy(np.linspace(0,ny-1,ny),y[:,it],c='black')
        ax2.semilogy(np.linspace(0,ny-1,ny),yn[:,it],c='tab:green')
        ax3.plot(np.linspace(0,ny-1,ny),yn[:,it]-y[:,it],c='tab:green')
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel('Point number')
        ax1.set_ylabel('Measurement/Modelled vector')
        ax2.set_ylabel('Measurement/Modelled vector')
        ax3.set_ylabel('Residuals')

        plt.tight_layout()
        plt.show()

###############################################################################################

def read_fla(runname):
    
    """
        FUNCTION NAME : read_fla()
        
        DESCRIPTION : Read the .fla file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
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
            imie :: Only relevant for scattering calculations. (0) Phase function is computed
                    from the associated Henyey-Greenstein hgphase*.dat files. (1) Phase function
                    computed from the Mie-Theory calculated PHASEN.DAT
            iuv :: Additional flag for including UV cross sections off (0) or on (1)
        
        CALLING SEQUENCE:
        
            inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv = read_fla(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
        """
    
    #Opening file
    f = open(runname+'.fla','r')
    s = f.readline().split()
    inormal = int(s[0])
    s = f.readline().split()
    iray = int(s[0])
    s = f.readline().split()
    ih2o = int(s[0])
    s = f.readline().split()
    ich4 = int(s[0])
    s = f.readline().split()
    io3 = int(s[0])
    s = f.readline().split()
    inh3 = int(s[0])
    s = f.readline().split()
    iptf = int(s[0])
    s = f.readline().split()
    imie = int(s[0])
    s = f.readline().split()
    iuv = int(s[0])
   
    return inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv


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
            imie :: Only relevant for scattering calculations. (0) Phase function is computed
                    from the associated Henyey-Greenstein hgphase*.dat files. (1) Phase function
                    computed from the Mie-Theory calculated PHASEN.DAT
            iuv :: Additional flag for including UV cross sections off (0) or on (1)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemesis .fla file       
 
        CALLING SEQUENCE:
        
            write_fla(runname,inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open(runname+'.fla','w')
    f.write('%i \t %s \n' % (inormal,'!INORMAL'))
    f.write('%i \t %s \n' % (iray,'!IRAY'))
    f.write('%i \t %s \n' % (ih2o,'!IH2O'))
    f.write('%i \t %s \n' % (ich4,'!ICH4'))
    f.write('%i \t %s \n' % (io3,'!IO3'))
    f.write('%i \t %s \n' % (inh3,'!INH3'))
    f.write('%i \t %s \n' % (iptf,'!IPTF'))
    f.write('%i\t %s \n' % (imie,'!IMIE'))
    f.write('%i\t %s \n' % (iuv,'!IUV'))
    f.close()

###############################################################################################

def write_set(runname,nmu,nf,nphi,isol,dist,lowbc,galb,tsurf,layht,nlayer,laytp,layint):

    """
        FUNCTION NAME : write_set()
        
        DESCRIPTION : Read the .set file
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            nmu :: Number of zenith ordinates
            nf :: Required number of Fourier components
            nphi :: Number of azimuth angles
            isol :: Sunlight on/off
            dist :: Solar distance (AU)
            lowbc :: Lower boundary condition (0 Thermal - 1 Lambertian)
            galb :: Ground albedo
            tsurf :: Surface temperature (if planet is not gasgiant)
            layht :: Base height of lowest layer
            nlayer :: Number of vertical levels to split the atmosphere into
            laytp :: Flag to indicate how layering is perfomed (radtran)
            layint :: Flag to indicate how layer amounts are calculated (radtran)

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemesis .set file

        CALLING SEQUENCE:
        
            l = write_set(runname,nmu,nf,nphi,isol,dist,lowbc,galb,tsurf,layht,nlayer,laytp,layint)
        
        MODIFICATION HISTORY : Juan Alday (15/10/2019)
        
    """

    #Calculating the Gauss-Lobatto quadtrature points
    iScatter = Scatter_0(NMU=nmu)

    #Writin the .set file
    f = open(runname+'.set','w')
    f.write('********************************************************* \n')
    f.write('Number of zenith angles : '+str(nmu)+' \n')
    for i in range(nmu):
        f.write('\t %10.12f \t %10.12f \n' % (iScatter.MU[i],iScatter.WTMU[i]))
    f.write('Number of Fourier components : '+str(nf)+' \n')
    f.write('Number of azimuth angles for fourier analysis : '+str(nphi)+' \n')
    f.write('Sunlight on(1) or off(0) : '+str(isol)+' \n')
    f.write('Distance from Sun (AU) : '+str(dist)+' \n')
    f.write('Lower boundary cond. Thermal(0) Lambert(1) : '+str(lowbc)+' \n')
    f.write('Ground albedo : '+str(galb)+' \n')
    f.write('Surface temperature : '+str(tsurf)+' \n')
    f.write('********************************************************* \n')
    f.write('Alt. at base of bot.layer (not limb) : '+str(layht)+' \n')
    f.write('Number of atm layers : '+str(nlayer)+' \n')
    f.write('Layer type : '+str(laytp)+' \n')
    f.write('Layer integration : '+str(layint)+' \n')
    f.write('********************************************************* \n')

    f.close()

###############################################################################################

def write_inp(runname,ispace,iscat,ilbl,woff,niter,philimit,nspec,ioff,lin,IFORM=-1):

    """
        FUNCTION NAME : write_inp()
        
        DESCRIPTION : Write the .inp file for a Nemesis run
        
        INPUTS :
        
            runname :: Name of the Nemesis run
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
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
        CALLING SEQUENCE:

            write_inp(runname,ispace,iscat,ilbl,woff,niter,philimit,nspec,ioff,lin,IFORM=iform)
         
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    #Opening file
    f = open(runname+'.inp','w')
    f.write('%i \t %i \t %i \n' % (ispace,iscat,ilbl))
    f.write('%10.5f \n' % (woff))
    f.write(runname+'.err \n')
    f.write('%i \n' % (niter))
    f.write('%10.5f \n' % (philimit))
    f.write('%i \t %i \n' % (nspec,ioff))
    f.write('%i \n' % (lin))
    if IFORM!=-1:
        f.write('%i \n' % (IFORM))
    f.close()

###############################################################################################

def write_err(runname,nwave,wave,fwerr):

    """
        FUNCTION NAME : write_err()
        
        DESCRIPTION : Write the .err file, including information about forward modelling error
        
        INPUTS :
        
            runname :: Name of Nemesis run
            nwave :: Number of wavelengths at which the albedo is defined
            wave(nwave) :: Wavenumber/Wavelength array
            fwerr(nwave) :: Forward modelling error
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemeis .err file       
 
        CALLING SEQUENCE:
        
            write_err(runname,nwave,wave,fwerr)
         
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open(runname+'.err','w')
    f.write('\t %i \n' % (nwave))
    for i in range(nwave):
        f.write('\t %10.5f \t %10.5f \n' % (wave[i],fwerr[i]))
    f.close()

###############################################################################################

def write_sur(runname,nwave,wave,alb):

    """
        FUNCTION NAME : write_sur()
        
        DESCRIPTION : Write the .sur file 
        
        INPUTS :
        
            runname :: Name of Nemesis run
            nwave :: Number of wavelengths at which the albedo is defined
            wave(nwave) :: Wavenumber array
            emiss(nwave) :: Surface emissivity. Note that if GALB (defined in .set file) is negative
                            then the surface albedo is calculated as alb = 1 - emiss
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemeis .sur file       
 
        CALLING SEQUENCE:
        
            write_sur(runname,nwave,wave,emiss)
         
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    fsur = open(runname+'.sur','w')
    fsur.write('\t %i \n' % (nwave))
    for i in range(nwave):
        fsur.write('\t %10.5f \t %10.5f \n' % (wave[i],alb[i]))
    fsur.close()

###############################################################################################

def write_gcn(runname,NGAS,gasID,isoID,NGEOM,NCONV,VCONV,SPECMODGCN):
    
    """
        FUNCTION NAME : write_gcn()
        
        DESCRIPTION : Write the .gcn file including information about the contribution of each active gas
                      to each observation
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            NGAS :: Number of active gases in the atmosphere
            gasID(NGAS) :: Radtran ID for each gas
            isoID(NGAS) :: Radtran isotopologue ID for each gas
            NGEOM :: Number of geometries in observation
            NCONV(NGEOM) :: Number of convolution wavelengths in each geometry
            VCONV(NCONV,NGEOM) :: Wavelength/Wavenumber array
            SPECMODGCN(NCONV,NGEOM,NGAS) :: Modelled spectrum in each geometry for each active gas alone
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemesis .gcn file
        
        CALLING SEQUENCE:
        
            write_gcn(runname,NGAS,gasID,isoID,NGEOM,NCONV,VCONV,SPECMODGCN)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    f = open(runname+'.gcn','w')

    f.write('\t %i \t %i \n' % (NGAS,NGEOM))

    for i in range(NGAS):
        f.write('\t %i \t %i \n' % (gasID[i],isoID[i]))

    for i in range(NGEOM):
        f.write('\t %i \n' % (NCONV[i]))
        for j in range(NCONV[i]):
            str1 = str('{0:7.6f}'.format(VCONV[j,i]))
            for k in range(NGAS):
                str1 = str1+'\t'+str('{0:7.6e}'.format(SPECMODGCN[j,i,k]))

            f.write(str1+' \n')
    
    f.close()

###############################################################################################

def read_gcn(runname):
    
    """
        FUNCTION NAME : write_gcn()
        
        DESCRIPTION : Write the .gcn file including information about the contribution of each active gas
                      to each observation
        
        INPUTS :
        
            runname :: Name of the Nemesis run
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            NGAS :: Number of active gases in the atmosphere
            gasID(NGAS) :: Radtran ID for each gas
            isoID(NGAS) :: Radtran isotopologue ID for each gas
            NGEOM :: Number of geometries in observation
            NCONV(NGEOM) :: Number of convolution wavelengths in each geometry
            VCONV(NCONV,NGEOM) :: Wavelength/Wavenumber array
            SPECMODGCN(NCONV,NGEOM,NGAS) :: Modelled spectrum in each geometry for each active gas alone
        
        CALLING SEQUENCE:
        
            NGAS,gasID,isoID,NGEOM,NCONV,VCONV,SPECMODGCN = read_gcn(runname)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    f = open(runname+'.gcn','r')

    s = f.readline().split()
    NGAS = int(s[0])
    NGEOM = int(s[1])
    gasID = np.zeros(NGAS,dtype='int32')
    isoID = np.zeros(NGAS,dtype='int32')
    NCONV = np.zeros(NGEOM,dtype='int32')
    mconv = 3000

    VCONV1 = np.zeros([mconv,NGEOM])
    SPECMODGCN1 = np.zeros([mconv,NGEOM,NGAS])

    for i in range(NGAS):
        s = f.readline().split()
        gasID[i] = int(s[0])
        isoID[i] = int(s[1])
    
    for i in range(NGEOM):
        s = f.readline().split()
        NCONV[i] = int(s[0])
        for j in range(NCONV[i]):
            s = f.readline().split()
            VCONV1[j,i] = float(s[0])
            for k in range(NGAS):
                SPECMODGCN1[j,i,k] = float(s[k+1])

    f.close()

    VCONV = np.zeros([NCONV.max(),NGEOM])
    SPECMODGCN = np.zeros([NCONV.max(),NGEOM,NGAS])
    for i in range(NGEOM):
        VCONV[0:NCONV[i],i] = VCONV1[0:NCONV[i],i] 
        SPECMODGCN[0:NCONV[i],i,:] = SPECMODGCN1[0:NCONV[i],i,:]

    return NGAS,gasID,isoID,NGEOM,NCONV,VCONV,SPECMODGCN

###############################################################################################

def write_hlay(nlayer,heightlay):


    """
        FUNCTION NAME : write_hlay()
        
        DESCRIPTION : 

            Writes the height.lay file with the input required by Nemesis. This file specifies the
            base altitude of each layer in the atmosphere, which is read by the code when the Layer type is 5
 
        INPUTS :
      
            nlayer :: Number of layers in atmosphere
            heightlay(nlayer) :: Base altitude of each layer (km)

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            Nemesis height.lay file

        CALLING SEQUENCE:
        
            write_hlay(nlayer,heightlay)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    f = open('height.lay','w')
    header = 'Nemesis simulation - base altitude of atmospheric layers'
    f.write(header+"\n")
    f.write('\t %i \n' % (nlayer))
    for i in range(nlayer):
        f.write('\t %7.3f \n' % (heightlay[i]))
    f.close()


    ###############################################################################################

def read_input_files(runname):

    """
        FUNCTION NAME : read_input_files()
        
        DESCRIPTION : 

            Reads the NEMESIS input files and fills the parameters in the reference classes.
 
        INPUTS :
      
            runname :: Name of the NEMESIS run

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations

        CALLING SEQUENCE:
        
            Atmosphere,Measurement,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Variables = read_input_files(runname)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
    """

    #Initialise Atmosphere class and read file (.ref, aerosol.ref)
    ##############################################################

    Atm = Atmosphere_1(runname=runname)

    #Read gaseous atmosphere
    Atm.read_ref()

    #Read aerosol profiles
    Atm.read_aerosol()

    #Reading .set file and starting Scatter, Stellar, Surface and Layer Classes
    #############################################################################

    Layer = Layer_0(Atm.RADIUS)
    Scatter,Stellar,Surface,Layer = read_set(runname,Layer=Layer)

    #Reading .inp file and starting Measurement,Scatter and Spectroscopy classes
    #############################################################################

    Measurement,Scatter,Spec,WOFF,fmerrname,NITER,PHILIMIT,NSPEC,IOFF,LIN = read_inp(runname,Scatter=Scatter)

    #Reading .sur file if planet has surface
    #############################################################################

    isurf = planet_info[str(Atm.IPLANET)]["isurf"]
    if isurf==1:
        Surface.GASGIANT=False
        Surface.read_sur(runname)
    else:
        Surface.GASGIANT=True

    #Reading Spectroscopy parameters from .lls or .kls files
    ##############################################################

    if Spec.ILBL==0:
        Spec.read_kls(runname)
    elif Spec.ILBL==2:
        Spec.read_lls(runname)
    else:
        sys.exit('error :: ILBL has to be either 0 or 2')

    #Reading extinction and scattering cross sections
    #############################################################################

    Scatter.read_xsc(runname)

    if Scatter.NDUST!=Atm.NDUST:
        sys.exit('error :: Number of aerosol populations must be the same in .xsc and aerosol.ref files')


    #Initialise Measurement class and read files (.spx, .sha)
    ##############################################################

    Measurement.runname = runname
    Measurement.read_spx()

    #Reading .sha file if FWHM>0.0
    if Measurement.FWHM>0.0:
        Measurement.read_sha()
    #Reading .fil if FWHM<0.0
    elif Measurement.FWHM<0.0:
        Measurement.read_fil()

    #Calculating the 'calculation wavelengths'
    if Spec.ILBL==0:
        Measurement.wavesetb(Spec,IGEOM=0)
    elif Spec.ILBL==2:
        Measurement.wavesetc(Spec,IGEOM=0)
    else:
        sys.exit('error :: ILBL has to be either 0 or 2')

    #Now, reading k-tables or lbl-tables for the spectral range of interest
    Spec.read_tables(wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())


    #Reading stellar spectrum if required by Measurement units
    if( (Measurement.IFORM==1) or (Measurement.IFORM==2) or (Measurement.IFORM==3) or (Measurement.IFORM==4)):
        Stellar.read_sol(runname)

    #Initialise CIA class and read files (.cia)
    ##############################################################

    if os.path.exists(runname+'.cia')==True:
        CIA = CIA_0(runname=runname)
        CIA.read_cia()
    else:
        CIA = None

    #Reading .fla file
    #############################################################################

    inormal,iray,ih2o,ich4,io3,inh3,iptf,imie,iuv = read_fla(runname)
 
    if CIA is not None:
        CIA.INORMAL = inormal

    Scatter.IRAY = iray
    Scatter.IMIE = imie

    #Reading .apr file and Variables Class
    #################################################################

    Variables = Variables_0()
    Variables.read_apr(runname,Atm.NP)
    Variables.XN = copy(Variables.XA)
    Variables.SX = copy(Variables.SA)

    return Atm,Measurement,Spec,Scatter,Stellar,Surface,CIA,Layer,Variables