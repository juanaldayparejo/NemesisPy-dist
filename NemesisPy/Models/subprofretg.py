from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def subprofretg(runname,atm,ispace,iscat,xlat,xlon,Var,Xn,flagh2p):

    """
    FUNCTION NAME : subprogretg()

    DESCRIPTION : Updates the atmosphere based on the variables and parameterisations in the
                  state vector

    INPUTS :
    
        runname :: Name of the Nemesis run
        atm :: Python class defining the atmosphere
        ispace :: (0) Wavenumber in cm-1 (1) Wavelength in um
        iscat :: Type of scattering calculation
        xlat :: Latitude of spectrum to be simulated
        xlon :: Longitude of spectrum to be simulated
        Var :: Python class defining the model variables
        Xn :: Python class defining the state vector
        flagh2p :: Flag indicating whether para-H2 profile is variable

    OPTIONAL INPUTS: none
            
    OUTPUTS : 

        xmap(maxv,ngas+2+ncont,npro) :: Matrix relating functional derivatives calculated 
                                         by CIRSRADG to the elements of the state vector.
                                         Elements of XMAP are the rate of change of 
                                         the profile vectors (i.e. temperature, vmr prf
                                         files) with respect to the change in the state
                                         vector elements. So if X1(J) is the modified 
                                         temperature,vmr,clouds at level J to be 
                                         written out to runname.prf or aerosol.prf then
                                        XMAP(K,L,J) is d(X1(J))/d(XN(K)) and where
                                        L is the identifier (1 to NGAS+1+2*NCONT)

    CALLING SEQUENCE:

        xmap = subprofretg(runname,atm,ispace,iscat,xlat,xlon,Var,Xn,jpre,flagh2p)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)

    """

    #Modify profile via hydrostatic equation to make sure the atm is in hydrostatic equilibrium
    if Xn.JPRE==-1:
        jhydro = 0
        #Then we modify the altitude levels and keep the pressures fixed
        atm.adjust_hydrostatH()
        atm.calc_grav()   #Updating the gravity values at the new heights
    else:
        #Then we modifify the pressure levels and keep the altitudes fixed
        jhydro = 1
        for i in range(Var.NVAR):
            if Var.VARIDENT[i,0]==666:
                htan = Var.VARPARAM[i,0] * 1000.
        ptan = np.exp(Xn.XN[Xn.JPRE]) * 101325.
        atm.adjust_hydrostatP(htan,ptan)

    #Adjust VMRs to add up to 1 if AMFORM=1 and re-calculate molecular weight in atmosphere
    if atm.AMFORM==1:
        atm.adjust_VMR()
        atm.calc_molwt()
    elif atm.AMFORM==2:
        atm.calc_molwt()

    #Calculate atmospheric density
    rho = atm.calc_rho() #kg/m3
    
    #Initialising xmap
    xmap = np.zeros([Xn.NX,atm.NVMR+2+atm.NDUST,atm.NP])

    #Going through the different variables an updating the atmosphere accordingly
    ix = 0
    for ivar in range(Var.NVAR):

        if Var.VARIDENT[ivar,0]<=100:
            
            #Reading the atmospheric profile which is going to be changed by the current variable
            xref = np.zeros([atm.NP])

            if Var.VARIDENT[ivar,0]==0:     #Temperature is to be retrieved
                xref[:] = atm.T
                ipar = atm.NVMR
            elif Var.VARIDENT[ivar,0]>0:    #Gas VMR is to be retrieved
                jvmr = np.where( (np.array(atm.ID)==Var.VARIDENT[ivar,0]) & (np.array(atm.ISO)==Var.VARIDENT[ivar,0]) )
                jvmr = int(jvmr[0])
                xref[:] = atm.VMR[:,jvmr]
                ipar = jvmr
            elif Var.VARIDENT[ivar,0]<0:  
                jcont = int(Var.VARIDENT[ivar,0])
                if jcont>atm.NDUST+1:
                    sys.exit('error :: Variable outside limits',Var.VARIDENT[ivar,0],Var.VARIDENT[ivar,1],Var.VARIDENT[ivar,2])
                elif jcont==atm.NDUST:   #Para-H2
                    if flagh2p==True:
                        xref[:] = atm.PARAH2
                    else:
                        sys.exit('error :: Para-H2 is declared as variable but atmosphere is not from Giant Planet')
                elif jcont==atm.NDUST+1: #Fractional cloud cover
                    xref[:] = atm.FRAC
                else:
                    xref[:] = atm.DUST[:,jcont]

                ipar = atm.NVMR + jcont


        x1 = np.zeros(atm.NP)        

        if Var.VARIDENT[ivar,2]==-1:
#       Model -1. Continuous aerosol profile in particles cm-3
#       ***************************************************************

            xprof = np.zeros(Var.NXVAR[ivar])
            xprof[:] = Xn.XN[ix:ix+Var.NXVAR[ivar]]
            atm,xmap1 = modelm1(atm,ipar,xprof,MakePlot=True)
            xmap[ix:ix+Var.NXVAR[ivar],:,0:atm.NP] = xmap1[:,:,:]

            ix = ix + Var.NXVAR[ivar]

        if Var.VARIDENT[ivar,2]==0:
#       Model 0. Continuous profile
#       ***************************************************************

            xprof = np.zeros(Var.NXVAR[ivar])
            xprof[:] = Xn.XN[ix:ix+Var.NXVAR[ivar]]
            atm,xmap1 = model0(atm,ipar,xprof,MakePlot=True)
            xmap[ix:ix+Var.NXVAR[ivar],:,0:atm.NP] = xmap1[:,:,:]

            ix = ix + Var.NXVAR[ivar]

        elif Var.VARIDENT[ivar,2]==2:
#       Model 2. Scaling factor
#       ***************************************************************

            atm,xmap1 = model2(atm,ipar,Xn.XN[ix],MakePlot=True)
            xmap[ix:ix+Var.NXVAR[ivar],:,0:atm.NP] = xmap1[:,:,:]

            ix = ix + Var.NXVAR[ivar]

        elif Var.VARIDENT[ivar,2]==3:
#       Model 2. Log scaling factor
#       ***************************************************************

            atm,xmap1 = model3(atm,ipar,Xn.XN[ix],MakePlot=True)
            xmap[ix:ix+Var.NXVAR[ivar],:,0:atm.NP] = xmap1[:,:,:]

            ix = ix + Var.NXVAR[ivar]

        elif Var.VARIDENT[ivar,0]==228:
#       Model 228. Retrieval of instrument line shape for ACS-MIR (v1)
#       ***************************************************************
            ipar = -1
            ix = ix + Var.NXVAR[ivar]

        elif Var.VARIDENT[ivar,0]==229:
#       Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
#       ***************************************************************
            ipar = -1
            ix = ix + Var.NXVAR[ivar]

        elif Var.VARIDENT[ivar,0]==666:
#       Model 666. Retrieval of tangent pressure at given tangent height
#       ***************************************************************
            ipar = -1
            ix = ix + Var.NXVAR[ivar]

        else:
            print(Var.VARIDENT[ivar,0])
            sys.exit('error :: Model parameterisation has not yet been included')


    #Now check if any gas in the retrieval saturates
    

    #Adjust VMRs to add up to 1 if AMFORM=1
    if atm.AMFORM==1:
        atm.adjust_VMR()
        atm.calc_molwt()

    #Re-scale H/P based on the hydrostatic equilibrium equation
    if jhydro==0:
        #Then we modify the altitude levels and keep the pressures fixed
        atm.adjust_hydrostatH()
        atm.calc_grav()   #Updating the gravity values at the new heights
    else:
        #Then we modifify the pressure levels and keep the altitudes fixed
        atm.adjust_hydrostatP(htan,ptan)

    #Write out modified profiles
    atm.write_to_file()