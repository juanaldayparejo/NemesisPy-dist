from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def subprofretg(runname,Variables,Measurement,Atmosphere,Scatter,Stellar,Surface,Layer,flagh2p):

    """
    FUNCTION NAME : subprogretg()

    DESCRIPTION : Updates the atmosphere based on the variables and parameterisations in the
                  state vector. Changes to other parameters in the model based on the variables 
                  and parameterisations in the state vector are also performed here. However,
                  the functional derivatives to these other parameters are not included since
                  they cannot be determined analytically.

    INPUTS :
    
        runname :: Name of the Nemesis run
        Variables :: Python class defining the parameterisations and state vector
        Measurement :: Python class defining the measurements 
        Atmosphere :: Python class defining the reference atmosphere
        Scatter :: Python class defining the parameters required for scattering calculations
        Stellar :: Python class defining the stellar spectrum
        Surface :: Python class defining the surface
        Layer :: Python class defining the layering scheme to be applied in the calculations

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

        xmap = subprofretg(runname,Variables,Measurement,Atmosphere,Scatter,Stellar,Surface,Layer,flagh2p)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)

    """

    #Modify profile via hydrostatic equation to make sure the atm is in hydrostatic equilibrium
    if Variables.JPRE==-1:
        jhydro = 0
        #Then we modify the altitude levels and keep the pressures fixed
        Atmosphere.adjust_hydrostatH()
        Atmosphere.calc_grav()   #Updating the gravity values at the new heights
    else:
        #Then we modifify the pressure levels and keep the altitudes fixed
        jhydro = 1
        for i in range(Variables.NVAR):
            if Variables.VARIDENT[i,0]==666:
                htan = Variables.VARPARAM[i,0] * 1000.
        ptan = np.exp(Variables.XN[Variables.JPRE]) * 101325.
        Atmosphere.adjust_hydrostatP(htan,ptan)

    #Adjust VMRs to add up to 1 if AMFORM=1 and re-calculate molecular weight in atmosphere
    if Atmosphere.AMFORM==1:
        Atmosphere.adjust_VMR()
        Atmosphere.calc_molwt()
    elif Atmosphere.AMFORM==2:
        Atmosphere.calc_molwt()

    #Calculate atmospheric density
    rho = Atmosphere.calc_rho() #kg/m3
    
    #Initialising xmap
    xmap = np.zeros((Variables.NX,Atmosphere.NVMR+2+Atmosphere.NDUST,Atmosphere.NP))

    #Going through the different variables an updating the atmosphere accordingly
    ix = 0
    for ivar in range(Variables.NVAR):

        if Variables.VARIDENT[ivar,2]<=100:
            
            #Reading the atmospheric profile which is going to be changed by the current variable
            xref = np.zeros([Atmosphere.NP])

            if Variables.VARIDENT[ivar,0]==0:     #Temperature is to be retrieved
                xref[:] = Atmosphere.T
                ipar = Atmosphere.NVMR
            elif Variables.VARIDENT[ivar,0]>0:    #Gas VMR is to be retrieved
                jvmr = np.where( (np.array(Atmosphere.ID)==Variables.VARIDENT[ivar,0]) & (np.array(Atmosphere.ISO)==Variables.VARIDENT[ivar,1]) )
                jvmr = int(jvmr[0])
                xref[:] = Atmosphere.VMR[:,jvmr]
                ipar = jvmr
            elif Variables.VARIDENT[ivar,0]<0:  
                jcont = -int(Variables.VARIDENT[ivar,0])
                if jcont>Atmosphere.NDUST+2:
                    sys.exit('error :: Variable outside limits',Variables.VARIDENT[ivar,0],Variables.VARIDENT[ivar,1],Variables.VARIDENT[ivar,2])
                elif jcont==Atmosphere.NDUST+1:   #Para-H2
                    if flagh2p==True:
                        xref[:] = Atmosphere.PARAH2
                    else:
                        sys.exit('error :: Para-H2 is declared as variable but atmosphere is not from Giant Planet')
                elif abs(jcont)==Atmosphere.NDUST+2: #Fractional cloud cover
                    xref[:] = Atmosphere.FRAC
                else:
                    xref[:] = Atmosphere.DUST[:,jcont-1]

                ipar = Atmosphere.NVMR + jcont

        x1 = np.zeros(Atmosphere.NP)        

        if Variables.VARIDENT[ivar,2]==-1:
#       Model -1. Continuous aerosol profile in particles cm-3
#       ***************************************************************

            xprof = np.zeros(Variables.NXVAR[ivar])
            xprof[:] = Variables.XN[ix:ix+Variables.NXVAR[ivar]]
            Atmosphere,xmap1 = modelm1(Atmosphere,ipar,xprof)
            xmap[ix:ix+Variables.NXVAR[ivar],:,0:Atmosphere.NP] = xmap1[:,:,:]

            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,2]==0:
#       Model 0. Continuous profile
#       ***************************************************************

            xprof = np.zeros(Variables.NXVAR[ivar])
            xprof[:] = Variables.XN[ix:ix+Variables.NXVAR[ivar]]
            Atmosphere,xmap1 = model0(Atmosphere,ipar,xprof)
            xmap[ix:ix+Variables.NXVAR[ivar],:,0:Atmosphere.NP] = xmap1[:,:,:]

            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,2]==2:
#       Model 2. Scaling factor
#       ***************************************************************

            Atmosphere,xmap1 = model2(Atmosphere,ipar,Variables.XN[ix])
            xmap[ix:ix+Variables.NXVAR[ivar],:,0:Atmosphere.NP] = xmap1[:,:,:]

            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,2]==3:
#       Model 2. Log scaling factor
#       ***************************************************************

            Atmosphere,xmap1 = model3(Atmosphere,ipar,Variables.XN[ix])
            xmap[ix:ix+Variables.NXVAR[ivar],:,0:Atmosphere.NP] = xmap1[:,:,:]

            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==228:
#       Model 228. Retrieval of instrument line shape for ACS-MIR (v1)
#       ***************************************************************
            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==229:
#       Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
#       ***************************************************************

            par1 = Variables.XN[ix]
            par2 = Variables.XN[ix+1]
            par3 = Variables.XN[ix+2]
            par4 = Variables.XN[ix+3]
            par5 = Variables.XN[ix+4]
            par6 = Variables.XN[ix+5]
            par7 = Variables.XN[ix+6]

            Measurement = model229(Measurement,par1,par2,par3,par4,par5,par6,par7)

            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==231:
#       Model 231. Continuum addition to transmission spectra using a linearly varying scaling factor
#       ***************************************************************

            #The computed transmission spectra is multiplied by TRANS = TRANS0 * (T0 + T1*(WAVE-WAVE0))
            #Where the parameters to fit are T0 and T1

            #The effect of this model takes place after the computation of the spectra in CIRSrad!
            if int(Variables.NXVAR[ivar]/2)!=Measurement.NGEOM:
                sys.exit('error using Model 231 :: The number of levels for the addition of continuum must be the same as NGEOM')

            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==666:
#       Model 666. Retrieval of tangent pressure at given tangent height
#       ***************************************************************
            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==667:
#       Model 667. Retrieval of dillusion factor to account for thermal gradients in planets
#       ***************************************************************
            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        else:
            print('error in Variable ',Variables.VARIDENT[ivar,0],Variables.VARIDENT[ivar,1],Variables.VARIDENT[ivar,2])
            sys.exit('error :: Model parameterisation has not yet been included')


    #Now check if any gas in the retrieval saturates

    #Adjust VMRs to add up to 1 if AMFORM=1
    if Atmosphere.AMFORM==1:
        Atmosphere.adjust_VMR()
        Atmosphere.calc_molwt()

    #Re-scale H/P based on the hydrostatic equilibrium equation
    if jhydro==0:
        #Then we modify the altitude levels and keep the pressures fixed
        Atmosphere.adjust_hydrostatH()
        Atmosphere.calc_grav()   #Updating the gravity values at the new heights
    else:
        #Then we modifify the pressure levels and keep the altitudes fixed
        Atmosphere.adjust_hydrostatP(htan,ptan)

    #Write out modified profiles to .prf file
    #Atmosphere.write_to_file()

    return xmap