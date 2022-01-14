from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def subprofretg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,Layer,flagh2p):

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
        Spectroscopy :: Python class defining the spectroscopic parameters
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

        elif Variables.VARIDENT[ivar,2]==9:
#       Model 9. Simple cloud represented by base height, fractional scale height
#                and the total integrated cloud density
#       ***************************************************************

            tau = np.exp(Variables.XN[ix])  #Integrated dust column-density
            fsh = np.exp(Variables.XN[ix+1]) #Fractional scale height
            href = Variables.XN[ix+2] #Base height (km)

            Atmosphere,xmap1 = model9(Atmosphere,ipar,href,fsh,tau)
            xmap[ix:ix+Variables.NXVAR[ivar],:,0:Atmosphere.NP] = xmap1[:,:,:]

            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==228:
#       Model 228. Retrieval of instrument line shape for ACS-MIR and wavelength calibration
#       **************************************************************************************

            V0 = Variables.XN[ix]
            C0 = Variables.XN[ix+1]
            C1 = Variables.XN[ix+2]
            C2 = Variables.XN[ix+3]
            P0 = Variables.XN[ix+4]
            P1 = Variables.XN[ix+5]
            P2 = Variables.XN[ix+6]
            P3 = Variables.XN[ix+7]

            Measurement,Spectroscopy = model228(Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3)

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

        elif Variables.VARIDENT[ivar,0]==230:
#       Model 230. Retrieval of multiple instrument line shapes for ACS-MIR
#       ***************************************************************

            nwindows = int(Variables.VARPARAM[ivar,0])
            liml = np.zeros(nwindows)
            limh = np.zeros(nwindows)
            i0 = 1
            for iwin in range(nwindows):
                liml[iwin] = Variables.VARPARAM[ivar,i0]
                limh[iwin] = Variables.VARPARAM[ivar,i0+1]
                i0 = i0 + 2

            par1 = np.zeros((7,nwindows))
            for iwin in range(nwindows):
                for jwin in range(7):
                    par1[jwin,iwin] = Variables.XN[ix]
                    ix = ix + 1

            Measurement = model230(Measurement,nwindows,liml,limh,par1)

            ipar = -1

        elif Variables.VARIDENT[ivar,0]==231:
#       Model 231. Continuum addition to transmission spectra using a varying scaling factor (given a polynomial of degree N)
#       ***************************************************************

            #The computed transmission spectra is multiplied by R = R0 * POL
            #Where POL is given by POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

            #The effect of this model takes place after the computation of the spectra in CIRSrad!
            if int(Variables.VARPARAM[ivar,0])!=Measurement.NGEOM:
                sys.exit('error using Model 231 :: The number of levels for the addition of continuum must be the same as NGEOM')

            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==232:
#       Model 232. Continuum addition to transmission spectra using the angstrom coefficient
#       ***************************************************************

            #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
            #Where the parameters to fit are TAU0 and ALPHA

            #The effect of this model takes place after the computation of the spectra in CIRSrad!
            if int(Variables.NXVAR[ivar]/2)!=Measurement.NGEOM:
                sys.exit('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==233:
#       Model 233. Continuum addition to transmission spectra using a variable angstrom coefficient
#       ***************************************************************

            #The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
            #Where the aerosol opacity is modelled following

            # np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

            #The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
            #233 converges to model 232 when a2=0.

            #The effect of this model takes place after the computation of the spectra in CIRSrad!
            if int(Variables.NXVAR[ivar]/3)!=Measurement.NGEOM:
                sys.exit('error using Model 233 :: The number of levels for the addition of continuum must be the same as NGEOM')

            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==446:
#       Model 446. model for retrieving an aerosol density profile + aerosol particle size (log-normal distribution)
#       ***************************************************************

            #This model fits a continuous vertical profile for the aerosol density and the particle size, which
            #is assumed to follow a log-normal distribution

            if int(Variables.NXVAR[ivar]/2)!=Atmosphere.NP:
                sys.exit('error using Model 446 :: The number of levels for the addition of continuum must be the same as NPRO')

            aero_dens = np.zeros(Atmosphere.NP)
            aero_rsize = np.zeros(Atmosphere.NP)
            aero_dens[:] = np.exp(Variables.XN[ix:ix+Atmosphere.NP])
            aero_rsize[:] = np.exp(Variables.XN[ix+Atmosphere.NP:ix+Atmosphere.NP+Atmosphere.NP])
            #aero_rsize[:] = Variables.XN[ix+Atmosphere.NP:ix+Atmosphere.NP+Atmosphere.NP]

            nlevel = int(Variables.VARPARAM[ivar,0])
            aero_id = int(Variables.VARPARAM[ivar,1])
            sigma_rsize = Variables.VARPARAM[ivar,2]
            idust0 = int(Variables.VARPARAM[ivar,3])
            WaveNorm = Variables.VARPARAM[ivar,4]

            #Reading the refractive index from the dictionary
            Scatter.WAVER = aerosol_info[str(aero_id)]["wave"]
            Scatter.REFIND_REAL = aerosol_info[str(aero_id)]["refind_real"]
            Scatter.REFIND_IM = aerosol_info[str(aero_id)]["refind_im"]

            if Atmosphere.NDUST<nlevel:
                sys.exit('error in Model 446 :: The number of aerosol populations must at least be equal to the number of altitude levels')

            Atmosphere,Scatter,xmap1 = model446(Atmosphere,Scatter,idust0,aero_id,aero_dens,aero_rsize,sigma_rsize,WaveNorm)

            xmap[ix:ix+Variables.NXVAR[ivar],:,0:Atmosphere.NP] = xmap1[:,:,:]

            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==666:
#       Model 666. Retrieval of tangent pressure at given tangent height
#       ***************************************************************
            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==667:
#       Model 667. Retrieval of dilution factor to account for thermal gradients in planets
#       ***************************************************************
            ipar = -1
            ix = ix + Variables.NXVAR[ivar]

        elif Variables.VARIDENT[ivar,0]==999:
#       Model 999. Retrieval of surface temperature
#       ***************************************************************

            tsurf = Variables.XN[ix]
            Surface.TSURF = tsurf

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