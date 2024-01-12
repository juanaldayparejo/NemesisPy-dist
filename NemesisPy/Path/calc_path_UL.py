from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def calc_path_UL(Atmosphere,Scatter,Measurement,Layer):

    """
    FUNCTION NAME : calc_path()

    DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files), 
                  different parameters in the Path class are changed to perform correctly
                  the radiative transfer calculations
                  
                  This version is optimised for including several paths in the same atmospheric calculation

    INPUTS :
    
        Atmosphere :: Python class defining the reference atmosphere
        Scatter :: Python class defining the parameters required for scattering calculations
        Measurement :: Python class defining the measurements and observations
        Layer :: Python class defining the atmospheric layering scheme for the calculation

    OPTIONAL INPUTS: none
            
    OUTPUTS : 

        Layer :: Python class after computing the layering scheme for the radiative transfer calculations
        Path :: Python class defining the calculation type and the path

    CALLING SEQUENCE:

        Layer,Path = calc_path(Atmosphere,Scatter,Layer)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)
    """

    from NemesisPy.Path import AtmCalc_0,Path_0

    #Based on the new reference atmosphere, we split the atmosphere into layers
    ################################################################################

    #Limb or nadir observation?
    #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

    LAYANG = 0.0
    if Scatter.EMISS_ANG<0.0:
        Layer.LAYHT = Scatter.SOL_ANG * 1.0e3
        LAYANG = 90.0

    BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH\
        = Layer.integrate(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)

    #Setting the flags for the Path and calculation types
    ##############################################################################

    limb = False
    nadir = False
    ipzen = 0
    therm = False
    wf = False
    netflux = False
    outflux = False
    botflux = False
    upflux = False
    cg = False
    hemisphere = False
    nearlimb = False
    single = False
    sphsingle = False
    scatter = False
    broad = False
    absorb = False
    binbb = True

    if Scatter.EMISS_ANG>=90.0:
        limb=False
        nadir=True
        angle=Scatter.EMISS_ANG
        botlay=0
    else:
        sys.exit('error in calc_path_UL :: ')


        
    if self.ISCAT!=1:
        print('warning in calc_path_UL :: This version of the code is meant to use multiple scattering')
        print('turning multiple scattering ON :: ISCAT = 1')
        ScatterX.ISCAT = 1

    if Scatter.ISCAT==1: #Multiple scattering
        therm=False
        scatter=True

    #Performing the calculation of the atmospheric path
    ##############################################################################

    #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
    NCALC = 1    #Number of calculations (geometries) to be performed
    AtmCalc_List = []
    iAtmCalc = AtmCalc_0(Layer,LIMB=limb,NADIR=nadir,BOTLAY=botlay,ANGLE=angle,IPZEN=ipzen,\
                         THERM=therm,WF=wf,NETFLUX=netflux,OUTFLUX=outflux,BOTFLUX=botflux,UPFLUX=upflux,\
                         CG=cg,HEMISPHERE=hemisphere,NEARLIMB=nearlimb,SINGLE=single,SPHSINGLE=sphsingle,\
                         SCATTER=scatter,BROAD=broad,ABSORB=absorb,BINBB=binbb)
    AtmCalc_List.append(iAtmCalc)

    #We initialise the total Path class, indicating that the calculations can be combined
    Path = Path_0(AtmCalc_List,COMBINE=True)

    return Layer,Path
