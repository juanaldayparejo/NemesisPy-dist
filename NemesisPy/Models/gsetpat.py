from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def gsetpat(runname,Variables,Measurement,Atmosphere,Scatter,Stellar,Surface,Layer,flagh2p):

    """
    FUNCTION NAME : gsetpat()

    DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files), 
                  different parameters in the Path class are changed to perform correctly
                  the radiative transfer calculations

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

        Path :: Python class defining the calculation type and the path

    CALLING SEQUENCE:

        Path = gsetpat(runname,Variables,Measurement,Atmosphere,Scatter,Stellar,Surface,Layer,Path)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)
    """

    #Based on the new reference atmosphere, we split the atmosphere into layers
    ################################################################################

    #Limb or nadir observation?
    #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

    LAYANG = 0.0
    if Scatter.EMISS_ANG<0.0:
        Layer.LAYHT = Scatter.SOL_ANG
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

    if Scatter.EMISS_ANG>=0.0:
        nadir=True
        angle=EMISS_ANG
        botlay=0
    else:
        limb=True
        angle=90.0
        botlay=0

    if Scatter.ISCAT==0:   #No scattering
        if Measurement.IFORM==4:  #Atmospheric transmission multiplied by solar flux (no thermal emission then)
            therm=False
        else:
            therm=True
        scatter=False
    elif Scatter.ISCAT==1: #Multiple scattering
        therm=False
        scatter=True
    elif Scatter.ISCAT==2: #Internal scattered radiation field
        therm=False
        scatter=True
        nearlimb=True
    elif Scatter.ISCAT==3: #Single scattering in plane-parallel atmosphere
        therm=False
        single=True
    elif Scatter.ISCAT==4: #Single scattering in spherical atmosphere
        therm=False
        sphsingle=True


    #Performing the calculation of the atmospheric path
    ##############################################################################

    #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
    NCALC = 1    #Number of calculations (geometries) to be performed
    AtmCalc_List = []
    for ICALC in range(NCALC):
        iAtmCalc = AtmCalc_0(Layer,LIMB=limb,NADIR=nadir,BOTLAY=botlay,ANGLE=angle,IPZEN=ipzen,)
        AtmCalc_List.append(iAtmCalc)

    #We initialise the total Path class, indicating that the calculations can be combined
    Path = Path_0(AtmCalc_List,COMBINE=False)

    return Path
