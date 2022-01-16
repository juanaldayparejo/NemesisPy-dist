from NemesisPy.Profile import *
from NemesisPy.Models.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################

def calc_path_SO(Atmosphere,Scatter,Measurement,Layer):

    """
    FUNCTION NAME : calc_path_SO()

    DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files), 
                  different parameters in the Path class are changed to perform correctly
                  the radiative transfer calculations

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

        Layer,Path = calc_path_SO(Atmosphere,Scatter,Measurement,Layer)
 
    MODIFICATION HISTORY : Juan Alday (15/03/2021)
    """

    from NemesisPy.Layer import AtmCalc_0,Path_0
    from NemesisPy import find_nearest

    #Based on the new reference atmosphere, we split the atmosphere into layers
    ################################################################################

    #Limb or nadir observation?
    #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

    #Based on the new reference atmosphere, we split the atmosphere into layers
    #In solar occultation LAYANG = 90.0
    LAYANG = 90.0

    BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, CONT, LAYSF, DELH\
        = Layer.integrate(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, LAYANG=LAYANG, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST)

    #Based on the atmospheric layerinc, we calculate each required atmospheric path to model the measurements
    #############################################################################################################

    #Calculating the required paths that need to be calculated
    ITANHE = []
    for igeom in range(Measurement.NGEOM):

        base0,ibase = find_nearest(Layer.BASEH/1.0e3,Measurement.TANHE[igeom])
        if base0<=Measurement.TANHE[igeom]:
            ibasel = ibase
            ibaseh = ibase + 1
            if ibaseh==Layer.NLAY:
                ibaseh = ibase 
        else:
            ibasel = ibase - 1
            ibaseh = ibase

        ITANHE.append(ibasel)
        ITANHE.append(ibaseh)

    ITANHE = np.unique(ITANHE)

    NCALC = len(ITANHE)    #Number of calculations (geometries) to be performed
    AtmCalc_List = []
    for ICALC in range(NCALC):
        iAtmCalc = AtmCalc_0(Layer,LIMB=True,BOTLAY=ITANHE[ICALC],ANGLE=90.0,IPZEN=0,THERM=False)
        AtmCalc_List.append(iAtmCalc)
    
    #We initialise the total Path class, indicating that the calculations can be combined
    Path = Path_0(AtmCalc_List,COMBINE=True)

    return Layer,Path
