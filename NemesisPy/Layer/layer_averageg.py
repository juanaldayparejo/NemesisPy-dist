import numpy as np
from scipy.integrate import simps
from .interp import interp,interpg
import sys
sys.path.append('../')
from NemesisPy.Data import *

k_B = 1.38065e-23

def layer_averageg(RADIUS, H, P, T, ID, VMR, DUST, BASEH, BASEP,
                  LAYANG=0.0, LAYINT=0, LAYHT=0.0, NINT=101, AMFORM=1):
    """
    Calculates average layer properties.
    Takes an atmosphere profile and a layering shceme specified by
    layer base altitudes and pressures and returns averaged height,
    pressure, temperature, VMR for each layer.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Input profile heights
    @param P: 1D array
        Input profile pressures
    @param T: 1D array
        Input profile temperatures
    @param ID: 1D array
        Gas identifiers.
    @param VMR: 2D array
        VMR[i,j] is Volume Mixing Ratio of gas j at vertical point i
        the column j corresponds to the gas with RADTRANS ID ID[j].
    @param DUST: 2D array
        DUST[i,j] is dust density of dust popoulation j at vertical point i  (particles/m3)
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
    @param LAYINT: int
        Layer integration scheme
        0 = use properties at mid path
        1 = use absorber amount weighted average values
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NINT: int
        Number of integration points to be used if LAYINT=1.
    @param AMFORM: int,
        Flag indicating how the molecular weight must be calculated:
        0 - The mean molecular weight of the atmosphere is passed in XMOLWT
        1 - The mean molecular weight of each layer is calculated (atmosphere previously adjusted to have sum(VMR)=1.0)
        2 - The mean molecular weight of each layer is calculated (VMRs do not necessarily add up to 1.0)

    Returns
    -------
    @param HEIGHT: 1D array
        Representative height for each layer
    @param PRESS: 1D array
        Representative pressure for each layer
    @param TEMP: 1D array
        Representative pressure for each layer
    @param TOTAM: 1D array
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        number of molecules per area.
    @param AMOUNT: 1D array
        Representative absorber amounts of each gas at each layer.
        AMOUNT[I,J] is the representative number of gas J molecules
        in layer I in the form of number of molecules per area.
    @param PP: 1D array
        Representative partial pressure for each gas at each layer.
        PP[I,J] is the representative partial pressure of gas J in layer I.
    @param DELH: 1D array
        Layer thicnkness.
    @param BASET: 1D array
        Layer base temperature.
    @param LAYSF: 1D array
        Layer scaling factor.
    @param DTE: 2D array
        Matrix relating the temperature in each layer (TEMP) to the input temperature profile (T)
    @param DAM: 2D array
        Matrix relating the absorber amounts in each layer (AMOUNT) to the input VMR profile (VMR)
    @param DCO: 2D array
        Matrix relating the dust amounts in each layer (CONT) to the input dust profile (DUST)
    """

    # Calculate layer geometric properties
    NLAY = len(BASEH)
    NPRO = len(H)
    DELH = np.zeros(NLAY)           # layer thickness
    DELH[0:-1] = (BASEH[1:]-BASEH[:-1])
    DELH[-1] = (H[-1]-BASEH[-1])

    sin = np.sin(LAYANG*np.pi/180)  # sin(viewing angle)
    cos = np.cos(LAYANG*np.pi/180)  # cos(viewing angle)

    z0 = RADIUS + LAYHT    # distance from planet centre to lowest layer base
    zmax = RADIUS+H[-1]    # maximum height defined in the profile

    SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
    BASES = np.sqrt((RADIUS+BASEH)**2-(z0*sin)**2)-z0*cos # Path lengths at base of layer
    DELS = np.zeros(NLAY)           # path length in each layer
    DELS[0:-1] = (BASES[1:]-BASES[:-1])
    DELS[-1] = (SMAX-BASES[-1])
    LAYSF = DELS/DELH               # Layer Scaling Factor
    BASET = interp(H,T,BASEH)       # layer base temperature

    # Note number of: profile points, layers, VMRs, aerosol types, flags
    if VMR.ndim == 1:
        NVMR = 1
    else:
        NVMR = len(VMR[0])

    if DUST.ndim == 1:
        NDUST = 1
    else:
        NDUST = len(DUST[0])

    # HEIGHT = average layer height
    HEIGHT = np.zeros(NLAY)
    # PRESS = average layer pressure
    PRESS  = np.zeros(NLAY)
    # TEMP = average layer temperature
    TEMP   = np.zeros(NLAY)
    # TOTAM = total no. of molecules/aera
    TOTAM  = np.zeros(NLAY)
    # DUDS = no. of molecules per area per distance
    DUDS   = np.zeros(NLAY)
    # AMOUNT = no. of molecules/aera for each gas (molecules/m2)
    AMOUNT = np.zeros((NLAY, NVMR))
    # PP = gas partial pressures
    PP     = np.zeros((NLAY, NVMR))
    # MOLWT = mean molecular weight
    MOLWT  = np.zeros(NLAY)
    # CONT = no. of particles/area for each dust population (particles/m2)
    CONT = np.zeros((NLAY,NDUST))
    #DTE = matrix to relate the temperature in each layer (TEMP) to the temperature in the input profiles (T)
    DTE = np.zeros((NLAY, NPRO))
    #DCO = matrix to relate the dust abundance in each layer (CONT) to the dust abundance in the input profiles (DUST)
    DCO = np.zeros((NLAY, NPRO))
    #DAM = matrix to relate the gaseous abundance in each layer (AMOUNT) to the gas VMR in the input profiles (VMR)
    DAM = np.zeros((NLAY, NPRO))

    #Defining the weights for the integration with the Simpson's rule
    w = np.ones(NINT) * 4.
    w[::2] = 2.
    w[0] = 1.0
    w[NINT-1] = 1.0

    # Calculate average properties depending on intergration type
    if LAYINT == 0:
        # use layer properties at half path length in each layer
        S = np.zeros(NLAY)
        S[:-1] = (BASES[1:] + BASES[:-1])/2
        S[-1] = (SMAX+BASES[-1])/2
        # Derive other properties from S
        HEIGHT = np.sqrt(S**2+z0**2+2*S*z0*cos) - RADIUS
        PRESS = interp(H,P,HEIGHT)
        #TEMP = interp(H,T,HEIGHT)
        TEMP,J,F = interpg(H,T,HEIGHT)
        for ilay in range(NLAY):
            DTE[ilay,J[ilay]] = DTE[ilay,J[ilay]] + (1.0-F[ilay])
            DTE[ilay,J[ilay]+1] = DTE[ilay,J[ilay]+1] + (F[ilay])

        # Ideal gas law: N/(Area*Path_length) = P/(k_B*T)
        DUDS = PRESS/(k_B*TEMP)
        TOTAM = DUDS*DELS
        # Use the volume mixing ratio information
        if VMR.ndim > 1:
            AMOUNT = np.zeros((NLAY, NVMR))
            for J in range(NVMR):
                AMOUNT[:,J],JJ,F = interpg(H, VMR[:,J], HEIGHT)
            PP = (AMOUNT.T * PRESS).T
            AMOUNT = (AMOUNT.T * TOTAM).T
        else:
            AMOUNT,JJ,F = interpg(H, VMR, HEIGHT)
            PP = AMOUNT * PRESS
            AMOUNT = AMOUNT * TOTAM
            if AMFORM==0:
                sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
            else:
                for I in range(NLAY):
                    MOLWT[I] = Calc_mmw(VMR[I], ID)

        for ilay in range(NLAY):
            DAM[ilay,JJ[ilay]] = DAM[ilay,JJ[ilay]] + (1.0-F[ilay])*TOTAM[ilay]
            DAM[ilay,JJ[ilay]+1] = DAM[ilay,JJ[ilay]+1] + (F[ilay])*TOTAM[ilay]

        #Use the dust density information
        if DUST.ndim > 1:
            for J in range(NDUST):
                DD,JJ,F = interpg(H, DUST[:,J],HEIGHT)  
                CONT[:,J] = DD * DELS

        else:
            #DD = interp(H, DUST,HEIGHT)  
            DD,JJ,F = interpg(H, DUST,HEIGHT)
            CONT = DD * DELS

        for ilay in range(NLAY):
            DCO[ilay,JJ[ilay]] = DCO[ilay,JJ[ilay]] + (1.0-F[ilay])
            DCO[ilay,JJ[ilay]+1] = DCO[ilay,JJ[ilay]+1] + (F[ilay])

    elif LAYINT == 1:
        # Curtis-Godson equivalent path for a gas with constant mixing ratio
        for I in range(NLAY):
            S0 = BASES[I]
            if I < NLAY-1:
                S1 = BASES[I+1]
            else:
                S1 = SMAX
            # sub-divide each layer into NINT layers
            S = np.linspace(S0, S1, NINT)
            h = np.sqrt(S**2+z0**2+2*S*z0*cos)-RADIUS
            p = interp(H,P,h)
            temp,JJ,F = interpg(H,T,h)
            duds = p/(k_B*temp)

            amount = np.zeros((NINT, NVMR))
            molwt = np.zeros(NINT)

            TOTAM[I] = simps(duds,S)
            HEIGHT[I]  = simps(h*duds,S)/TOTAM[I]
            PRESS[I] = simps(p*duds,S)/TOTAM[I]
            TEMP[I]  = simps(temp*duds,S)/TOTAM[I]

            for iint in range(NINT):
                DTE[I,JJ[iint]] = DTE[I,JJ[iint]] + (1.-F[iint])*w[iint]*duds[iint]
                DTE[I,JJ[iint]+1] = DTE[I,JJ[iint]+1] + (F[iint])*w[iint]*duds[iint]

            if VMR.ndim > 1:
                amount = np.zeros((NINT, NVMR))
                for J in range(NVMR):
                    amount[:,J],JJ,F = interpg(H, VMR[:,J], h)
                    AMOUNT[I,J] = simps(amount[:,J]*duds,S)
                pp = (amount.T * p).T     # gas partial pressures
                for J in range(NVMR):
                    PP[I, J] = simps(pp[:,J]*duds,S)/TOTAM[I]
                
                if AMFORM==0:
                    sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
                else:
                    for K in range(NINT):
                        molwt[K] = Calc_mmw(amount[K,:], ID)
                    MOLWT[I] = simps(molwt*duds,S)/TOTAM[I]
            else:
                amount,JJ,F = interpg(H, VMR, h)
                pp = amount * p
                AMOUNT[I] = simps(amount*duds,S)
                PP[I] = simps(pp*duds,S)/TOTAM[I]

                if AMFORM==0:
                    sys.exit('error :: AMFORM=0 needs to be implemented in Layer.py')
                else:
                    for K in range(NINT):
                        molwt[K] = Calc_mmw(amount[K], ID)
                    MOLWT[I] = simps(molwt*duds,S)/TOTAM[I]

            for iint in range(NINT):
                DAM[I,JJ[iint]] = DAM[I,JJ[iint]] + (1.-F[iint])*duds[iint]*w[iint]
                DAM[I,JJ[iint]+1] = DAM[I,JJ[iint]+1] + (F[iint])*duds[iint]*w[iint]

            if DUST.ndim > 1:
                dd = np.zeros((NINT,NDUST))
                for J in range(NDUST):
                    dd[:,J],JJ,F = interpg(H, DUST[:,J], h)
                    CONT[I,J] = simps(dd[:,J],S)
            else:
                dd,JJ,F = interpg(H, DUST, h) 
                CONT[I] = simps(dd,S)

            for iint in range(NINT):
                DCO[I,JJ[iint]] = DCO[I,JJ[iint]] + (1.-F[iint])*w[iint]
                DCO[I,JJ[iint]+1] = DCO[I,JJ[iint]+1] + (F[iint])*w[iint]

    #Finishing the integration for the matrices
    for IPRO in range(NPRO):
        DTE[:,IPRO] = DTE[:,IPRO] * DELS / 100. / 3. / TOTAM
        DAM[:,IPRO] = DAM[:,IPRO] * DELS / 100. / 3.
        DCO[:,IPRO] = DCO[:,IPRO] * DELS / 100. / 3.

    # Scale back to vertical layers
    TOTAM = TOTAM / LAYSF
    if VMR.ndim > 1:
        AMOUNT = (AMOUNT.T * LAYSF**-1 ).T
    else:
        AMOUNT = AMOUNT/LAYSF

    if DUST.ndim > 1:
        CONT = (CONT.T * LAYSF**-1 ).T
    else:
        CONT = CONT/LAYSF

    for IPRO in range(NPRO):
        DAM[:,IPRO] = DAM[:,IPRO] / LAYSF
        DCO[:,IPRO] = DCO[:,IPRO] / LAYSF

    return HEIGHT,PRESS,TEMP,TOTAM,AMOUNT,PP,CONT,DELH,BASET,LAYSF,DTE,DAM,DCO
