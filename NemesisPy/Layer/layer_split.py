#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from .interp import interp
from NemesisPy.Files import *
Test = False

def layer_split(RADIUS, H, P, LAYANG=0.0, LAYHT=0.0, NLAY=20,
        LAYTYP=1, INTERTYP=1, H_base=None, P_base=None):
    """
    Splits an atmosphere into NLAY layers.
    Takes a set of altitudes H with corresponding pressures P and returns
    the altitudes and pressures of the base of the layers.

    Inputs
    ------
    @param RADIUS: real
        Reference planetary radius where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    @param H: 1D array
        Heights at which the atmosphere profile is specified.
        (At altitude H[i] the pressure is P[i].)
    @param P: 1D array
        Pressures at which the atmosphere profile is specified.
        (At pressure P[i] the altitude is H[i].)
    @param LAYANG: real
        Zenith angle in degrees defined at LAYHT.
        Default 0.0 (nadir geometry). Only needed for layer type 3.
    @param LAYHT: real
        Height of the base of the lowest layer. Default 0.0.
    @param NLAY: int
        Number of layers to split the atmosphere into. Default 20.
    @param LAYTYP: int
        Integer specifying how to split up the layers. Default 1.
        0 = by equal changes in pressure
        1 = by equal changes in log pressure
        2 = by equal changes in height
        3 = by equal changes in path length at LAYANG
        4 = layer base pressure levels specified by P_base
        5 = layer base height levels specified by H_base
        Note 4 and 5 force NLAY = len(P_base) or len(H_base).
    @param H_base: 1D array
        Heights of the layer bases defined by user. Default None.
    @param P_base: 1D array
        Pressures of the layer bases defined by user. Default None.
    @param INTERTYP: int
        Interger specifying interpolation scheme.  Default 1.
        1=linear, 2=quadratic spline, 3=cubic spline

    Returns
    -------
    @param BASEH: 1D array
        Heights of the layer bases.
    @param BASEP: 1D array
        Pressures of the layer bases.
    """

    if LAYHT<H[0]:
        print('Warning from layer_split() :: LAYHT < H(0). Resetting LAYHT')
        LAYHT = H[0]

    #assert (LAYHT>=H[0]) and (LAYHT<H[-1]) , \
    #    'Lowest layer base height LAYHT not contained in atmospheric profile'
    assert not (H_base and P_base), \
        'Cannot input both layer base heights and base pressures'

    if LAYTYP == 0: # split by equal pressure intervals
        PBOT = interp(H,P,LAYHT,INTERTYP)  # pressure at base of lowest layer
        BASEP = np.linspace(PBOT,P[-1],NLAY+1)[:-1]
        BASEH = interp(P,H,BASEP,INTERTYP)

    elif LAYTYP == 1: # split by equal log pressure intervals
        PBOT = interp(H,P,LAYHT,INTERTYP)  # pressure at base of lowest layer
        BASEP = np.logspace(np.log10(PBOT),np.log10(P[-1]),NLAY+1)[:-1]
        BASEH = interp(P,H,BASEP,INTERTYP)

    elif LAYTYP == 2: # split by equal height intervals
        BASEH = np.linspace(H[0]+LAYHT, H[-1], NLAY+1)[:-1]
        BASEP = interp(H,P,BASEH,INTERTYP)

    elif LAYTYP == 3: # split by equal line-of-sight path intervals
        assert LAYANG<=90 and LAYANG>=0,\
            'Zennith angle should be in [0,90]'
        sin = np.sin(LAYANG*np.pi/180)      # sin(zenith angle)
        cos = np.cos(LAYANG*np.pi/180)      # cos(zenith angle)
        z0 = RADIUS + LAYHT                 # distance from centre to lowest layer's base
        zmax = RADIUS+H[-1]                 # maximum height
        SMAX = np.sqrt(zmax**2-(z0*sin)**2)-z0*cos # total path length
        BASES = np.linspace(0, SMAX, NLAY+1)[:-1]
        BASEH = np.sqrt(BASES**2+z0**2+2*BASES*z0*cos) - RADIUS
        logBASEP = interp(H,np.log(P),BASEH,INTERTYP)
        BASEP = np.exp(logBASEP)

    elif LAYTYP == 4: # split by specifying input base pressures
        assert P_base, 'Need input layer base pressures'
        assert  (P_base[-1] >= P[-1]) and (P_base[0] <= P[0]), \
            'Input layer base pressures out of range of atmosphere profile'
        BASEP = P_base
        NLAY = len(BASEP)
        BASEH = interp(P,H,BASEP,INTERTYP)

    elif LAYTYP == 5: # split by specifying input base heights
        NLAY,H_base = read_hlay()
        #assert H_base, 'Need input layer base heighs'
        #assert (H_base[-1] <= H[-1]) and (H_base[0] >= H[0]), \
        #    'Input layer base heights out of range of atmosphere profile'
        BASEH = H_base * 1.0e3
        NLAY = len(H_base)
        logBASEP = interp(H,np.log(P),BASEH,INTERTYP)
        BASEP = np.exp(logBASEP)


    else:
        raise('Layering scheme not defined')

    return BASEH, BASEP

def read_hlay():

    """
        FUNCTION NAME : read_hlay()
        
        DESCRIPTION : Read the height.lay file used to set the altitude of the base of the layers
                      in a Nemesis run
        
        INPUTS : None
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            nlay :: Number of layers in atmospheric model

        
        CALLING SEQUENCE:
        
            nlay,hbase = read_hlay()
        
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    f = open('height.lay','r')

    header = f.readline().split()

    s = f.readline().split()
    nlay = int(s[0])
    hbase = np.zeros(nlay)
    for i in range(nlay):
        s = f.readline().split()
        hbase[i] = float(s[0])

    return nlay,hbase

if Test == True:
    LAYTYP = 1
    LAYANG = 0
    NLAY = 10
    RADIUS = 74065.70e3

    H = np.array([0.,178.74 ,333.773,460.83 ,572.974,680.655,
          787.549,894.526,1001.764,1109.3  ])*1e3

    P = np.array([1.9739e+01,3.9373e+00,7.8539e-01,1.5666e-01,3.1250e-02,
          6.2336e-03,1.2434e-03,2.4804e-04,4.9476e-05,9.8692e-06])*101325

    T = np.array([1529.667,1408.619,1128.838,942.708,879.659,864.962,
            861.943,861.342,861.222,861.199])

    BASEH, BASEP = layer_split(RADIUS, H, P, LAYANG=LAYANG, LAYHT=0.0, NLAY=NLAY,
        LAYTYP=LAYTYP, INTERTYP=1, H_base=None, P_base=None)

    print('Layer type', LAYTYP)
    print('Layer angle', LAYANG)
    print('BASEH\n',BASEH*1e-3,'\n','BASEP\n',BASEP/1.013e5)
