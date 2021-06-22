#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from Profile import Atmosphere_1
from Layer import Layer
from NemesisPy.Models.Models import *
from NemesisPy.Models.subprofretg import subprofretg
from NemesisPy.Data import *
from NemesisPy.Files.Files import *
import matplotlib.pyplot as plt
###############################################################################
#                                                                             #
#                             ATMOSPHERIC input                               #
#                                                                             #
###############################################################################
#H = np.array([0.,178.74 ,333.773,460.83 ,572.974,680.655,
#        787.549,894.526,1001.764,1109.3  ])*1e3
 
#P = np.array([1.9739e+01,3.9373e+00,7.8539e-01,1.5666e-01,3.1250e-02,
#       6.2336e-03,1.2434e-03,2.4804e-04,4.9476e-05,9.8692e-06])*101325.

#T = np.array([1529.667,1408.619,1128.838,942.708,879.659,864.962,
#        861.943,861.342,861.222,861.199])

# H2O,He,H2
#ID, ISO = [1,40,39], [1,0,0]
#NVMR = len(ISO)
#VMR = np.array([[0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915],
#       [0.001,0.14985,0.84915]])


#NP  = 10
#runname, LATITUDE, IPLANET, AMFORM = 'wasp43b', 45.0, 4, 1
#Atm = Atmosphere_1(runname=runname,NP=NP,NVMR=NVMR,ID=ID,ISO=ISO,
#                   LATITUDE=LATITUDE,IPLANET=IPLANET,AMFORM=AMFORM)
#Atm.edit_H(H)
#Atm.edit_P(P)
#Atm.edit_T(T)
#Atm.edit_VMR(VMR)
#Atm.calc_molwt()
#Atm.calc_grav()

runname = 'cirstest'
Atm = read_ref(runname)
Atm = read_aerosol(Atm=Atm)

###############################################################################
#                                                                             #
#                               MODEL input                                   #
#                                                                             #
###############################################################################

Var,Xn = read_apr(runname,Atm.NP)

ispace = 0
iscat = 0
xlat = 0.0
xlon = 0.0
flagh2p = False

xmap = subprofretg(runname,Atm,ispace,iscat,xlat,xlon,Var,Xn,flagh2p)
