#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from Profile import Atmosphere_1
from Layer import Layer
import matplotlib.pyplot as plt
###############################################################################
#                                                                             #
#                               MODEL input                                   #
#                                                                             #
###############################################################################
H = np.array([0.,178.74 ,333.773,460.83 ,572.974,680.655,
        787.549,894.526,1001.764,1109.3  ])*1e3

P = np.array([1.9739e+01,3.9373e+00,7.8539e-01,1.5666e-01,3.1250e-02,
       6.2336e-03,1.2434e-03,2.4804e-04,4.9476e-05,9.8692e-06])*101325

T = np.array([1529.667,1408.619,1128.838,942.708,879.659,864.962,
        861.943,861.342,861.222,861.199])

# H2O,He,H2
ID, ISO = [1,40,39], [0,0,0]
NVMR = len(ISO)
VMR = np.array([[0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915],
       [0.001,0.14985,0.84915]])

NP  = 10
runname, LATITUDE, IPLANET, AMFORM = 'wasp43b', 0.0, 0, 1
Atm = Atmosphere_1(runname=runname,NP=NP,NVMR=NVMR,ID=ID,ISO=ISO,
                    LATITUDE=LATITUDE,IPLANET=IPLANET,AMFORM=AMFORM)
Atm.edit_H(H)
Atm.edit_P(P)
Atm.edit_T(T)
Atm.edit_VMR(VMR)
###############################################################################
#                                                                             #
#                               Layer input                                   #
#                                                                             #
###############################################################################
# Calculate average layer properties
NLAY = 10
RADIUS = 74065.70e3
LAYANG = 0
LAYINT = 1
H,P,T = Atm.H, Atm.P, Atm.T
VMR = Atm.VMR
# DUST = np.zeros(10)
LAYTYP, AMFORM, LAYHT = 1, 1, 0.0
H_base, P_base, INTERTYP = None, None, 1

# layer properties
Layer = Layer(RADIUS=RADIUS, LAYTYP=LAYTYP, NLAY=NLAY, LAYINT=LAYINT, NINT=101,
              AMFORM=AMFORM, INTERTYP=INTERTYP, H_base=H_base, P_base=P_base)

BASEH, BASEP, BASET, HEIGHT, PRESS, TEMP, TOTAM, AMOUNT, PP, LAYSF, DELH\
    = Layer.integrate(H=H,P=P,T=T, LAYANG=LAYANG, ID=ID,VMR=VMR)

"""
print('{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} '.format(
'BASEH', 'DELH', "BASEP", 'BASET', 'TOTAM', 'PRESS', 'TEMP'),end="")
for j in range(NVMR):
    print('{:<8} {:<8}'.format('AMOUNT', 'PP'), end='')
for i in range(len(BASEH)):
    print('\n{} {:<8.2f} {:<8.2f} {:<8.3E} {:<8.0f} {:<8.3e} {:<8.3e} {:<8.0f} '.format(i+1,
    BASEH[i]/1e3, DELH[i]/1e3, BASEP[i]/101325, BASET[i],
    TOTAM[i]/1e4, PRESS[i]/101325, TEMP[i]),end="")
    for j in range(NVMR):
        print('{:<10.3E} {:<10.3E}'.format(AMOUNT[i,j]/1e4,PP[i,j]/101325),end="")
    print('\n')

NBASEH = np.array([   0.  ,  170.95,  319.07,  441.14,  547.66,  647.41,  744.11,
        838.79,  931.45, 1021.76])

NBASEP = np.array([1.9739e+01, 4.6262e+00, 1.0842e+00, 2.5410e-01, 5.9554e-02,
       1.3957e-02, 3.2711e-03, 7.6665e-04, 1.7968e-04, 4.2110e-05])

NBASET = np.array([1529.667, 1413.896, 1155.364,  971.556,  893.888,  869.5  ,
        863.17 ,  861.655,  861.301,  861.218])

NTOTAM = np.array([1.0312e+27, 2.1605e+26, 4.7421e+25, 1.0578e+25, 2.3625e+24,
       5.4042e+23, 1.2735e+23, 3.0836e+22, 7.6502e+21, 1.9399e+21])

NPRESS = np.array([1.3659e+01, 2.8635e+00, 6.3027e-01, 1.4252e-01, 3.3006e-02,
       7.8920e-03, 1.9317e-03, 4.7819e-04, 1.1864e-04, 2.9322e-05])

NTEMP = np.array([1483.088, 1311.001, 1076.866,  932.43 ,  880.159,  865.926,
        862.359,  861.481,  861.264,  861.211])

plt.plot(TOTAM*1e-4, BASEH*1e-3, label='Python')
plt.scatter(NTOTAM, NBASEH, label='Fortran', color='k')
plt.ylabel('height (km)')
plt.xlabel('total amount (no./cm^2)')
plt.legend()
"""
###############################################################################
#                                                                             #
#                               Driver input                                  #
#                                                                             #
###############################################################################
runname = 'test'
ICONV = 24
VMIN,DELV,NPOINT,FWHM=1.143,0.209844,17,0.000
WING, VREL = 0.000, 0.000
FLAGH2P, NCONT, FLAGC = 0, 1, 0
XFILE = '{}.xsc'.format(runname)
NLAYER, NPATH, NGAS = 10, 1, 6
IDGAS = [1,2,3,4,5,6]
ISOGAS = [0,0,0,0,0,0]
IPROC = [0,0,0,0,0,0]
DOP = np.zeros(NLAY)
"""
runname, VMIN, VMAX, dV, FWHM, ICONV, PAR1, PAR2, FLAGH2P, NCONT, FLAGC
XFILE,NLAYER, NPATH, NGAS
IDGAS[I],ISOGAS[I],IPROC[I]
BASEH[I],DELH[I],BASEP[I],BASET[I],TOTAM[I],PRESS[I],TEMP[I],DOP[I]
"""
f = open('{}.drv'.format(runname), 'w')
f.write('original name of this file: {}.drv\n'.format(runname))
# Write INTERVAL data
if ICONV >= 0 and ICONV < 10:
    f.write('{:<7.5f} {:<7.5f} {:<7.0f} {:<7.5f} :Vmin, dV, Npoints, FWHM-LBL\n'
            .format(VMIN,DELV,NPOINT,FWHM))
    f.write('{:<7.5f} {:<21.5f}   :wing continuum limit and overall limit\n'
            .format(WING,VREL))
elif ICONV >= 10 and ICONV < 20:
    f.write('{:<7.5f} {:<7.5f} {:<7.0f} {:<7.5f} :Vmin, dV, Npoints, FWHM-BAND\n'
            .format(VMIN,DELV,NPOINT,FWHM))
    f.write('{:<7.5f} {:<21.5f}   :Additional codes PAR1 and PAR2\n'
            .format(WING,VREL))
elif ICONV >= 20 and ICONV < 30:
    f.write('{:<7.5f} {:<7.5f} {:<7.0f} {:<7.5f} :Vmin, dV, Npoints, FWHM-CORRK\n'
            .format(VMIN,DELV,NPOINT,FWHM))
    f.write('{:<7.5f} {:<21.5f}   :Additional codes PAR1 and PAR2\n'
            .format(WING,VREL))
else:
    raise('ICONV out of range')
# Write opacity file
f.write('{}.kls\n'.format(runname))
# Write ICONV, FLAGH2P, NCONT, FLAGC
f.write('{:<7.0f} {:<7.0f} {:<7.0f} {:<7.0f} :spectral model code, FLAGH2P, NCONT, FLAGC\n'.format(
        ICONV, FLAGH2P, NCONT, FLAGC))
# Aerosol opacity data
if NCONT > 0:
    f.write('{:<28}    :Dust x-section file\n'.format(XFILE))

# LAYER DATA
# NLAYER, NPATH, NGAS
# Gas IDGAS, ISOGAS, IPROC
f.write('{:<7.0f} {:<7.0f} {:<14.0f}  :number of layers, paths and gases\n'
        .format(NLAYER, NPATH, NGAS))
for I in range(NGAS):
    f.write('{:<28.0f}    :identifier for gas\n'.format(IDGAS[I]))
    f.write('{:<7.0f} {:<21.0f}   :isotope ID and process parameter\n'.format(ISOGAS[I],IPROC[I]))

# Layer average properties
f.write('Format of layer data\n')
f.write(       '{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n'
        .format('Layer,', 'BaseH,','del H,','BaseP,','BaseT,','TOTAM,','Pressure,','Temp,','Doppler,',
        'AMOUNT,','PP,', 'continuum'))


# Write Layer Properties
for I in range(NLAYER):
    f.write('{:<6} {:<10.2f} {:<10.2f} {:<10.3E} {:<10.2f} {:<10.3e} {:<10.3e} {:<10.2f} {:<10.0f}\n'
        .format(I+1,BASEH[I]/1e3,DELH[I]/1e3,BASEP[I]/101325,BASET[I],TOTAM[I]/1e4,PRESS[I]/101325,TEMP[I],DOP[I]))
    for J in range(NVMR):
        f.write('{:<94} {:<10.3E} {:<10.3E}\n'.format(' ',AMOUNT[I,J]/1e4,PP[I,J]/101325))
        # IF(NCONT.GT.0)WRITE(4,513)(CONT(J,I),J=1,NCONT)

# Write Path Properties
# ': Nlayers, model & error limit, path'
for I in NPATH:
    f.write('{:<7.0f} {:<7.0f} {:<7.0f}  :Nlayers, IMOD, Error limit, path {:<7.0f}\n'
        .format(NLAYIN[I], IMOD[I], ERRLIM[I], I))
    for J in range(NLAYIN[I]):
        f.write('{:<7.0f} {:<7.0f} {:<7.0f} {:<7.0f}:layer or path, emission temp, scale'
            .format(J, LAYINC[J,I], EMTEMP[J,I], SCALE[J,I]))

f.close()

"""
NFILT : number of filter profile points
FILTER(I),VFILT(I) : filter profile point
NCALC : number of calculations
ITYPE(I),NINTP(I),NREALP(I),NCHP(I) : type and # of parameters for calc
ICALD(J,I)
NREALP(I)
RCALD(J,I)
NCHP(I)
CCALD(J,I)
"""