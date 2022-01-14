#Importing libraries
import numpy as np
from NemesisPy import *
from scipy.interpolate import interp1d

filename = 'warren1986_co2ice.dat'
nwave = file_lines(filename)

f = open(filename,'r')

wave = np.zeros(nwave)
mr = np.zeros(nwave)
mi = np.zeros(nwave)

swave = ''
smr = ''
smi = ''
for i in range(nwave):
    s = f.readline().split()
    s0 = s[0]
    s0 = s0.replace(u'\u2212', '-').replace('$', '')
    s1 = s[1]
    s1 = s1.replace(u'\u2212', '-').replace('$', '')
    wave[i] = float(s0) 
    mr[i] = float(s1)
    if len(s)==3:
        s2 = s[2]
        s2 = s2.replace(u'\u2212', '-').replace('$', '')
        mi[i] = float(s2)
    else:
        mi[i] = -1.

f.close()

swave = ''
for i in range(nwave):
    swave = swave+str(mi[i])+','

print(swave)
