from NemesisPy.Profile import *
from NemesisPy.Models import *
from NemesisPy.Data import *
import numpy as np
import matplotlib.pyplot as plt

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

def write_pat_SO(runname,iscat,nconv,vconv,fwhm,layht,nlayer,baseH,layint,flagh2p):
    
    """
        FUNCTION NAME : write_pat_SO()
        
        DESCRIPTION : Subroutine to write the .pat file for a solar occultation measurement
                      in which thermal emission from the atmosphere is not included
                      (just transmission)
        
        INPUTS :
        
            runname :: Name of the Nemesis run
            iscat :: Flag indicating which kind of scattering calculation must be included
            nconv :: Number of convolution wavelengths
            vconv(nconv) :: Convolution wavelengths
            fwhm :: Full-width at half-maximum of the instrument 
            layht :: Altitude of the base of the lowest layer (km) relative to a reference level 
                     defined in the ‘.prf’ file
            nlayer :: Number of atmospheric layers to be included in the atmosphere
            baseH(nlayer) :: Base altitude of each of the layers in which the atmosphere needs to be spit.
                     Typically these coincide with the altitude of the acquisitions made in the 
                     solar occultation measurement
            layint :: Type of layer integration
            flagh2p :: Flaf indicating if para-H2 is variable (True or False)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            Nemesis .pat file
        
        CALLING SEQUENCE:
        
            write_pat_SO(runname,iscat,nconv,vconv,fwhm,layht,nlayer,baseH,layint,flagh2p)
        
        MODIFICATION HISTORY : Juan Alday (29/07/2021)
        
    """

    f = open(runname+'.pat','w')

    #Writing .pat file
    ########################

    f.write(' \n')

    #Writing wavelength interval and resolution
    f.write('interval \n')

    if nconv>1:
        delv = (vconv.max() - vconv.min())/(nconv-1)
    else:
        delv = fwhm

    f.write('\t %7.5f \t %7.5f \t %7.5e \t %7.5f \n' % (vconv.min(),vconv.max(),delv,fwhm))
    f.write('\t 24 \t 0 \t 0 \n')
    f.write(' \n')

    #Writing the name of the file storing the k-tables
    f.write('spec data '+runname+'.kls \n')
    f.write(' \n')

    #Writing the name of the file storing the reference atmosphere
    f.write('model '+runname+'.prf \n')
    f.write(' \n')

    f.write('dust model aerosol.prf \n')
    f.write(' \n')

    #Writing the name of the file storing the aerosol cross sections
    f.write('dust spectra '+runname+'.xsc \n')
    f.write(' \n')

    #Writing the name of the file storing the para-H2 profile
    if flagh2p==True:
        f.write('fparah2 model parah2.prf \n')
        f.write(' \n')

    #Writing the specifications for splitting the atmosphere into layers
    f.write('layer \n')
    f.write('nlay \t %i \n' % (nlayer))
    f.write('layht \t %7.4f \n' % (layht))
    layang = 90.0    #In solar occultations the viewing angle is 90 deg
    f.write('layang \t %7.4f \n' % (layang))
    f.write('layint \t %i \n' % (layint))
    laytyp = 5       #The base altitude of the layers is stored in the height.lay file
    f.write('laytyp \t %i \n' % (laytyp))
    f.write(' \n')

    #Writing the specifications of each of the atmospheric paths to calculate
    for i in range(nlayer):
        print(i)
        f.write('atm \n')
        f.write('limb \t %i \n' % (i))
        f.write('notherm \n')   #We could potentially include the thermal emission
        if iscat!=0:
            f.write('scatter \n')
        else:
            f.write('noscatter \n')
        f.write('nowf \n')
        f.write('nocg \n')
        f.write('noabsorb \n')
        f.write('binbb \n')
        f.write('nobroad \n')
        f.write(' \n')

    f.write('noclrlay \n')
    f.write('nocombine \n')

    f.close()

