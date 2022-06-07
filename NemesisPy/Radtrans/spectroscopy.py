# NAME:
#       spectroscopy.py (nemesislib)
#
# DESCRIPTION:
#
#	This library contains functions to perform spectroscopic calculations of gaseous species
#
#
# CATEGORY:
#
#	NEMESIS
# 
# MODIFICATION HISTORY: Juan Alday 15/03/2021

import numpy as np
from struct import *
import pylab
import sys,os,errno,shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as font_manager
import matplotlib as mpl
from NemesisPy import *
from numba import jit
from copy import copy

###############################################################################################
def calc_klbl(filename,wavemin,wavemax,npoints,press,temp,MakePlot=False):
    
    """
        
        FUNCTION NAME : calc_klbl()
        
        DESCRIPTION : Calculate the absorption coefficient of a gas at a given pressure and temperature
                      looking at pre-tabulated line-by-line tables
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .lta extension)
            wavemin :: Minimum Wavenumber (cm-1)
            wavemax :: Maximum Wavenumber (cm-1)
            npoints :: Number of p-T levels at which the absorption coefficient must be computed
            press(npoints) :: Pressure (atm)
            temp(npoints) :: Temperature (K)
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            wavelta :: Calculation wavenumbers (cm-1)
            k(nwave,npoints) :: Absorption coefficient (cm2)
        
        CALLING SEQUENCE:
        
            wavelta,k = calc_klbl(filename,wavemin,wavemax,npoints,press,temp)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """
    
    from NemesisPy.Utils import find_nearest

    #Reading the lbl-tables
    #wavemin = wave.min()
    #wavemax = wave.max()
    #wavemin = wave[0] - (wave[1]-wave[0])
    #wavemax = wave[nwave-1] + (wave[nwave-1]-wave[nwave-2])
    
    npress,ntemp,gasID,isoID,presslevels,templevels,nwavelta,wavelta,k = read_lbltable(filename,wavemin,wavemax)
    
    #Interpolating to the correct pressure and temperature
    ########################################################
    
    kgood = np.zeros((nwavelta,npoints))
    for ipoint in range(npoints):
        press1 = press[ipoint]
        temp1 = temp[ipoint]
        
        #Getting the levels just above and below the desired points
        lpress  = np.log(press1)
        press0,ip = find_nearest(presslevels,press1)

        if presslevels[ip]>=press1:
            iphi = ip
            if ip==0:
                ipl = 0
            else:
                ipl = ip - 1
        elif presslevels[ip]<press1:
            ipl = ip
            if ip==npress-1:
                iphi = npress - 1
            else:
                iphi = ip + 1
        
        temp0,it = find_nearest(templevels,temp1)

        if templevels[it]>=temp1:
            ithi = it
            if it==0:
                itl = 0
            else:
                itl = it - 1
        elif templevels[it]<temp1:
            itl = it
            if it==ntemp-1:
                ithi = ntemp - 1
            else:
                ithi = it + 1

        plo = np.log(presslevels[ipl])
        phi = np.log(presslevels[iphi])
        tlo = templevels[itl]
        thi = templevels[ithi]
        klo1 = np.zeros(nwavelta)
        klo2 = np.zeros(nwavelta)
        khi1 = np.zeros(nwavelta)
        khi2 = np.zeros(nwavelta)
        klo1[:] = k[:,ipl,itl]
        klo2[:] = k[:,ipl,ithi]
        khi1[:] = k[:,iphi,itl]
        khi2[:] = k[:,iphi,ithi]

        #Interpolating to get the absorption coefficient at desired p-T
        v = (lpress-plo)/(phi-plo)
        u = (temp1-tlo)/(thi-tlo)

        if(thi==tlo):
            u = 0
        if(phi==plo):
            v = 0
        
        igood = np.where((klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0))
        igood = igood[0]
        kgood[igood,ipoint] = (1.0-v)*(1.0-u)*np.log(klo1[igood]) + v*(1.0-u)*np.log(khi1[igood]) + v*u*np.log(khi2[igood]) + (1.0-v)*u*np.log(klo2[igood])
        kgood[igood,ipoint] = np.exp(kgood[igood,ipoint])

    if MakePlot==True:
        fig, ax = plt.subplots(1,1,figsize=(10,6))
    
        ax.semilogy(wavelta,klo1*1.0e-20,label='p = '+str(np.exp(plo))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wavelta,klo2*1.0e-20,label='p = '+str(np.exp(plo))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wavelta,khi1*1.0e-20,label='p = '+str(np.exp(phi))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wavelta,khi2*1.0e-20,label='p = '+str(np.exp(phi))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wavelta,kgood[:,ipoint]*1.0e-20,label='p = '+str(press1)+' atm - T = '+str(temp1)+' K',color='black')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()
    
    return wavelta,kgood


###############################################################################################

def calc_k(filename,wavemin,wavemax,npoints,press,temp,MakePlot=False):
    
    """
        
        FUNCTION NAME : calc_k()
        
        DESCRIPTION : Calculate the k coefficients of a gas at a given pressure and temperature
                      looking at pre-tabulated correlated-k tables
        
        INPUTS :
        
            filename :: Name of the file (supposed to have a .lta extension)
            wavemin :: Wavenumbers to calculate the spectrum (cm-1)
            wavemax :: Maximum Wavenumber to calculate the spectrum (cm-1)
            npoints :: Number of p-T levels at which the absorption coefficient must be computed
            press(npoints) :: Pressure (atm)
            temp(npoints) :: Temperature (K)
        
        OPTIONAL INPUTS:
        
            MakePlot :: If True, a summary plot is generated
        
        OUTPUTS :
        
            wavek :: Calculation wavenumbers (cm-1)
            ng :: Number of g-ordinates
            g_ord :: G-ordinates
            del_g :: Interval between contiguous g-ordinates
            k(nwave,ng,npoints) :: K coefficients 
        
        CALLING SEQUENCE:
        
            wavekta,ng,g_ord,del_g,k = calc_k(filename,wavemin,wavemax,npoints,press,temp)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """

    from NemesisPy import find_nearest

    gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g = read_ktable(filename,wavemin,wavemax)

    #Interpolating to the correct pressure and temperature
    ########################################################
    
    k_good = np.zeros([nwave,ng,npoints])
    for ipoint in range(npoints):
        press1 = press[ipoint]
        temp1 = temp[ipoint]

        #Getting the levels just above and below the desired points
        lpress  = np.log(press1)
        press0,ip = find_nearest(presslevels,press1)

        if presslevels[ip]>=press1:
            iphi = ip
            if ip==0:
                ipl = 0
            else:
                ipl = ip - 1
        elif presslevels[ip]<press1:
            ipl = ip
            if ip==npress-1:
                iphi = npress - 1
            else:
                iphi = ip + 1
        
        temp0,it = find_nearest(templevels,temp1)

        if templevels[it]>=temp1:
            ithi = it
            if it==0:
                itl = 0
            else:
                itl = it - 1
        elif templevels[it]<temp1:
            itl = it
            if it==ntemp-1:
                ithi = ntemp - 1
            else:
                ithi = it + 1
    
        plo = np.log(presslevels[ipl])
        phi = np.log(presslevels[iphi])
        tlo = templevels[itl]
        thi = templevels[ithi]
        klo1 = np.zeros([nwave,ng])
        klo2 = np.zeros([nwave,ng])
        khi1 = np.zeros([nwave,ng])
        khi2 = np.zeros([nwave,ng])
        klo1[:] = k_g[:,:,ipl,itl]
        klo2[:] = k_g[:,:,ipl,ithi]
        khi2[:] = k_g[:,:,iphi,ithi]
        khi1[:] = k_g[:,:,iphi,itl]

        #Interpolating to get the k-coefficients at desired p-T
        if ipl==iphi:
            v = 0.5
        else:
            v = (lpress-plo)/(phi-plo)
        
        if itl==ithi:
            u = 0.5
        else:
            u = (temp1-tlo)/(thi-tlo)

        k_good[:,:,ipoint] = np.exp((1.0-v)*(1.0-u)*np.log(klo1[:,:]) + v*(1.0-u)*np.log(khi1[:,:]) + v*u*np.log(khi2[:,:]) + (1.0-v)*u*np.log(klo2[:,:]))
    
    if MakePlot==True:
        fig, ax = plt.subplots(1,1,figsize=(10,6))
    
        k_abs = np.matmul(k_good[:,:,npoints-1], del_g)
        k_abslo1 = np.matmul(klo1[:,:], del_g)
        k_abslo2 = np.matmul(klo2[:,:], del_g)
        k_abshi1 = np.matmul(khi1[:,:], del_g)
        k_abshi2 = np.matmul(khi2[:,:], del_g)
        ax.semilogy(wave,k_abslo1,label='p = '+str(np.exp(plo))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wave,k_abslo2,label='p = '+str(np.exp(plo))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wave,k_abshi1,label='p = '+str(np.exp(phi))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wave,k_abshi2,label='p = '+str(np.exp(phi))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wave,k_abs,label='p = '+str(press1)+' atm - T = '+str(temp1)+' K',color='black')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()

    return wave,ng,g_ord,del_g,k_good


###############################################################################################
def k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f):
    
    """
        
        FUNCTION NAME : k_overlap()
        
        DESCRIPTION : This subroutine combines the absorption coefficient distributions of
                      several overlapping gases. The overlap is implicitly assumed to be random
                      and the k-distributions are assumed to have NG-1 mean values and NG-1
                      weights. Correspondingly there are NG ordinates in total.
        
        INPUTS :
        
            nwave :: Number of wavelengths
            ng :: Number of g-ordinates
            del_g :: Intervals of g-ordinates
            ngas :: Number of gases to combine
            npoints :: Number of p-T points over to run the overlapping routine
            k_gas(nwave,ng,ngas,npoints) :: K-distributions of the different gases
            f(ngas,npoints) :: fraction of the different gases at each of the p-T points

        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            k(nwave,ng,npoints) :: Combined k-distribution
        
        CALLING SEQUENCE:
        
            k = k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """

    k = np.zeros((nwave,ng,npoints))

    if ngas<=1:  #There are not enough gases to combine
        k[:,:,:] = k_gas[:,:,:,0]
    else:

        for ip in range(npoints): #running for each p-T case

            for igas in range(ngas-1):

                #getting first and second gases to combine
                if igas==0:
                    k_gas1 = np.zeros((nwave,ng))
                    k_gas2 = np.zeros((nwave,ng))
                    k_gas1[:,:] = k_gas[:,:,ip,igas]
                    k_gas2[:,:] = k_gas[:,:,ip,igas+1]
                    f1 = f[igas,ip]
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))
                else:
                    #k_gas1 = np.zeros((nwave,ng))
                    #k_gas2 = np.zeros((nwave,ng))
                    k_gas1[:,:] = k_combined[:,:]
                    k_gas2[:,:] = k_gas[:,:,ip,igas+1]
                    f1 = f_combined
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))

                for iwave in range(nwave):

                    k_g_combined, f_combined = k_overlap_two_gas(k_gas1[iwave,:], k_gas2[iwave,:], f1, f2, del_g)
                    k_combined[iwave,:] = k_g_combined[:]

            k[:,:,ip] = k_combined[:,:]

    return k

###############################################################################################
@jit(nopython=True)
def k_overlapg(nwave,ng,del_g,ngas,npoints,k_gas,dkgasdT,f):
    
    """
        
        FUNCTION NAME : k_overlapg()
        
        DESCRIPTION : This subroutine combines the absorption coefficient distributions of
                      several overlapping gases. The overlap is implicitly assumed to be random
                      and the k-distributions are assumed to have NG-1 mean values and NG-1
                      weights. Correspondingly there are NG ordinates in total.
        
        INPUTS :
        
            nwave :: Number of wavelengths
            ng :: Number of g-ordinates
            del_g :: Intervals of g-ordinates
            ngas :: Number of gases to combine
            npoints :: Number of p-T points over to run the overlapping routine
            k_gas(nwave,ng,ngas,npoints) :: K-distributions of the different gases
            dkgasdT(nwave,ng,ngas,npoints) :: Rate of change of K-distributions of the different gases with temperature
            f(ngas,npoints) :: Absorber amounts for each of the gases in each of the layers (cm-2)

        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            k(nwave,ng,npoints) :: Combined k-distribution
            dk(nwave,ng,npoints,ngas+1) :: Combined rate of change of k-distribution with the gas VMRs (0 to NGAS-1) and with temperature (NGAS-1)
        
        CALLING SEQUENCE:
        
            k,dk = k_overlapg(nwave,ng,del_g,ngas,npoints,k_gas,dkgasdT,f)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """

    k = np.zeros((nwave,ng,npoints))
    dk = np.zeros((nwave,ng,npoints,ngas+1))

    for ip in range(npoints): #running for each p-T case

        k_combined = np.zeros((nwave,ng))
        dk_combined = np.zeros((nwave,ng,ngas+1))

        for iwave in range(nwave):

            k_g = np.zeros((ng,ngas))
            dkgdT = np.zeros((ng,ngas))
            q = np.zeros(ngas)
            k_g[:,:] = k_gas[iwave,:,ip,:]
            dkgdT[:,:] = dkgasdT[iwave,:,ip,:]
            q[:] = f[:,ip]

            if ng==1:
                k_g_combined, dk_g_combined = k_overlapg_gas(k_g, dkgdT, q, del_g)
                k_combined[iwave,:] = k_g_combined[:]
                dk_combined[iwave,:,:] = dk_g_combined[:,:]
            else:
                k_g_combined, dk_g_combined = k_overlapg_gas(k_g, dkgdT, q, del_g)
                k_combined[iwave,:] = k_g_combined[:]
                dk_combined[iwave,:,:] = dk_g_combined[:,:]

        k[:,:,ip] = k_combined[:,:]
        dk[:,:,ip,:] = dk_combined[:,:,:]

    return k,dk

###############################################################################################
@jit(nopython=True)
def k_overlap_two_gas(k_g1, k_g2, q1, q2, del_g):
    
    """
        
        FUNCTION NAME : mix_two_gas_k()
        
        DESCRIPTION : This subroutine combines the absorption coefficient distributions of
                      two overlapping gases. The overlap is implicitly assumed to be random
                      and the k-distributions are assumed to have NG-1 mean values and NG-1
                      weights. Correspondingly there are NG ordinates in total.
        
        INPUTS :
        
            k_g1(ng) :: k-coefficients for gas 1 at a particular wave bin and temperature/pressure.
            k_g2(ng) :: k-coefficients for gas 2 at a particular wave bin and temperature/pressure.
            q1 :: Volume mixing ratio of gas 1
            q2 :: Volume mixing ratio of gas 2
            del_g(ng) ::Gauss quadrature weights for the g-ordinates, assumed same for both gases.

        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            k_g_combine(ng) :: Combined k-distribution of both gases
            q_combined :: Combined Volume mixing ratio of both gases
        
        CALLING SEQUENCE:
        
            k_g_combined,VMR_combined = k_overlap_two_gas(k_g1, k_g2, q1, q2, del_g)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """

    ng = len(del_g)  #Number of g-ordinates
    k_g = np.zeros(ng)
    q_combined = q1 + q2

    if((k_g1[ng-1]<=0.0) and (k_g2[ng-1]<=0.0)):
        pass
    elif( (q1<=0.0) and (q2<=0.0) ):
        pass
    elif((k_g1[ng-1]==0.0) or (q1==0.0)):
        k_g[:] = k_g2[:] * q2/(q1+q2)
    elif((k_g2[ng-1]==0.0) or (q2==0.0)):
        k_g[:] = k_g1[:] * q1/(q1+q2)
    else:

        nloop = ng * ng
        weight = np.zeros(nloop)
        contri = np.zeros(nloop)
        ix = 0
        for i in range(ng):
            for j in range(ng):
                weight[ix] = del_g[i] * del_g[j]
                contri[ix] = (k_g1[i]*q1 + k_g2[j]*q2)/(q1+q2)
                ix = ix + 1

        #getting the cumulative g ordinate
        g_ord = np.zeros(ng+1)
        g_ord[0] = 0.0
        for ig in range(ng):
            g_ord[ig+1] = g_ord[ig] + del_g[ig]

        if g_ord[ng]<1.0:
            g_ord[ng] = 1.0

        #sorting contri array
        isort = np.argsort(contri)
        contrib1 = contri[isort]
        weight1 = weight[isort]

        #creating combined g-ordinate array
        gdist = np.zeros(nloop)
        gdist[0] = weight1[0]
        for i in range(nloop-1):
            ix = i + 1
            gdist[ix] = weight1[ix] + gdist[i]

        ig = 0
        sum1 = 0.0
        for i in range(nloop):
            
            if( (gdist[i]<g_ord[ig+1]) & (ig<=ng-1) ):
                k_g[ig] = k_g[ig] + contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
            else:
                frac = (g_ord[ig+1]-gdist[i-1])/(gdist[i]-gdist[i-1])
                k_g[ig] = k_g[ig] + frac * contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
                k_g[ig] = k_g[ig] / sum1
                ig = ig + 1
                if(ig<=ng-1):
                    sum1 = (1.-frac)*weight1[i]
                    k_g[ig] = k_g[ig] + (1.-frac) * contrib1[i] * weight1[i]

        if ig==ng-1:
            k_g[ig] = k_g[ig] / sum1

    return k_g, q_combined
    
###############################################################################################
@jit(nopython=True)
def k_overlapg_gas(k_g, dkgdT, q, del_g):
    
    """
        
        FUNCTION NAME : k_overlapg_gas()
        
        DESCRIPTION : This subroutine combines the absorption coefficient distributions of
                      two overlapping gases. The overlap is implicitly assumed to be random
                      and the k-distributions are assumed to have NG-1 mean values and NG-1
                      weights. Correspondingly there are NG ordinates in total.
        
        INPUTS :
        
            k_g(ng,ngas) :: k-coefficients for each gas at a particular wave bin and temperature/pressure.
            dkgdT(ng,ngas) :: Rate of change of k-coefficients with temperature for each gas at a particular wave bin and temperature/pressure.
            q(ngas) :: Absorber amount of each gas (cm-2)
            del_g(ng) ::Gauss quadrature weights for the g-ordinates, assumed same for both gases.
        
        OPTIONAL INPUTS: None
        
        OUTPUTS :
        
            k_g_combined(ng) :: Combined k-distribution of both gases
            dk_g_combined(ng,3) :: Combined rate of change of the k-distribution opacity with respect to the absorber amounts (cm-2) of each of the gases and the temperature
            q_combined :: Combined absorber amount (cm-2)
        
        CALLING SEQUENCE:
        
            k_g_combined,dk_g_combined = k_overlapg_gas(k_g, dkgdT, q, del_g)
        
        MODIFICATION HISTORY : Juan Alday (25/09/2019)
        
    """

    ng = len(del_g)  #Number of g-ordinates
    ngas = len(k_g[0,:]) #Number of active gases in atmosphere

    k_g_combined = np.zeros(ng)  #Combined opacity
    dk_g_combined = np.zeros((ng,ngas+1)) #dTAU/dAMOUNT or dTAU/dT

    if ngas==1:

        q1 = q[0]
        k_g1 = k_g[0,0]
        dkg1dT = dkgdT[0,0]

        k_g_combined[0] = k_g1 * q1
        dk_g_combined[0,0] = k_g1
        dk_g_combined[0,1] = dkg1dT * q1 #dk/dT

    else:

        for igas in range(ngas-1):

            if igas==0:

                q1 = q[igas]
                q2 = q[igas+1]
                k_g1 = np.zeros(ng)
                k_g2 = np.zeros(ng)
                dkg1dT = np.zeros(ng)
                dkg2dT = np.zeros(ng)
                k_g1[:] = k_g[:,igas]
                k_g2[:] = k_g[:,igas+1]
                dkg1dT[:] = dkgdT[:,igas]
                dkg2dT[:] = dkgdT[:,igas+1]

                # skip if first k-distribution = 0.0
                if((k_g1[ng-1]==0.0)):
                    k_g_combined[:] = k_g2[:] * q2
                    dk_g_combined[:,igas] = 0.0  #dk/dq1
                    dk_g_combined[:,igas+1] = k_g2[:] #dk/dq2
                    dk_g_combined[:,igas+2] = dkg2dT[:] * q2 #dk/dT
                    icomp = 0
                elif((k_g2[ng-1]==0.0)):
                    k_g_combined[:] = k_g1[:] * q1
                    dk_g_combined[:,igas] = k_g1[:]  #dk/dq1
                    dk_g_combined[:,igas+1] = 0.0  #dk/dq2
                    dk_g_combined[:,igas+2] = dkg1dT[:] * q1 #dk/dT
                    icomp = 0
                else:
                    icomp = 1
                    nloop = ng * ng
                    weight = np.zeros(nloop)
                    contri = np.zeros(nloop)
                    grad = np.zeros((nloop,ngas+1))
                    ix = 0
                    for i in range(ng):
                        for j in range(ng):
                            weight[ix] = del_g[i] * del_g[j]
                            contri[ix] = (k_g1[i]*q1 + k_g2[j]*q2)
                            grad[ix,0] = k_g1[i]
                            grad[ix,1] = k_g2[j]
                            grad[ix,2] = dkg1dT[i] * q1 + dkg2dT[j] * q2
                            ix = ix + 1

                    k_g_combined,dk_g_combined = rankg(weight,contri,grad,del_g)

            else:

                q2 = q[igas+1]
                k_g1 = np.zeros(ng)
                k_g2 = np.zeros(ng)
                k_g1[:] = k_g_combined[:]
                k_g2[:] = k_g[:,igas+1]
                dkg1dT = np.zeros(ng)
                dkg2dT = np.zeros(ng)
                dkg1dT[:] = dk_g_combined[:,igas+1]  #dk/dT of previous sum of gases
                dkg2dT[:] = dkgdT[:,igas+1] #dK/dT of new gas

                if((k_g1[ng-1]==0.0)):
                    k_g_combined[:] = k_g2[:] * q2
                    for jgas in range(igas):
                        dk_g_combined[:,jgas] = 0.0 
                    dk_g_combined[:,igas+1] = k_g2[:] #dk/dq2
                    dk_g_combined[:,igas+2] = dkg2dT[:] * q2 #dk/dT
                    icomp = 0
                elif((k_g2[ng-1]==0.0)):
                    k_g_combined[:] = k_g1[:]
                    dk_g_combined[:,igas+1] = 0.0  #dk/dq2
                    dk_g_combined[:,igas+2] = dkg1dT[:] #dk/dT
                    icomp = 0
                else:
                    icomp = 1
                    nloop = ng * ng
                    weight = np.zeros(nloop)
                    contri = np.zeros(nloop)
                    grad = np.zeros((nloop,ngas+1))
                    ix = 0
                    for i in range(ng):
                        for j in range(ng):
                            weight[ix] = del_g[i] * del_g[j]
                            contri[ix] = (k_g1[i] + k_g2[j]*q2)
                            for jgas in range(igas):
                                grad[ix,jgas] = dk_g_combined[i,jgas]
                            grad[ix,igas+1] = k_g2[j]
                            grad[ix,igas+2] = dkg1dT[i] + dkg2dT[j] * q2
                            ix = ix + 1

                    k_g_combined,dk_g_combined = rankg(weight,contri,grad,del_g)

    return k_g_combined,dk_g_combined

@jit(nopython=True)
def rankg(weight, cont, grad, del_g):
    """
    Combine the randomly overlapped k distributions of two gases into a single
    k distribution.

    Parameters
    ----------
    weight(NG*NG) : ndarray
        Weights of points in the random k-dist
    cont(NG*NG) : ndarray
        Random k-coeffs in the k-dist.
    grad(NG*NG,NGAS+1) : ndarray
        Gradients of random k-coeffs in the k-dist.
    del_g(NG) : ndarray
        Required weights of final k-dist.

    Returns
    -------
    k_g(NG) : ndarray
        Combined k-dist.
    """
    ng = len(del_g)
    ngas = len(grad[0,:]) - 1
    nloop = ng*ng

    # sum delta gs to get cumulative g ordinate
    g_ord = np.zeros(ng+1)
    g_ord[1:] = np.cumsum(del_g)
    g_ord[ng] = 1

    # Sort random k-coeffs into ascending order. Integer array ico records
    # which swaps have been made so that we can also re-order the weights.
    ico = np.argsort(cont)
    cont = cont[ico]
    grad[:,:] = grad[ico,:]
    weight = weight[ico] # sort weights accordingly
    gdist = np.cumsum(weight)
    k_g = np.zeros(ng)
    dkdq = np.zeros((ng,ngas+1))

    ig = 0
    sum1 = 0.0
    cont_weight = cont * weight
    grad_weight = np.zeros((nloop,ngas+1))
    for igas in range(ngas+1):
        grad_weight[:,igas] = grad[:,igas] * weight[:]


    for iloop in range(nloop):
        if gdist[iloop] < g_ord[ig+1]:
            k_g[ig] = k_g[ig] + cont_weight[iloop]
            for igas in range(ngas+1):
                dkdq[ig,igas] = dkdq[ig,igas] + grad_weight[iloop,igas]
            sum1 = sum1 + weight[iloop]
        else:
            frac = (g_ord[ig+1] - gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
            k_g[ig] = k_g[ig] + np.float32(frac)*cont_weight[iloop]
            for igas in range(ngas+1):
                dkdq[ig,igas] = dkdq[ig,igas] + np.float32(frac)*grad_weight[iloop,igas]

            sum1 = sum1 + frac * weight[iloop]
            k_g[ig] = k_g[ig]/np.float32(sum1)
            for igas in range(ngas+1):
                dkdq[ig,igas] = dkdq[ig,igas]/np.float32(sum1)

            ig = ig +1
            sum1 = (1.0-frac)*weight[iloop]
            k_g[ig] = np.float32(1.0-frac)*cont_weight[iloop]
            for igas in range(ngas+1):
                dkdq[ig] = np.float32(1.0-frac)*grad_weight[iloop,igas]

    if ig == ng-1:
        k_g[ig] = k_g[ig]/np.float32(sum1)
        for igas in range(ngas+1):
            dkdq[ig,igas] = dkdq[ig,igas]/np.float32(sum1)

    return k_g,dkdq

###############################################################################################

def lblconv(fwhm,ishape,nwave,vwave,y,nconv,vconv,runname=''):

    """
        FUNCTION NAME : lblconv()
        
        DESCRIPTION : Convolve the modelled spectrum with a given instrument line shape
        
        INPUTS :
        
            fwhm :: FWHM of the instrument line shape
                if FWHM<0.0 then the function defining the ILS for each convolution wavelength
                is assumed to be stored in the .fil file. In that case, the keyword runname
                needs to be included

            ishape :: Shape of the instrument function (if FWHM>0.0)
                ishape = 0 :: Square instrument lineshape
                ishape = 1 :: Triangular instrument shape
                ishape = 2 :: Gaussian instrument shape

            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave) :: Modelled spectrum
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers

        OPTIONAL INPUTS:

            runname :: Name of the Nemesis run
        
        OUTPUTS :
        
            yout(nconv) :: Convolved spectrum

        CALLING SEQUENCE:
        
            yout = lblconv(fwhm,ishape,nwave,vwave,y,nconv,vconv)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    from NemesisPy import read_fil

    yout = np.zeros([nconv])
    ynor = np.zeros([nconv])

    if fwhm>0.0:
        #Set total width of Hamming/Hanning function window in terms of
        #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
        nfw = 3.

        for j in range(nconv):
            yfwhm = fwhm
            vcen = vconv[j]
            if ishape==0:
                v1 = vcen-0.5*yfwhm
                v2 = v1 + yfwhm
            elif ishape==1:
                v1 = vcen-yfwhm
                v2 = vcen+yfwhm
            elif ishape==2:
                sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
                v1 = vcen - 3.*sig
                v2 = vcen + 3.*sig
            else:
                v1 = vcen - nfw*yfwhm
                v2 = vcen + nfw*yfwhm


            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            for i in range(np1):
                f1=0.0
                if ishape==0:
                    #Square instrument lineshape
                    f1=1.0
                elif ishape==1:
                    #Triangular instrument shape
                    f1=1.0 - abs(vwave[inwave[i]] - vcen)/yfwhm
                elif ishape==2:
                    #Gaussian instrument shape
                    f1 = np.exp(-((vwave[inwave[i]]-vcen)/sig)**2.0)
                else:
                    sys.exit('lblconv :: ishape not included yet in function')

                if f1>0.0:
                    yout[j] = yout[j] + f1*y[inwave[i]]
                    ynor[j] = ynor[j] + f1

            yout[j] = yout[j]/ynor[j]

    if fwhm<0.0:
        #Line shape for each convolution number in each case is read from .fil file
        nconv1,vconv1,nfil,vfil,afil = read_fil(runname)

        if nconv1 != nconv:
            sys.exit('lblconv :: Convolution wavenumbers must be the same in .spx and .fil files')

        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            xp = np.zeros([nfil[j]])
            yp = np.zeros([nfil[j]])
            xp[:] = vfil[0:nfil[j],j]
            yp[:] = afil[0:nfil[j],j]
            for i in range(np1):
                #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                f1 = np.interp(vwave[inwave[i]],xp,yp)
                if f1>0.0:
                    yout[j] = yout[j] + f1*y[inwave[i]]
                    ynor[j] = ynor[j] + f1

            yout[j] = yout[j]/ynor[j]

    return yout

###############################################################################################

def wavesetb(runname,nconv,vconv,fwhm):

    """
    FUNCTION NAME : wavesetb()

    DESCRIPTION : Subroutine to calculate which 'calculation' wavelengths are needed to cover the required 'convolution wavelengths'.

    INPUTS : 

        runname :: Name of the Nemesis run
        nconv :: Number of convolution wavelengths
        vconv(nconv) :: Convolution wavelengths
        fwhm :: FWHM of convolved spectrum

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    nwave :: Number of calculation wavenumbers
	    vwave(mwave) :: Calculation wavenumbers
 
    CALLING SEQUENCE:

	    nwave,vwave = wavesetb_nemesis(runname,nconv,vconv,fwhm)
 
    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    #Reading the .kls file to get the initial and end wavenumbers in the .kta files
    ngasact,strlta = read_kls(runname)
    nwavelta = np.zeros([ngasact],dtype='int')
    
    for i in range(ngasact):
        nwave,vmin,delv,fwhmk,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels = read_ktahead(strlta[i])
        nwavelta[i] = nwave

    if len(np.unique(nwavelta)) != 1:
        sys.exit('error :: Number of wavenumbers in all .kta files must be the same')

    vkstart = vmin
    vkstep = delv
    vkend = vkstart + delv*(nwave-1)  

    #Determining the calculation numbers
    savemax = 1000000
    save = np.zeros([savemax])
    ico = 0

    if (vkstep < 0.0 or fwhm == 0.0):
        ico = nconv
        for i in range(nconv):
            save[i] = vconv[i]

    if fwhm < 0.0:
        nconv1,vconv1,nfil,vfil,afil = read_fil(runname)
        if nconv != nconv1:
            sys.exit('error :: onvolution wavenumbers must be the same in .spx and .fil files')

        for i in range(nconv1):
            vcentral = vconv1[i]
            for j in range(nconv):
                dv = abs(vcentral-vconv[j])
                if dv < 0.00001:
                    j1 = int((vfil[0,i]-vkstart)/vkstep - 1)
                    j2 = int((vfil[nfil[i]-1,i]-vkstart)/vkstep + 1)
                    v1 = vkstart + (j1-1)*vkstep
                    v2 = vkstart + (j2-1)*vkstep
                    if (v1 < vkstart or v2 > vkend):
                        print('warning from wavesetc')
                        print('Channel wavelengths not covered by lbl-tables')
                        print('v1,v2,vkstart,vkend',v1,v2,vkstart,vkend)
                    for k in range(j2-j1):
                        jj = k + j1
                        vj = vkstart + jj*vkstep
                        save[ico]=vj
                        ico = ico + 1


    elif fwhm > 0.0:

        for i in range(nconv):
            j1 = int( (vconv[i]-0.5*fwhm-vkstart)/vkstep )
            j2 = 2 + int( (vconv[i]+0.5*fwhm-vkstart)/vkstep )
            v1 = vkstart + (j1-1)*vkstep
            v2 = vkstart + (j2-1)*vkstep

            if (v1 < vkstart or v2 > vkend):
                print('warning from wavesetc')
                print('Channel wavelengths not covered by lbl-tables')
                print('v1,v2,vkstart,vkend',v1,v2,vkstart,vkend)

            for k in range(j2-j1):
                jj = k + j1
                vj = v1 + (jj-j1)*vkstep
                save[ico]=vj
                ico = ico + 1

    nco = ico
    #sort calculation wavenumbers into order
    save1 = np.zeros([nco])
    save1 = np.sort(save[0:nco])
  
    #creating calculation wavnumber array
    nwave = nco
    vwave = np.zeros([nwave])
    vwave[:] = save1[:]

    #Now weed out repeated wavenumbers
    vwave[1]=save[1]
    xdiff = 0.9*vkstep  
    ico = 0
    for i in range(nco-1):
        test = abs(save1[i+1]-vwave[ico])
        if test >= xdiff:
            ico = ico + 1
            vwave[ico] = save1[i+1]
            nwave = ico

    print('wavesetb_nemesis :: nwave = '+str(nwave))

    return nwave,vwave



###############################################################################################

def wavesetc_v2(runname,nconv,vconv,fwhm):


    """
    FUNCTION NAME : wavesetc_nemesis()

    DESCRIPTION : Subroutine to calculate which 'calculation' wavelengths are needed to cover the required 'convolution wavelengths'.

    INPUTS : 

        runname :: Name of the Nemesis run
        nconv :: NUmber of convolution wavelengths
        vconv(nconv) :: Convolution wavelengths
        fwhm :: Full width at half maximum of instrument line shape

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    nwave :: Number of calculation wavenumbers
	    vwave(nave) :: Calculation wavenumbers
 
    CALLING SEQUENCE:

	    nwave,vwave = wavesetc(runname,nconv,vconv,fwhm)
 
    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    #from NemesisPy.Files import read_lls,read_sha

    #Reading the .lls file to get the initial and end wavenumbers in the .lta files
    ngasact,strlta = read_lls(runname)
    nwavelta = np.zeros([ngasact],dtype='int')
    
    for i in range(ngasact):
        nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels = read_ltahead(strlta[i])
        nwavelta[i] = nwave

    if len(np.unique(nwavelta)) != 1:
        sys.exit('error :: Number of wavenumbers in all .lta files must be the same')

    vkstart = vmin
    vkstep = delv
    vkend = vkstart + delv*(nwave-1)    

    #Determining the calculation numbers
    savemax = 10000000
    save = np.zeros([savemax])
    ico = 0
    if fwhm < 0.0:
        nconv1,vconv1,nfil,vfil,afil = read_fil(runname)
        if nconv != nconv1:
            sys.exit('error :: convolution wavenumbers must be the same in .spx and .fil files')

        for i in range(nconv1):
            vcentral = vconv1[i]
            for j in range(nconv):
                dv = abs(vcentral-vconv[j])
                if dv < 0.00001:
                    j1 = int((vfil[0,i]-vkstart)/vkstep - 1)
                    j2 = int((vfil[nfil[i]-1,i]-vkstart)/vkstep + 1)
                    v1 = vkstart + (j1-1)*vkstep
                    v2 = vkstart + (j2-1)*vkstep
                    if (v1 < vkstart or v2 > vkend):
                        print('warning from wavesetc')
                        print('Channel wavelengths not covered by lbl-tables')
                        print('v1,v2,vkstart,vkend',v1,v2,vkstart,vkend)
                    for k in range(j2-j1):
                        jj = k + j1
                        vj = vkstart + jj*vkstep
                        save[ico]=vj
                        ico = ico + 1


    elif fwhm > 0.0:
        ishape = read_sha(runname)
        if ishape == 0:
            dv = 0.5*fwhm
        elif ishape == 1:
            dv = fwhm
        elif ishape == 2:
            dv = 3.* 0.5 * fwhm / np.sqrt(np.log(2.0))
        else:
            dv = 3.*fwhm

        for i in range(nconv):
            j1 = int( (vconv[i]-dv-vkstart)/vkstep )
            j2 = 2 + int( (vconv[i]+dv-vkstart)/vkstep )
            #v1 = vkstart + (j1-1)*vkstep
            #v2 = vkstart + (j2-1)*vkstep
            v1 = vkstart + (j1)*vkstep
            v2 = vkstart + (j2)*vkstep

            if (v1 < vkstart or v2 > vkend):
                print('warning from wavesetc')
                print('Channel wavelengths not covered by lbl-tables')
                print('v1,v2,vkstart,vkend',v1,v2,vkstart,vkend)

            for k in range(j2-j1):
                jj = k + j1
                vj = v1 + (jj-j1)*vkstep
                save[ico]=vj
                ico = ico + 1

 
    nco = ico
    #sort calculation wavenumbers into order
    save1 = np.zeros([nco])
    save1 = np.sort(save[0:nco])
  
    #creating calculation wavnumber array
    nwave = nco
    vwave1 = np.zeros([nwave])
    vwave1[:] = save1[:]

    #Now weed out repeated wavenumbers
    vwave1[0]=save[0]
    xdiff = 0.9*vkstep  
    ico = 0
    for i in range(nco-1):
        test = abs(save1[i+1]-vwave1[ico])
        if test >= xdiff:
            ico = ico + 1
            vwave1[ico] = save1[i+1]
            nwave = ico

    vwave = np.zeros(nwave)
    vwave[0:nwave] = vwave1[0:nwave]
    #vwave = np.round(vwave,3)

    return nwave,vwave


###############################################################################################

def wavesetc(runname,nconv,vconv,fwhm):


    """
    FUNCTION NAME : wavesetc()

    DESCRIPTION : Subroutine to calculate which 'calculation' wavelengths are needed to cover the required 'convolution wavelengths'.

    INPUTS : 

        runname :: Name of the Nemesis run
        nconv :: NUmber of convolution wavelengths
        vconv(nconv) :: Convolution wavelengths
        fwhm :: Full width at half maximum of instrument line shape

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    nwave :: Number of calculation wavenumbers
	    vwave(nave) :: Calculation wavenumbers
 
    CALLING SEQUENCE:

	    nwave,vwave = wavesetc(runname,nconv,vconv,fwhm)
 
    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    from NemesisPy.Files import read_lls,read_sha

    #Reading the .lls file to get the initial and end wavenumbers in the .lta files
    ngasact,strlta = read_lls(runname)
    nwavelta = np.zeros([ngasact],dtype='int')
    
    for i in range(ngasact):
        nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels = read_ltahead(strlta[i])
        nwavelta[i] = nwave

    if len(np.unique(nwavelta)) != 1:
        sys.exit('error :: Number of wavenumbers in all .lta files must be the same')

    vmax = vmin + delv*(nwavelta-1)
    wavelta = np.linspace(vmin,vmax,nwavelta)
    #wavelta = np.round(wavelta,3)


    #Calculating the maximum and minimum wavenumbers
    if fwhm>0.0:
        ishape = read_sha(runname)
        if ishape == 0:
            dv = 0.5*fwhm
        elif ishape == 1:
            dv = fwhm
        elif ishape == 2:
            dv = 3.* 0.5 * fwhm / np.sqrt(np.log(2.0))
        else:
            dv = 3.*fwhm

        wavemin = vconv[0] - dv
        wavemax = vconv[nconv-1] + dv

        if (wavemin<wavelta.min() or wavemax>wavelta.max()):
            sys.exit('error from wavesetc :: Channel wavelengths not covered by lbl-tables')


    elif fwhm<=0.0:
        
        nconv1,vconv1,nfil,vfil,afil = read_fil(runname)
        if nconv != nconv1:
            sys.exit('error :: convolution wavenumbers must be the same in .spx and .fil files')


        for i in range(nconv1):
            vcentral = vconv1[i]
            wavemin = 1.0e6
            wavemax = 0.0
            for j in range(nconv):
                dv = abs(vcentral-vconv[j])
                if dv < 0.0001:
                    vminx = vfil[0,i]
                    vmaxx = vfil[nfil[i]-1,i]
                    if vminx<wavemin:
                        wavemin = vminx
                    if vmaxx>wavemax:
                        wavemax= vmaxx
                else:
                    print('warning from wavesetc :: Convolution wavenumbers in .spx and .fil do not coincide')

        if (wavemin<wavelta.min() or wavemax>wavelta.max()):
            sys.exit('error from wavesetc :: Channel wavelengths not covered by lbl-tables')


    #Selecting the necessary wavenumbers
    iwave = np.where( (wavelta>=wavemin) & (wavelta<=wavemax) )
    iwave = iwave[0]
    vwave = wavelta[iwave]
    nwave = len(vwave)

    return nwave,vwave

###############################################################################################

def planck(ispace,wave,temp,MakePlot=False):


    """
    FUNCTION NAME : planck()

    DESCRIPTION : Function to calculate the blackbody radiation given by the Planck function

    INPUTS : 

        ispace :: Flag indicating the spectral units
                  (0) Wavenumber (cm-1)
                  (1) Wavelength (um)
        wave(nwave) :: Wavelength or wavenumber array
        temp :: Temperature of the blackbody (K)

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    bb(nwave) :: Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
 
    CALLING SEQUENCE:

	    bb = planck(ispace,wave,temp)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    c1 = 1.1911e-12
    c2 = 1.439
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
    else:
        sys.exit('error in planck :: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    if MakePlot==True:
        fig,ax1 = plt.subplots(1,1,figsize=(10,3))
        ax1.plot(wave,bb)
        if ispace==0:
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax1.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
        else:
            ax1.set_xlabel('Wavelength ($\mu$m)')
            ax1.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
        ax1.grid()
        plt.tight_layout()
        plt.show()

    return bb

###############################################################################################

def planckg(ispace,wave,temp,MakePlot=False):


    """
    FUNCTION NAME : planckg()

    DESCRIPTION : Function to calculate the blackbody radiation given by the Planck function
                    as well as its derivative with respect to temperature

    INPUTS : 

        ispace :: Flag indicating the spectral units
                  (0) Wavenumber (cm-1)
                  (1) Wavelength (um)
        wave(nwave) :: Wavelength or wavenumber array
        temp :: Temperature of the blackbody (K)

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    bb(nwave) :: Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
        dBdT(nwave) :: Temperature gradient (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)/K
 
    CALLING SEQUENCE:

	    bb,dBdT = planckg(ispace,wave,temp)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    c1 = 1.1911e-12
    c2 = 1.439
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
        ap = c1 * c2 * (y**4.)/temp**2.
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
        ap = c1 * c2 * (y**6.) / 1.0e4 / temp**2.
    else:
        sys.exit('error in planck :: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    tmpp = c2 * y / temp
    bp = (np.exp(tmp) - 1.)**2.
    tp = np.exp(tmpp) * ap
    dBdT = tp/bp

    if MakePlot==True:
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,5))
        ax1.plot(wave,bb)
        ax2.plot(wave,dBdT)
        if ispace==0:
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax1.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
            ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax2.set_ylabel('dB/dT (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$ K$^{-1}$)')    
        else:
            ax1.set_xlabel('Wavelength ($\mu$m)')
            ax1.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
            ax2.set_xlabel('Wavelength ($\mu$m)')
            ax2.set_ylabel('Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$ K$^{-1}$)')
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        plt.show()

    return bb,dBdT