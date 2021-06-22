# NAME:
#       files.py (nemesislib)
#
# DESCRIPTION:
#
#	This library contains functions to read and write files that are formatted as 
#	required by the NEMESIS radiative transfer code         
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
import nemesislib.utils as utils
import nemesislib.spectroscopy as spec
from NemesisPy.Profile import *
from NemesisPy.Data import *
from NemesisPy.Models.Models import *

###############################################################################################

def calc_gain_matrix(nx,ny,kk,sa,se):


    """
        FUNCTION NAME : calc_gain_matrix()
        
        DESCRIPTION : 

            Calculate gain matrix and averaging kernels. The gain matrix is calculated with
               dd = sx*kk_T*(kk*sx*kk_T + se)^-1    (if nx>=ny)
               dd = ((sx^-1 + kk_T*se^-1*kk)^-1)*kk_T*se^-1  (if ny>nx)

 
        INPUTS :
       
            nx :: Number of elements in state vector 
            ny :: Number of elements in measurement vector
            kk(ny,nx) :: Jacobian matrix
            sa(nx,nx) :: A priori covariance matric
            se(ny,ny) :: Measurement error covariance matrix

        OPTIONAL INPUTS: none
        
        OUTPUTS :

            dd(nx,ny) :: Gain matrix
            aa(nx,nx) :: Averaging kernels
 
        CALLING SEQUENCE:
        
            dd,aa = calc_gain_matrix(nx,ny,kk,sa,se)       
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)
        
    """

    #Calculating the transpose of kk
    kt = np.transpose(kk)

    #Calculating the gain matrix dd
    if (nx >= ny):
        #Multiply sa*kt
        m = np.matmul(sa,kt)

        #Multiply kk*m so that a = kk*sa*kt
        a = np.matmul(kk,m)

        #Add se to a so that b = kk*sa*kt + se
        b = np.add(a,se)

        #Inverting b so that we calculate c = (kk*sa*kt + se)^(-1)
        c = np.linalg.inv(b)

        #Multiplying sa*kt (m above) to c
        dd = np.matmul(m,c)

    else:

        #Calculating the inverse of Sa and Se
        sai = np.linalg.inv(sa)
#        sei = np.linalg.inv(se)
        sei = np.zeros([ny,ny])
        for i in range(ny):
            sei[i,i] = 1./se[i,i]  #As it is just a diagonal matrix

        #Calculate kt*sei
        m = np.matmul(kt,sei)

        #Calculate m*kk so that kt*se^(-1)*kk
        a = np.matmul(m,kk)

        #Add sai to a so that b = kt*se^(-1)*kk + sa^(-1)
        b = np.add(sai,a)

        #Invert b so that c = (kt*se^(-1)*kk + sa^(-1))^(-1)
        c = np.linalg.inv(b)

        #Multiplying c by kt*sei (m from before) 
        dd = np.matmul(c,m) 

    aa = np.matmul(dd,kk)

    return dd,aa


###############################################################################################

def calc_serr(nx,ny,sa,se,dd,aa):


    """
        FUNCTION NAME : calc_serr()
        
        DESCRIPTION : 

            This subroutine calculates the error covariance matrices after the final iteration has been completed.

            The subroutine calculates the MEASUREMENT covariance matrix according to the 
            equation (re: p130 of Houghton, Taylor and Rodgers) :
               
                                  sm = dd*se*dd_T

            The subroutine calculates the SMOOTHING error covariance matrix according to the equation:
  
                                  sn = (aa-I)*sx*(aa-I)_T  

            The subroutine also calculates the TOTAL error matrix:

                                  st=sn+sm

        INPUTS :
       
            nx :: Number of elements in state vector 
            ny :: Number of elements in measurement vector
            sa(mx,mx) :: A priori covariance matric
            se(my,my) :: Measurement error covariance matrix
            dd(mx,my) :: Gain matrix
            aa(mx,mx) :: Averaging kernels

        OPTIONAL INPUTS: none
        
        OUTPUTS :

            sm(nx,nx) :: Final measurement covariance matrix
            sn(nx,nx) :: Final smoothing error covariance matrix
            st(nx,nx) :: Final full covariance matrix
 
        CALLING SEQUENCE:
        
            sm,sn,st = calc_serr(nx,ny,sa,se,dd,aa)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    #Multiplying dd*se
    a = np.matmul(dd,se)

    #Multiplying a*dt so that dd*se*dt
    dt = np.transpose(dd)
    sm = np.matmul(a,dt)

    #Calculate aa-ii where I is a diagonal matrix
    b = np.zeros([nx,nx])
    for i in range(nx):
        for j in range(nx):
            b[i,j] = aa[i,j]
        b[i,i] = b[i,i] - 1.0
    bt = np.transpose(b)

    #Multiply b*sa so that (aa-I)*sa
    c = np.matmul(b,sa)
  
    #Multiply c*bt so tthat (aa-I)*sx*(aa-I)_T  
    sn = np.matmul(c,bt)

    #Add sn and sm and get total retrieved error
    st = np.add(sn,sm)

    return sm,sn,st

###############################################################################################

def calc_phiret(ny,y,yn,se,nx,xn,xa,sa):

    """
        FUNCTION NAME : calc_phiret_nemesis()
        
        DESCRIPTION : 

            Calculate the retrieval cost function to be minimised in the optimal estimation 
            framework, which combines departure from a priori and closeness to spectrum.
 
        INPUTS :
      
            ny :: Number of elements in measurement vector
            y(my) :: Measurement vector
            yn(my) :: Modelled measurement vector
            se(my,my) :: Measurement error covariance matrix
            nx :: Number of elements in state vector 
            xn(mx) :: State vector
            xa(mx) :: A priori state vector
            sa(mx,mx) :: A priori covariance matrix       
 
        OPTIONAL INPUTS: none
        
        OUTPUTS :

            chisq :: Closeness of fit to measurement vector
            phi :: Total cost function
 
        CALLING SEQUENCE:
        
            chisq,phi = calc_phiret_nemesis(ny,y,yn,se,nx,xn,xa,sa)       
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    #Calculating yn-y
    b = np.zeros([ny,1])
    b[:,0] = yn[0:ny] - y[0:ny]
    bt = np.transpose(b)

    #Calculating inverse of sa and se
    sai = np.linalg.inv(sa)
#    sei = np.linalg.inv(se)
    sei = np.zeros([ny,ny])
    for i in range(ny):
        sei[i,i] = 1./se[i,i]  #As it is just a diagonal matrix

    #Multiplying se^(-1)*b
    a = np.matmul(sei,b)
 
    #Multiplying bt*a so that (yn-y)^T * se^(-1) * (yn-y)
    c = np.matmul(bt,a)

    phi1 = c[0,0]
    chisq = phi1

    #Calculating xn-xa
    d = np.zeros([nx,1])
    d[:,0] = xn[0:nx] - xa[0:nx]
    dt = np.transpose(d)
   
    #Multiply sa^(-1)*d 
    e = np.matmul(sai,d)

    #Multiply dt*e so that (xn-xa)^T * sa^(-1) * (xn-xa)
    f = np.matmul(dt,e)

    phi2 = f[0,0]
   
    print('calc_phiret_nemesis: phi1,phi2 = '+str(phi1)+','+str(phi2)+')')
    phi = phi1+phi2

    return chisq,phi

###############################################################################################

def assess(nx,ny,kk,sa,se):


    """
        FUNCTION NAME : assess()
        
        DESCRIPTION : 

            This subroutine assesses the retrieval matrices to see
            whether an exact retrieval may be expected.

            One formulation of the gain matrix is dd = sx*kk_T*(kk*sx*kk_T + se)^-1

            If the retrieval is exact, the se will be very small. Since Se is
            diagonal all we need do is compare to  the diagonal elements of
 
        INPUTS :
      
            nx :: Number of elements in state vector 
            ny :: Number of elements in measurement vector
            kk(my,mx) :: Jacobian matrix
            sa(mx,mx) :: A priori covariance matric
            se(my,my) :: Measurement error covariance matrix
 
        OPTIONAL INPUTS: none
        
        OUTPUTS : none

        CALLING SEQUENCE:
        
            assess(nx,ny,kk,sa,se)       
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    #Calculating transpose of kk
    kt = np.transpose(kk)

    #Multiply sa*kt
    m = np.matmul(sa,kt)

    #Multiply kk*m so that a = kk*sa*kt
    a = np.matmul(kk,m)

    #Add se to a
    b = np.add(a,se)

    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for i in range(ny):
            sum1 = sum1 + b[i,i]
            sum2 = sum2 + se[i,i]
            sum3 = sum3 + b[i,i]/se[i,i]

    sum1 = sum1/ny
    sum2 = sum2/ny
    sum3 = sum3/ny
  
    print('Assess:')
    print('Average of diagonal elements of Kk*Sx*Kt : '+str(sum1))
    print('Average of diagonal elements of Se : '+str(sum2))
    print('Ratio = '+str(sum1/sum2))
    print('Average of Kk*Sx*Kt/Se element ratio : '+str(sum3))
    if sum3 > 10.0:
        print('******************* ASSESS WARNING *****************')
        print('Insufficient constraint. Solution likely to be exact')
        print('****************************************************')
