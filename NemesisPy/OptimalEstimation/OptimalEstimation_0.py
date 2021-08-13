from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from copy import *

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

Optimal Estimation Class. It includes all parameters that are relevant for the retrieval of parameters using
                          the Optimal Estimation formalism
"""

class OptimalEstimation_0:

    def __init__(self, NITER=1, NX=1, NY=1, PHILIMIT=0.1):

        """
        Inputs
        ------
        @param NITER: int,
            Number of iterations in retrieval 
        @param PHILIMIT: real,
            Percentage convergence limit. If the percentage reduction of the cost function PHI
            is less than philimit then the retrieval is deemed to have converged.
        @param NY: int,
            Number of elements in measurement vector    
        @param NX: int,
            Number of elements in state vector

        Attributes
        ----------
        @attribute PHI: real
            Current value of the Cost function
        @attribute CHISQ: real
            Current value of the reduced chi-squared
        @attribute Y: 1D array
            Measurement vector
        @attribute SE: 1D array
            Measurement covariance matrix
        @attribute YN: 1D array
            Modelled measurement vector
        @attribute XA: 1D array
            A priori state vector
        @attribute SA: 1D array
            A priori covariance matrix
        @attribute XN: 1D array
            Current state vector
        @attribute KK: 2D array
            Jacobian matrix
        @attribute DD: 2D array
            Gain matrix
        @attribute AA: 2D array
            Averaging kernels        
        @attribute SM: 2D array
            Measurement error covariance matrix
        @attribute SN: 2D array
            Smoothing error covariance matrix
        @attribute ST: 2D array
            Retrieved error covariance matrix (SN+SM)

        Methods
        -------
        OptimalEstimation.edit_Y()
        OptimalEstimation.edit_SE()
        OptimalEstimation.edit_YN()
        OptimalEstimation.edit_XA()
        OptimalEstimation.edit_SA()
        OptimalEstimation.edit_XN()
        OptimalEstimation.edit_KK()
        OptimalEstimation.calc_gain_matrix()
        """

        #Input parameters
        self.NITER = NITER
        self.NX = NX
        self.NY = NY
        self.PHILIMIT = PHILIMIT       

        # Input the following profiles using the edit_ methods.
        self.KK = None #(NY,NX)
        self.DD = None #(NX,NY)
        self.AA = None #(NX,NX)
        self.SM = None #(NX,NX)
        self.SN = None #(NX,NX)
        self.ST = None #(NX,NX)
        self.Y= None #(NY)
        self.YN = None #(NY)
        self.SE = None #(NY,NY)
        self.XA = None #(NX)
        self.SA = None #(NX,NX)
        self.XN = None #(NX)

    def edit_KK(self, KK_array):
        """
        Edit the Jacobian Matrix
        @param KK_array: 2D array
            Jacobian matrix
        """
        KK_array = np.array(KK_array)
        assert KK_array.shape == (self.NY, self.NX),\
            'KK should be NY by NX.'

        self.KK = KK_array

    def edit_Y(self, Y_array):
        """
        Edit the measurement vector
        @param Y_array: 1D array
            Measurement vector
        """
        Y_array = np.array(Y_array)
        assert len(Y_array) == (self.NY),\
            'Y should be NY.'

        self.Y = Y_array

    def edit_YN(self, YN_array):
        """
        Edit the modelled measurement vector
        @param YN_array: 1D array
            Modelled measurement vector
        """
        YN_array = np.array(YN_array)
        assert len(YN_array) == (self.NY),\
            'YN should be NY.'

        self.YN = YN_array

    def edit_SE(self, SE_array):
        """
        Edit the Measurement covariance matrix
        @param SE_array: 2D array
            Measurement covariance matrix
        """
        SE_array = np.array(SE_array)
        assert SE_array.shape == (self.NY, self.NY),\
            'SE should be NY by NY.'
        self.SE = SE_array

    def edit_XN(self, XN_array):
        """
        Edit the current state vector
        @param XN_array: 1D array
            State vector
        """
        XN_array = np.array(XN_array)
        assert len(XN_array) == (self.NX),\
            'XN should be NX.'
        self.XN = XN_array

    def edit_XA(self, XA_array):
        """
        Edit the a priori state vector
        @param XA_array: 1D array
            A priori State vector
        """
        XA_array = np.array(XA_array)
        assert len(XA_array) == (self.NX),\
            'XA should be NX.'
        self.XA = XA_array

    def edit_SA(self, SA_array):
        """
        Edit the A priori covariance matrix
        @param SA_array: 2D array
            A priori covariance matrix
        """
        SA_array = np.array(SA_array)
        assert SA_array.shape == (self.NX, self.NX),\
            'SA should be NX by NX.'
        self.SA = SA_array

    def calc_gain_matrix(self):
        """
        Calculate gain matrix and averaging kernels. The gain matrix is calculated with
            dd = sx*kk_T*(kk*sx*kk_T + se)^-1    (if nx>=ny)
            dd = ((sx^-1 + kk_T*se^-1*kk)^-1)*kk_T*se^-1  (if ny>nx)
        """

        #Calculating the transpose of kk
        kt = np.transpose(self.KK)

        #Calculating the gain matrix dd
        if (self.NX >= self.NY):

            #Multiply sa*kt
            m = np.matmul(self.SA,kt)

            #Multiply kk*m so that a = kk*sa*kt
            a = np.matmul(self.KK,m)

            #Add se to a so that b = kk*sa*kt + se
            b = np.add(a,self.SE)

            #Inverting b so that we calculate c = (kk*sa*kt + se)^(-1)
            c = np.linalg.inv(b)

            #Multiplying sa*kt (m above) to c
            self.DD = np.matmul(m,c)

        else:

            #Calculating the inverse of Sa and Se
            sai = np.linalg.inv(self.SA)
#           sei = np.linalg.inv(se)
            sei = np.zeros([self.NY,self.NY])
            for i in range(self.NY):
                sei[i,i] = 1./self.SE[i,i]  #As it is just a diagonal matrix

            #Calculate kt*sei
            m = np.matmul(kt,sei)

            #Calculate m*kk so that kt*se^(-1)*kk
            a = np.matmul(m,self.KK)

            #Add sai to a so that b = kt*se^(-1)*kk + sa^(-1)
            b = np.add(sai,a)

            #Invert b so that c = (kt*se^(-1)*kk + sa^(-1))^(-1)
            c = np.linalg.inv(b)

            #Multiplying c by kt*sei (m from before) 
            self.DD = np.matmul(c,m) 

        self.AA = np.matmul(self.DD,self.KK)

    def calc_phiret(self):
        """
        Calculate the retrieval cost function to be minimised in the optimal estimation 
        framework, which combines departure from a priori and closeness to spectrum.
        """

        #Calculating yn-y
        b = np.zeros([self.NY,1])
        b[:,0] = self.YN[0:self.NY] - self.Y[0:self.NY]
        bt = np.transpose(b)

        #Calculating inverse of sa and se
        sai = np.linalg.inv(self.SA)
#       sei = np.linalg.inv(se)
        sei = np.zeros([self.NY,self.NY])
        for i in range(self.NY):
            sei[i,i] = 1./self.SE[i,i]  #As it is just a diagonal matrix

        #Multiplying se^(-1)*b
        a = np.matmul(sei,b)
 
        #Multiplying bt*a so that (yn-y)^T * se^(-1) * (yn-y)
        c = np.matmul(bt,a)

        phi1 = c[0,0]
        self.CHISQ = phi1/self.NY

        #Calculating xn-xa
        d = np.zeros([self.NX,1])
        d[:,0] = self.XN[0:self.NX] - self.XA[0:self.NX]
        dt = np.transpose(d)
   
        #Multiply sa^(-1)*d 
        e = np.matmul(sai,d)

        #Multiply dt*e so that (xn-xa)^T * sa^(-1) * (xn-xa)
        f = np.matmul(dt,e)

        phi2 = f[0,0]
   
        print('calc_phiret: phi1,phi2 = '+str(phi1)+','+str(phi2)+')')
        self.PHI = phi1+phi2


    def assess(self):
        """
        This subroutine assesses the retrieval matrices to see whether an exact retrieval may be expected.
        """

        #Calculating transpose of kk
        kt = np.transpose(self.KK)

        #Multiply sa*kt
        m = np.matmul(self.SA,kt)

        #Multiply kk*m so that a = kk*sa*kt
        a = np.matmul(self.KK,m)

        #Add se to a
        b = np.add(a,self.SE)

        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        for i in range(self.NY):
            sum1 = sum1 + b[i,i]
            sum2 = sum2 + self.SE[i,i]
            sum3 = sum3 + b[i,i]/self.SE[i,i]

        sum1 = sum1/self.NY
        sum2 = sum2/self.NY
        sum3 = sum3/self.NY
  
        print('Assess:')
        print('Average of diagonal elements of Kk*Sx*Kt : '+str(sum1))
        print('Average of diagonal elements of Se : '+str(sum2))
        print('Ratio = '+str(sum1/sum2))
        print('Average of Kk*Sx*Kt/Se element ratio : '+str(sum3))
        if sum3 > 10.0:
            print('******************* ASSESS WARNING *****************')
            print('Insufficient constraint. Solution likely to be exact')
            print('****************************************************')

    def calc_next_xn(self):
        """
        This subroutine performs the optimal estimation retrieval of the
        vector x from a set of measurements y and forward derivative matrix
        kk. The equation solved is (re: p147 of Houghton, Taylor and Rodgers):

                    xn+1 = x0 + dd*(y-yn) - aa*(x0 - xn)    
        """

        m1 = np.zeros([self.NY,1])
        m1[:,0] = self.Y - self.YN
        #dd1 = np.zeros([self.NX,self.NY])
        #dd1[0:nx,0:ny] = dd[0:nx,0:ny]

        m2 = np.zeros([self.NX,1])
        m2[:,0] = self.XA - self.XN
        #aa1 = np.zeros([nx,nx])
        #aa1[0:nx,0:nx] = aa[0:nx,0:nx]

        mp1 = np.matmul(self.DD,m1)
        mp2 = np.matmul(self.AA,m2)

        x_out = np.zeros(self.NX)

        for i in range(self.NX):
            x_out[i] = self.XA[i] + mp1[i,0] - mp2[i,0]

        return x_out

    def calc_serr(self):
        """
         Calculates the error covariance matrices after the final iteration has been completed.

        The subroutine calculates the MEASUREMENT ERROR covariance matrix according to the 
        equation (re: p130 of Houghton, Taylor and Rodgers) :
               
                                  sm = dd*se*dd_T

        The subroutine calculates the SMOOTHING ERROR covariance matrix according to the equation:
  
                                  sn = (aa-I)*sx*(aa-I)_T  

        The subroutine also calculates the TOTAL ERROR matrix:

                                  st=sn+sm
        """

        #Multiplying dd*se
        a = np.matmul(self.DD,self.SE)

        #Multiplying a*dt so that dd*se*dt
        dt = np.transpose(self.DD)
        self.SM = np.matmul(a,dt)

        #Calculate aa-ii where I is a diagonal matrix
        b = copy(self.AA)
        for i in range(self.NX):
            b[i,i] = b[i,i] - 1.0
        bt = np.transpose(b)

        #Multiply b*sa so that (aa-I)*sa
        c = np.matmul(b,self.SA)
  
        #Multiply c*bt so tthat (aa-I)*sx*(aa-I)_T  
        self.SN = np.matmul(c,bt)

        #Add sn and sm and get total retrieved error
        self.ST = np.add(self.SN,self.SM)



    def write_mre(self,runname,Variables,Measurement):
        """
        Write the results of an Optimal Estimation retrieval into the .mre file

        @param runname: str
            Name of the NEMESIS run
        @param Variables: class
            Python class describing the different parameterisations retrieved
        @param Measurement: class
            Python class descrbing the measurement and observation
        """

        #Opening file
        f = open(runname+'.mre','w')
    
        str1 = '! Total number of retrievals'
        nspec = 1
        f.write("\t" + str(nspec)+ "\t" + str1 + "\n")

        for ispec in range(nspec):
 
            #Writing first lines
            ispec1 = ispec + 1
            str2 = '! ispec,ngeom,ny,nx,ny'
            f.write("\t %i %i %i %i %i \t %s \n" % (ispec,Measurement.NGEOM,self.NY,self.NX,self.NY,str2)) 
            str3 = 'Latitude, Longitude'
            f.write("\t %5.7f \t %5.7f \t %s \n" % (Measurement.LATITUDE,Measurement.LONGITUDE,str3)) 

            if Measurement.ISPACE==0: #Wavenumber space (cm-1)
                if Measurement.IFORM==0:
                    str4='Radiances expressed as nW cm-2 sr-1 (cm-1)-1'       
                    xfac=1.0e9
                elif Measurement.IFORM==1:
                    str4='F_plan/F_star Ratio of planet'
                    xfac = 1.0
                elif Measurement.IFORM==2:
                    str4='Transit depth: 100*Planet_area/Stellar_area'
                    xfac = 1.0
                elif Measurement.IFORM==3:
                    str4='Spectral Radiation of planet: W (cm-1)-1'
                    xfac=1.0e18
                elif Measurement.IFORM==4:
                    str4='Solar flux: W cm-2 (cm-1)-1'
                    xfac=1.0
                elif Measurement.IFORM==5:
                    str4='Transmission'
                    xfac=1.0
                else:
                    print('warning in .mre :: IFORM not defined. Default=0')
                    str4='Radiances expressed as nW cm-2 sr-1 cm' 
                    xfac=1.0e9

            elif Measurement.ISPACE==1: #Wavelength space (um)

                if Measurement.IFORM==0:
                    str4='Radiances expressed as uW cm-2 sr-1 um-1'       
                    xfac=1.0e6
                elif Measurement.IFORM==1:
                    str4='F_plan/F_star Ratio of planet'
                    xfac = 1.0
                elif Measurement.IFORM==2:
                    str4='Transit depth: 100*Planet_area/Stellar_area'
                    xfac = 1.0
                elif Measurement.IFORM==3:
                    str4='Spectral Radiation of planet: W um-1'
                    xfac=1.0e18
                elif Measurement.IFORM==4:
                    str4='Solar flux: W cm-2 um-1'
                    xfac=1.0
                elif Measurement.IFORM==5:
                    str4='Transmission'
                    xfac=1.0
                else:
                    print('warning in .mre :: IFORM not defined. Default=0')
                    str4='Radiances expressed as uW cm-2 sr-1 um-1' 
                    xfac=1.0e6

            f.write(str4+"\n")

            #Writing spectra
            l = ['i','lambda','R_meas','error','%err','R_fit','%Diff']
            f.write("\t %s %s %s %s %s %s %s \n" % (l[0],l[1],l[2],l[3],l[4],l[5],l[6]))
            ioff = 0
            for igeom in range(Measurement.NGEOM):
                for iconv in range(Measurement.NCONV[igeom]):
                    i = ioff+iconv
                    err1 = np.sqrt(self.SE[i,i])
                    if self.Y[i] != 0.0:
                        xerr1 = abs(100.0*err1/self.Y[i])
                        relerr = abs(100.0*(self.Y[i]-self.YN[i])/self.Y[i])
                    else:
                        xerr1=-1.0
                        relerr1=-1.0

                    if Measurement.IFORM==0:
                        strspec = "\t %4i %14.8f %15.8e %15.8e %7.2f %15.8f %9.5f \n"
                    elif Measurement.IFORM==1:
                        strspec = "\t %4i %10.4f %15.8e %15.8e %7.2f %15.8f %9.5f \n"
                    elif Measurement.IFORM==2:
                        strspec = "\t %4i %9.4f %12.6e %12.6e %6.2f %12.6f %6.2f \n"
                    elif Measurement.IFORM==3:
                        strspec = "\t %4i %10.4f %15.8e %15.8e %7.2f %15.8f %9.5f \n"
                    else:
                        strspec = "\t %4i %14.8f %15.8e %15.8e %7.2f %15.8f %9.5f \n"

                    f.write(strspec % (i+1,Measurement.VCONV[iconv,igeom],self.Y[i]*xfac,err1*xfac,xerr1,self.YN[i]*xfac,relerr))
                
                ioff = ioff + Measurement.NCONV[igeom]     

            #Writing a priori and retrieved state vectors
            str1 = '' 
            f.write(str1+"\n")
            f.write('nvar=    '+str(Variables.NVAR)+"\n")
            
            nxtemp = 0
            for ivar in range(Variables.NVAR):

                f.write('Variable '+str(ivar+1)+"\n")
                f.write("\t %i \t %i \t %i\n" % (Variables.VARIDENT[ivar,0],Variables.VARIDENT[ivar,1],Variables.VARIDENT[ivar,2]))
                f.write("%10.8e \t %10.8e \t %10.8e \t %10.8e \t %10.8e\n" % (Variables.VARPARAM[ivar,0],Variables.VARPARAM[ivar,1],Variables.VARPARAM[ivar,2],Variables.VARPARAM[ivar,3],Variables.VARPARAM[ivar,4]))

                l = ['i','ix','xa','sa_err','xn','xn_err']
                f.write("\t %s %s %s %s %s %s\n" % (l[0],l[1],l[2],l[3],l[4],l[5]))
                for ip in range(Variables.NXVAR[ivar]):
                    ix = nxtemp + ip 
                    xa1 = self.XA[ix]
                    ea1 = np.sqrt(abs(self.SA[ix,ix]))
                    xn1 = self.XN[ix]
                    en1 = np.sqrt(abs(self.ST[ix,ix]))
                    if Variables.LX[ix]==1:
                        xa1 = np.exp(xa1)
                        ea1 = xa1*ea1
                        xn1 = np.exp(xn1)
                        en1 = xn1*en1
                    
                    strx = "\t %4i %4i %12.5e %12.5e %12.5e %12.5e \n"
                    f.write(strx % (ip+1,ix+1,xa1,ea1,xn1,en1))

                nxtemp = nxtemp + Variables.NXVAR[ivar]  

        f.close()  

    def write_cov(self,runname,Variables):
        """
        Write information about the Optimal Estimation matrices into the .cov file

        @param runname: str
            Name of the NEMESIS run
        @param Variables: class
            Python class describing the different parameterisations retrieved
        """

        #Open file
        f = open(runname+'.cov','w')

        npro=1
        f.write("%i %i\n" % (npro,Variables.NVAR))

        for ivar in range(Variables.NVAR):
            f.write("%i \t %i \t %i\n" % (Variables.VARIDENT[ivar,0],Variables.VARIDENT[ivar,1],Variables.VARIDENT[ivar,2]))
            f.write("%10.8e \t %10.8e \t %10.8e \t %10.8e \t %10.8e\n" % (Variables.VARPARAM[ivar,0],Variables.VARPARAM[ivar,1],Variables.VARPARAM[ivar,2],Variables.VARPARAM[ivar,3],Variables.VARPARAM[ivar,4]))

        f.write("%i %i\n" % (self.NX,self.NY))

        for i in range(self.NX):
            for j in range(self.NX):
                f.write("%10.8e\n" % (self.SA[i,j]))
            for j in range(self.NX):
                f.write("%10.8e\n" % (self.SM[i,j]))
            for j in range(self.NX):
                f.write("%10.8e\n" % (self.SN[i,j]))
            for j in range(self.NX):
                f.write("%10.8e\n" % (self.ST[i,j]))

        for i in range(self.NX):
            for j in range(self.NX):
                f.write("%10.8e\n" % (self.AA[i,j]))

        for i in range(self.NX):
            for j in range(self.NY):
                f.write("%10.8e\n" % (self.DD[i,j]))

        for i in range(self.NY):
            for j in range(self.NX):
                f.write("%10.8e\n" % (self.KK[i,j]))

        for i in range(self.NY):
            f.write("%10.8e\n" % (self.SE[i,i]))

        f.close() 