from NemesisPy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from copy import *
import pickle

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

    def __init__(self, IRET=0, NITER=1, NX=1, NY=1, PHILIMIT=0.1, NCORES=1):

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
        @param NCORES: int,
            Number of cores available for parallel computations

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
        OptimalEstimation.read_hdf5()
        OptimalEstimation.write_input_hdf5()
        OptimalEstimation.write_output_hdf5()
        OptimalEstimation.edit_Y()
        OptimalEstimation.edit_SE()
        OptimalEstimation.edit_YN()
        OptimalEstimation.edit_XA()
        OptimalEstimation.edit_SA()
        OptimalEstimation.edit_XN()
        OptimalEstimation.edit_KK()
        OptimalEstimation.calc_gain_matrix()
        OptimalEstimation.plot_bestfit()
        """

        #Input parameters
        self.IRET = IRET
        self.NITER = NITER
        self.NX = NX
        self.NY = NY
        self.PHILIMIT = PHILIMIT      
        self.NCORES = NCORES  

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

    def assess_input(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.IRET), np.integer) == True , \
            'IRET must be int'
        assert self.IRET == 0 , \
            'IRET must be =0 for now'

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NITER), np.integer) == True , \
            'NITER must be int'
            
        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NCORES), np.integer) == True , \
            'NCORES must be int'
        assert self.NCORES >= 1 , \
            'NCORES must be >= 1'

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.PHILIMIT), np.float64) == True , \
            'IRET must be int'
        assert self.PHILIMIT > 0 , \
            'PHILIMIT must be >0'

    def write_input_hdf5(self,runname):
        """
        Write the Retrieval properties into an HDF5 file
        """

        import h5py

        #Assessing that all the parameters have the correct type and dimension
        self.assess_input()

        f = h5py.File(runname+'.h5','a')
        #Checking if Retrieval already exists
        if ('/Retrieval' in f)==True:
            del f['Retrieval']   #Deleting the Atmosphere information that was previously written in the file

        grp = f.create_group("Retrieval")

        dset = grp.create_dataset('NITER',data=self.NITER)
        dset.attrs['title'] = "Maximum number of iterations"
        
        dset = grp.create_dataset('NCORES',data=self.NCORES)
        dset.attrs['title'] = "Number of cores available for parallel computations"

        dset = grp.create_dataset('PHILIMIT',data=self.PHILIMIT)
        dset.attrs['title'] = "Percentage convergence limit"
        dset.attrs['units'] = "%"

        dset = grp.create_dataset('IRET',data=self.IRET)
        dset.attrs['title'] = "Retrieval engine type"
        if self.IRET==0:
            dset.attrs['type'] = "Optimal Estimation"

        f.close()

    def write_output_hdf5(self,runname,Variables,write_cov=True):
        """
        Write the Retrieval outputs into an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','a')
        
        #Checking if Retrieval already exists
        if ('/Retrieval' in f)==True:
            del f['Retrieval']   #Deleting the Atmosphere information that was previously written in the file

        grp = f.create_group("Retrieval")

        dset = grp.create_dataset('NITER',data=self.NITER)
        dset.attrs['title'] = "Maximum number of iterations"

        dset = grp.create_dataset('PHILIMIT',data=self.PHILIMIT)
        dset.attrs['title'] = "Percentage convergence limit"
        dset.attrs['units'] = "%"

        dset = grp.create_dataset('IRET',data=self.IRET)
        dset.attrs['title'] = "Retrieval engine type"
        if self.IRET==0:
            dset.attrs['type'] = "Optimal Estimation"

        #Optimal Estimation
        #####################################################################

        if self.IRET==0:
            
            dset = f.create_dataset('Retrieval/Output/OptimalEstimation/NY',data=self.NY)
            dset.attrs['title'] = "Number of elements in measurement vector"

            dset = f.create_dataset('Retrieval/Output/OptimalEstimation/Y',data=self.Y)
            dset.attrs['title'] = "Measurement vector"

            YERR = np.zeros(self.NY)
            for i in range(self.NY):
                YERR[i] = np.sqrt(self.SE[i,i])

            dset = f.create_dataset('Retrieval/Output/OptimalEstimation/YERR',data=YERR)
            dset.attrs['title'] = "Uncertainty in Measurement vector"

            dset = f.create_dataset('Retrieval/Output/OptimalEstimation/YN',data=self.YN)
            dset.attrs['title'] = "Modelled measurement vector"
            
            if write_cov==True:

                dset = f.create_dataset('Retrieval/Output/OptimalEstimation/NX',data=self.NX)
                dset.attrs['title'] = "Number of elements in state vector"

                dset = f.create_dataset('Retrieval/Output/OptimalEstimation/XN',data=self.XN)
                dset.attrs['title'] = "Retrieved state vector"

                dset = f.create_dataset('Retrieval/Output/OptimalEstimation/SX',data=self.ST)
                dset.attrs['title'] = "Retrieved covariance matrix"

                dset = f.create_dataset('Retrieval/Output/OptimalEstimation/XA',data=self.XA)
                dset.attrs['title'] = "A priori state vector"

                dset = f.create_dataset('Retrieval/Output/OptimalEstimation/SA',data=self.SA)
                dset.attrs['title'] = "A priori covariance matrix"
                
                dset = f.create_dataset('Retrieval/Output/OptimalEstimation/SY',data=self.SE)
                dset.attrs['title'] = "Measurement vector covariance matrix"



        #Writing the parameters in the same form as the input .apr file
        APRPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        APRERRPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        RETPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        RETERRPARAM = np.zeros((Variables.NXVAR.max(),Variables.NVAR))
        ix = 0
        for ivar in range(Variables.NVAR):

            for i in range(Variables.NXVAR[ivar]):
                
                xa1 = self.XA[ix]
                ea1 = np.sqrt(abs(self.SA[ix,ix]))
                xn1 = self.XN[ix]
                en1 = np.sqrt(abs(self.ST[ix,ix]))
                if Variables.LX[ix]==1:
                    xa1 = np.exp(xa1)
                    ea1 = xa1*ea1
                    xn1 = np.exp(xn1)
                    en1 = xn1*en1

                RETPARAM[i,ivar] = xn1
                RETERRPARAM[i,ivar] = en1
                APRPARAM[i,ivar] = xa1
                APRERRPARAM[i,ivar] = ea1

                ix = ix + 1


        dset = f.create_dataset('Retrieval/Output/Parameters/NVAR',data=Variables.NVAR)
        dset.attrs['title'] = "Number of retrieved model parameterisations"

        dset = f.create_dataset('Retrieval/Output/Parameters/NXVAR',data=Variables.NXVAR)
        dset.attrs['title'] = "Number of parameters associated with each model parameterisation"

        dset = f.create_dataset('Retrieval/Output/Parameters/VARIDENT',data=Variables.VARIDENT)
        dset.attrs['title'] = "Variable parameterisation ID"

        dset = f.create_dataset('Retrieval/Output/Parameters/VARPARAM',data=Variables.VARPARAM)
        dset.attrs['title'] = "Extra parameters required to model the parameterisations (not retrieved)"

        dset = f.create_dataset('Retrieval/Output/Parameters/RETPARAM',data=RETPARAM)
        dset.attrs['title'] = "Retrieved parameters required to model the parameterisations"

        dset = f.create_dataset('Retrieval/Output/Parameters/RETERRPARAM',data=RETERRPARAM)
        dset.attrs['title'] = "Uncertainty in the retrieved parameters required to model the parameterisations"

        dset = f.create_dataset('Retrieval/Output/Parameters/APRPARAM',data=APRPARAM)
        dset.attrs['title'] = "A priori parameters required to model the parameterisations"

        dset = f.create_dataset('Retrieval/Output/Parameters/APRERRPARAM',data=APRERRPARAM)
        dset.attrs['title'] = "Uncertainty in the a priori parameters required to model the parameterisations"

        f.close()

    def read_hdf5(self,runname):
        """
        Read the Retrieval properties from an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','r')

        #Checking if Surface exists
        e = "/Retrieval" in f
        if e==False:
            sys.exit('error :: Retrieval is not defined in HDF5 file')
        else:

            self.NITER = np.int32(f.get('Retrieval/NITER'))
            self.IRET = np.int32(f.get('Retrieval/IRET'))
            self.PHILIMIT = np.float64(f.get('Retrieval/PHILIMIT'))

            #Checking if Retrieval already exists
            if ('/Retrieval/Output' in f)==True:

                self.NX = np.int32(f.get('Retrieval/Output/OptimalEstimation/NX'))
                self.NY = np.int32(f.get('Retrieval/Output/OptimalEstimation/NY'))

                self.XN = np.array(f.get('Retrieval/Output/OptimalEstimation/XN'))
                self.XA = np.array(f.get('Retrieval/Output/OptimalEstimation/XA'))
                self.ST = np.array(f.get('Retrieval/Output/OptimalEstimation/SX'))
                self.SA = np.array(f.get('Retrieval/Output/OptimalEstimation/SA'))

                self.YN = np.array(f.get('Retrieval/Output/OptimalEstimation/YN'))
                self.Y = np.array(f.get('Retrieval/Output/OptimalEstimation/Y'))
                self.SE = np.array(f.get('Retrieval/Output/OptimalEstimation/SE'))

        f.close()

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

        # Calculating the transpose of kk
        kt = self.KK.T

        # Calculating the gain matrix dd
        if self.NX == self.NY:
            # Calculate kk*sa*kt
            a = self.KK @ (self.SA @ kt) + self.SE

            # Inverting a
            c = np.linalg.inv(a)

            # Multiplying (sa*kt) by c
            self.DD = (self.SA @ kt) @ c

        else:
            # Calculating the inverse of Sa and Se
            sai = np.linalg.inv(self.SA)
            sei_inv = np.diag(1.0 / np.diag(self.SE))

            # Calculate kt*sei_inv*kk
            a = kt @ sei_inv @ self.KK + sai

            # Invert a
            c = np.linalg.inv(a)

            # Multiplying c by (kt*sei_inv)
            self.DD = c @ (kt @ sei_inv)

        self.AA = self.DD @ self.KK

    def calc_phiret(self):
        """
        Calculate the retrieval cost function to be minimized in the optimal estimation
        framework, which combines departure from a priori and closeness to spectrum.
        """

        # Calculating yn-y
        b = self.YN[:self.NY] - self.Y[:self.NY]
        bt = b.T

        # Calculating inverse of sa and se
        sai = np.linalg.inv(self.SA)
        sei_inv = np.diag(1.0 / np.diag(self.SE))

        # Multiplying se_inv*b
        a = sei_inv @ b

        # Multiplying bt*a so that (yn-y)^T * se_inv * (yn-y)
        c = bt @ a

        phi1 = c
        self.CHISQ = phi1 / self.NY

        # Calculating xn-xa
        d = self.XN[:self.NX] - self.XA[:self.NX]
        dt = d.T

        # Multiply sa_inv*d
        e = sai @ d

        # Multiply dt*e so that (xn-xa)^T * sa_inv * (xn-xa)
        f = dt @ e

        phi2 = f

        print('calc_phiret: phi1, phi2 = ' + str(phi1) + ', ' + str(phi2) + ')')
        self.PHI = phi1 + phi2

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

        #sum1 = 0.0
        #sum2 = 0.0
        #sum3 = 0.0
        #for i in range(self.NY):
        #    sum1 = sum1 + b[i,i]
        #    sum2 = sum2 + self.SE[i,i]
        #    sum3 = sum3 + b[i,i]/self.SE[i,i]

        sum1 = np.sum(np.diagonal(b))
        sum2 = np.sum(np.diagonal(self.SE))
        sum3 = np.sum(np.diagonal(b)/np.diagonal(self.SE))

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
        b = deepcopy(self.AA)
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
                        strspec = "\t %4i %14.8f %15.8e %15.8e %7.2f %15.8e %9.5f \n"
                    elif Measurement.IFORM==1:
                        strspec = "\t %4i %10.4f %15.8e %15.8e %7.2f %15.8e %9.5f \n"
                    elif Measurement.IFORM==2:
                        strspec = "\t %4i %9.4f %12.6e %12.6e %6.2f %12.6e %6.2f \n"
                    elif Measurement.IFORM==3:
                        strspec = "\t %4i %10.4f %15.8e %15.8e %7.2f %15.8e %9.5f \n"
                    else:
                        strspec = "\t %4i %14.8f %15.8e %15.8e %7.2f %15.8e %9.5f \n"

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

    def write_cov(self,runname,Variables,pickle=False):
        """
        Write information about the Optimal Estimation matrices into the .cov file

        @param runname: str
            Name of the NEMESIS run
        @param Variables: class
            Python class describing the different parameterisations retrieved
        """

        if pickle==False:
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

        else:

            import pickle
            filehandler = open(runname+'.cov',"wb")
            pickle.dump(self,filehandler,pickle.HIGHEST_PROTOCOL)

    def read_cov(self,runname,Variables=None,pickle=False):
        """
        Write information about the Optimal Estimation matrices into the .cov file

        @param runname: str
            Name of the NEMESIS run
        @param Variables: class
            Python class describing the different parameterisations retrieved
        """

        if pickle==False:
            if Variables==None:
                Variables=Variables_0()
        
            f = open(runname+'.cov','r')

            #Reading variables that were retrieved
            tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
            npro = int(tmp[0])
            nvar = int(tmp[1])

            varident = np.zeros([nvar,3],dtype='int')
            varparam = np.zeros([nvar,5],dtype='int')
            for i in range(nvar):
                tmp = np.fromfile(f,sep=' ',count=3,dtype='int')
                varident[i,:] = tmp[:]

                tmp = np.fromfile(f,sep=' ',count=5,dtype='float')
                varparam[i,:] = tmp[:] 

            Variables.NVAR = nvar
            Variables.VARIDENT = varident
            Variables.VARPARAM = varparam
            Variables.calc_NXVAR(npro)

            #Reading optimal estimation matrices
            tmp = np.fromfile(f,sep=' ',count=2,dtype='int')
            nx = int(tmp[0])
            ny = int(tmp[1])

            sa = np.zeros([nx,nx])
            sm = np.zeros([nx,nx])
            sn = np.zeros([nx,nx])
            st = np.zeros([nx,nx])
            aa = np.zeros([nx,nx])
            dd = np.zeros([nx,ny])
            kk = np.zeros([ny,nx])
            se = np.zeros([ny,ny])
            for i in range(nx):
                for j in range(nx):
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    sa[i,j] = tmp[0]
                for j in range(nx):
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    sm[i,j] = tmp[0]
                for j in range(nx):
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    sn[i,j] = tmp[0]
                for j in range(nx):
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    st[i,j] = tmp[0]


            for i in range(nx):
                for j in range(nx):
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    aa[i,j] = tmp[0]


            for i in range(nx):
                for j in range(ny):
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    dd[i,j] = tmp[0]


            for i in range(ny):
                for j in range(nx):
                    tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                    kk[i,j] = tmp[0]

            for i in range(ny):
                tmp = np.fromfile(f,sep=' ',count=1,dtype='float')
                se[i,i] = tmp[0]

            f.close()

            self.NX = nx
            self.NY = ny
            self.edit_SA(sa)
            self.edit_SE(se)
            self.SM = sm
            self.SN = sn
            self.ST = st
            self.DD = dd
            self.AA = aa
            self.edit_KK(kk)

        else:

            import pickle

            filen = open(runname+'.cov','rb')
            pickleobj = pickle.load(filen)
            self.NX = pickleobj.NX
            self.NY = pickleobj.NY
            self.SA = pickleobj.SA
            self.SE = pickleobj.SE
            self.SM = pickleobj.SM
            self.SN = pickleobj.SN
            self.ST = pickleobj.ST
            self.DD = pickleobj.DD
            self.AA = pickleobj.AA
            self.KK = pickleobj.KK

    def plot_K(self):
        """
        Function to plot the Jaxobian matrix
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig,ax1 = plt.subplots(1,1,figsize=(10,3))
        im = ax1.imshow(np.transpose(self.KK),aspect='auto',origin='lower',cmap='jet')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Gradients (dR/dx)')
        ax1.grid()
        plt.tight_layout()
        plt.show()

    def plot_bestfit(self):
        """
        Function to plot the comparison between modelled and measured spectra
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        #fig,ax1 = plt.subplots(1,1,figsize=(10,3))
        fig = plt.figure(figsize=(10,4))
        ax1 = plt.subplot2grid((1,3),(0,0),colspan=1,rowspan=2)
        ax2 = plt.subplot2grid((1,3),(0,2),colspan=1,rowspan=1)
        ax1.plot(range(self.NY),self.Y,c='black',label='Measured spectra')
        ax1.plot(range(self.NY),self.YN,c='tab:red',label='Modelled spectra')
        ax2.plot(range(self.NY),self.Y-self.YN,c='tab:red')
        ax1.set_xlabel('Measurement vector element #')
        ax1.set_ylabel('Radiance')
        ax1.grid()
        plt.tight_layout()
        plt.show()


###############################################################################################
###############################################################################################
#   OPTIMAL ESTIMATION CONVERGENCE LOOP
###############################################################################################
###############################################################################################

def coreretOE(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,\
                 NITER=10,PHILIMIT=0.1,NCores=1,nemesisSO=False,write_itr=True):


    """
        FUNCTION NAME : coreretOE()
        
        DESCRIPTION : 

            This subroutine runs the Optimal Estimation iterarive algorithm to solve the inverse
            problem and find the set of parameters that fit the spectral measurements and are closest
            to the a priori estimates of the parameters.

        INPUTS :
       
            runname :: Name of the Nemesis run
            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the spectroscopic parameters of gaseous species
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations

        OPTIONAL INPUTS:

            NITER :: Number of iterations in retrieval
            PHILIMIT :: Percentage convergence limit. If the percentage reduction of the cost function PHI
                        is less than philimit then the retrieval is deemed to have converged.

            nemesisSO :: If True, the retrieval uses the function jacobian_nemesisSO(), adapated specifically
                         for solar occultation observations, rather than the more general jacobian_nemesis() function.

        OUTPUTS :

            OptimalEstimation :: Python class defining all the variables required as input or output
                                 from the Optimal Estimation retrieval
 
        CALLING SEQUENCE:
        
            OptimalEstimation = coreretOE(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,Layer)
 
        MODIFICATION HISTORY : Juan Alday (06/08/2021)

    """

    from NemesisPy import OptimalEstimation_0
    from NemesisPy import ForwardModel_0

    #Creating class and including inputs
    #############################################

    OptimalEstimation = OptimalEstimation_0()

    OptimalEstimation.NITER = NITER
    OptimalEstimation.PHILIMIT = PHILIMIT
    OptimalEstimation.NX = Variables.NX
    OptimalEstimation.NY = Measurement.NY
    OptimalEstimation.edit_XA(Variables.XA)
    OptimalEstimation.edit_XN(Variables.XN)
    OptimalEstimation.edit_SA(Variables.SA)
    OptimalEstimation.edit_Y(Measurement.Y)
    OptimalEstimation.edit_SE(Measurement.SE)

    #Opening .itr file
    #################################################################

    if OptimalEstimation.NITER>0:
        if write_itr==True:
            fitr = open(runname+'.itr','w')
            fitr.write("\t %i \t %i \t %i\n" % (OptimalEstimation.NX,OptimalEstimation.NY,OptimalEstimation.NITER))

    #Calculate the first measurement vector and jacobian matrix
    #################################################################

    ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
    print('nemesis :: Calculating Jacobian matrix KK')
    YN,KK = ForwardModel.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)
    
    OptimalEstimation.edit_YN(YN)
    OptimalEstimation.edit_KK(KK)

    #Calculate gain matrix and average kernels
    #################################################################

    print('nemesis :: Calculating gain matrix')
    OptimalEstimation.calc_gain_matrix()

    #Calculate initial value of cost function phi
    #################################################################

    print('nemesis :: Calculating cost function')
    OptimalEstimation.calc_phiret()

    OPHI = OptimalEstimation.PHI
    print('chisq/ny = '+str(OptimalEstimation.CHISQ))

    #Assessing whether retrieval is going to be OK
    #################################################################

    OptimalEstimation.assess()

    #Run retrieval for each iteration
    #################################################################

    #Initializing some variables
    alambda = 1.0   #Marquardt-Levenberg-type 'braking parameter'
    NX11 = np.zeros(OptimalEstimation.NX)
    XN1 = deepcopy(OptimalEstimation.XN)
    NY1 = np.zeros(OptimalEstimation.NY)
    YN1 = deepcopy(OptimalEstimation.YN)

    for it in range(OptimalEstimation.NITER):

        print('nemesis :: Iteration '+str(it)+'/'+str(OptimalEstimation.NITER))

        if write_itr==True:
            
        #Writing into .itr file
        ####################################

            fitr.write('%10.5f %10.5f \n' % (OptimalEstimation.CHISQ,OptimalEstimation.PHI))
            for i in range(OptimalEstimation.NX):fitr.write('%10.5f \n' % (XN1[i]))
            for i in range(OptimalEstimation.NX):fitr.write('%10.5f \n' % (OptimalEstimation.XA[i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.Y[i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.SE[i,i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (YN1[i]))
            for i in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.YN[i]))
            for i in range(OptimalEstimation.NX):
                for j in range(OptimalEstimation.NY):fitr.write('%10.5f \n' % (OptimalEstimation.KK[j,i]))


        #Calculating next state vector
        #######################################

        print('nemesis :: Calculating next iterated state vector')
        X_OUT = OptimalEstimation.calc_next_xn()
        #  x_out(nx) is the next iterated value of xn using classical N-L
        #  optimal estimation. However, we want to apply a braking parameter
        #  alambda to stop the new trial vector xn1 being too far from the
        #  last 'best-fit' value xn

        IBRAKE = 0
        while IBRAKE==0: #We continue in this while loop until we do not find problems with the state vector
    
            for j in range(OptimalEstimation.NX):
                XN1[j] = OptimalEstimation.XN[j] + (X_OUT[j]-OptimalEstimation.XN[j])/(1.0+alambda)
                
                #Check to see if log numbers have gone out of range
                if Variables.LX[j]==1:
                    if((XN1[j]>85.) or (XN1[j]<-85.)):
                        print('nemesis :: log(number gone out of range) --- increasing brake')
                        alambda = alambda * 10.
                        IBRAKE = 0
                        if alambda>1.e30:
                            sys.exit('error in nemesis :: Death spiral in braking parameters - stopping')
                        break
                    else:
                        IBRAKE = 1
                else:
                    IBRAKE = 1
                    pass
                        
            if IBRAKE==0:
                continue
                        
            #Check to see if any VMRs or other parameters have gone negative.
            Variables1 = deepcopy(Variables)
            Variables1.XN = XN1

            ForwardModel1 = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables1)
            #Variables1 = copy(Variables)
            #Variables1.XN = XN1
            #Measurement1 = copy(Measurement)
            #Atmosphere1 = copy(Atmosphere)
            #Scatter1 = copy(Scatter)
            #Stellar1 = copy(Stellar)
            #Surface1 = copy(Surface)
            #Spectroscopy1 = copy(Spectroscopy)
            #Layer1 = copy(Layer)
            #flagh2p = False
            #xmap = subprofretg(runname,Variables1,Measurement1,Atmosphere1,Spectroscopy1,Scatter1,Stellar1,Surface1,Layer1,flagh2p)
            ForwardModel1.subprofretg()

            #if(len(np.where(Atmosphere1.VMR<0.0))>0):
            #    print('nemesisSO :: VMR has gone negative --- increasing brake')
            #    alambda = alambda * 10.
            #    IBRAKE = 0
            #    continue
            
            #iwhere = np.where(Atmosphere1.T<0.0)
            iwhere = np.where(ForwardModel1.AtmosphereX.T<0.0)
            if(len(iwhere[0])>0):
                print('nemesis :: Temperature has gone negative --- increasing brake')
                alambda = alambda * 10.
                IBRAKE = 0
                continue


        #Calculate test spectrum using trial state vector xn1. 
        #Put output spectrum into temporary spectrum yn1 with
        #temporary kernel matrix kk1. Does it improve the fit? 
        Variables.edit_XN(XN1)
        print('nemesis :: Calculating Jacobian matrix KK')

        ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
        YN1,KK1 = ForwardModel.jacobian_nemesis(NCores=NCores,nemesisSO=nemesisSO)

        OptimalEstimation1 = deepcopy(OptimalEstimation)
        OptimalEstimation1.edit_YN(YN1)
        OptimalEstimation1.edit_XN(XN1)
        OptimalEstimation1.edit_KK(KK1)
        OptimalEstimation1.calc_phiret()
        print('chisq/ny = '+str(OptimalEstimation1.CHISQ))

        #Does the trial solution fit the data better?
        if (OptimalEstimation1.PHI <= OPHI):
            print('Successful iteration. Updating xn,yn and kk')
            OptimalEstimation.edit_XN(XN1)
            OptimalEstimation.edit_YN(YN1)
            OptimalEstimation.edit_KK(KK1)
            Variables.edit_XN(XN1)

            #Now calculate the gain matrix and averaging kernels
            OptimalEstimation.calc_gain_matrix()

            #Updating the cost function
            OptimalEstimation.calc_phiret()

            #Has the solution converged?
            tphi = 100.0*(OPHI-OptimalEstimation.PHI)/OPHI
            if (tphi>=0.0 and tphi<=OptimalEstimation.PHILIMIT and alambda<1.0):
                print('phi, phlimit : '+str(tphi)+','+str(OptimalEstimation.PHILIMIT))
                print('Phi has converged')
                print('Terminating retrieval')
                break
            else:
                OPHI=OptimalEstimation.PHI
                alambda = alambda*0.3  #reduce Marquardt brake

        else:
            #Leave xn and kk alone and try again with more braking
            alambda = alambda*10.0  #increase Marquardt brake


    #Calculating output parameters
    ######################################################

    #Calculating retrieved covariance matrices
    OptimalEstimation.calc_serr()

    #Make sure errors stay as a priori for kiter < 0
    if OptimalEstimation.NITER<0:
        OptimalEstimation.ST = deepcopy(OptimalEstimation.SA)

    #Closing .itr file
    if write_itr==True:
        if OptimalEstimation.NITER>0:
            fitr.close()

    #Writing the contribution of each gas to .gcn file
    #if nemesisSO==True:
    #    calc_gascn(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

    return OptimalEstimation

