module hapke

    double precision, parameter :: pi = 3.1415926536

contains

    !==================================================================================================
    subroutine hapke_BRDF(nwave,ntheta,w,K,BS0,hs,BC0,hc,ROUGHNESS,G1,G2,F,i,e,phi,BRDF)

        !Subroutine to calculate the bidirectional reflectance distribution function of the surface
        !following Hapke's semi-empirical model from Hapke (2012) - Chapter 12
        !fow several wavelengths and angles

        !Inputs
        integer, intent(in) :: nwave,ntheta                             !Number of wavelengths and angles
        double precision, intent(in) :: w(nwave)                        !Single scattering albedo 
        double precision, intent(in) :: K(nwave)                        !Porosity coefficient
        double precision, intent(in) :: BS0(nwave),hs(nwave)            !Amplitude of opposition effect and width of the opposition surge
        double precision, intent(in) :: BC0(nwave),hc(nwave)            !Amplitude of the coherent backscatter opposition effect and width of the backscatter function
        double precision, intent(in) :: ROUGHNESS(nwave)                !Roughness mean slope angle
        double precision, intent(in) :: G1(nwave),G2(nwave),F(nwave)    !Parameters describing the double Henyey-Greenstein phase function
        double precision, intent(in) :: i(ntheta),e(ntheta),phi(ntheta) !Incident, reflection and azimuth angle (degrees)

        !Outputs
        double precision, intent(out) :: BRDF(nwave,ntheta)             !Bidirectional reflectance

        do iwave=1,nwave
            do itheta=1,ntheta
                call calc_BRDF(w(iwave),K(iwave),BS0(iwave),hs(iwave),BC0(iwave),hc(iwave),&
                                ROUGHNESS(iwave),G1(iwave),G2(iwave),F(iwave),&
                                i(itheta),e(itheta),phi(itheta),BRDF(iwave,itheta))
            enddo
        enddo

        return


    end subroutine hapke_BRDF


    !==================================================================================================
    subroutine calc_BRDF(w,K,BS0,hs,BC0,hc,ROUGHNESS,G1,G2,F,i,e,phi,BRDF)

        !Subroutine to calculate the bidirectional reflectance distribution function of the surface
        !following Hapke's semi-empirical model from Hapke (2012) - Chapter 12

        implicit none

        !Inputs
        double precision, intent(in) :: w         !Single scattering albedo 
        double precision, intent(in) :: K         !Porosity coefficient
        double precision, intent(in) :: BS0,hs    !Amplitude of opposition effect and width of the opposition surge
        double precision, intent(in) :: BC0,hc    !Amplitude of the coherent backscatter opposition effect and width of the backscatter function
        double precision, intent(in) :: ROUGHNESS !Roughness mean slope angle
        double precision, intent(in) :: G1,G2,F   !Parameters describing the double Henyey-Greenstein phase function
        double precision, intent(in) :: i,e,phi   !Incident, reflection and azimuth angle (degrees)

        !Local
        double precision :: mu,mu0,cg,g           !cos(e),cos(i),cos(g),scattering phase angle
        double precision :: gamma,r0,theta_bar    !Gamma factor, diffusive reflectance and corrected Roughness mean slope angle
        double precision :: chi,E1e,E2e,E1i,E2i   !Parameters from the Hapke formalism
        double precision :: mueff,mu0eff          !Effective emission and incident angles
        double precision :: fphi,S                !f(phi),Shadowing function
        double precision :: He,H0e                !H-functions
        double precision :: nue,nui               !nu-functions
        double precision :: Bs,Bc                 !shadow-hiding opposition function Bs; backscatter angular function Bc
        double precision :: phase                 !Phase function
        double precision :: phix

        !Output
        double precision, intent(out) :: BRDF     !Bidirectional reflectance


        if((e.ge.90.d0).or.(i.ge.90.d0))then

            BRDF = 0.d0

        else

            !Calculating the cosine of the angles
            mu = dcos(e/180.*pi)
            mu0 = dcos(i/180.*pi)

            !Correcting the azimuth angle to be within 0-180 degrees
            if(phi.gt.180.d0)then
                phix = 180.d0 - (phi-180.d0)
            else
                phix = phi
            endif


            !Calcualting the scattering phase angle
            cg = mu * mu0 + dsqrt(1.d0 - mu**2.) * dsqrt(1.d0 - mu0**2.) * dcos(phix/180.d0*pi) 
            if(cg.gt.1.d0)then
                cg = 1.0d0
            endif
            if(cg.lt.0.d0)then
                cg = 0.d0
            endif
            g = dacos(cg)*180.d0/pi

            !Calculate some of the input parameters for the Hapke formalism
            gamma = dsqrt(1.d0 - w)
            r0 = (1.d0 - gamma)/(1.d0 + gamma)
            theta_bar = ROUGHNESS * (1.d0 - r0)
            chi = 1.d0/dsqrt(1.d0 + pi * tan(theta_bar/180.d0*pi)**2.)
            if(phi.eq.180.d0)then
                fphi = 0.d0
            else
                fphi = dexp(-2.d0*dabs(dtan(phix/2.d0/180.d0*pi)))  !f(phi)
            endif

            !Calculating the E-functions
            call calc_E1(e,theta_bar,E1e)
            call calc_E2(e,theta_bar,E2e)
            call calc_E1(i,theta_bar,E1i)
            call calc_E2(i,theta_bar,E2i)

            !Calculating the nu functions
            call calc_nu(e,theta_bar,E1e,E2e,chi,nue)
            call calc_nu(i,theta_bar,E1i,E2i,chi,nui)

            !Calculating the effective incident and reflection angles
            call calc_mueff(i,e,phix,theta_bar,E1e,E1i,E2e,E2i,chi,mueff)
            call calc_mu0eff(i,e,phix,theta_bar,E1e,E1i,E2e,E2i,chi,mu0eff)

            !Calculating the shadowing function S
            if(i.le.e)then
                S = mueff/nue * mu0/nui * chi / (1.0 - fphi + fphi*chi*mu0/nui)
            else
                S = mueff/nue * mu0/nui * chi / (1.0 - fphi + fphi*chi*mu/nue)
            endif

            !Calculating the shadow-hiding opposition function Bs
            Bs = BS0 / ( 1.d0 + (1.d0/hs) * dtan( g/2.d0/180.d0*pi) )
 
            !Calculating the backscatter anfular function Bc
            Bc = BC0 / ( 1.d0 + (1.3d0 + K) * &
                ( (1.d0/hc*dtan( g/2.d0/180.d0*pi)) + (1.d0/hc*dtan( g/2.d0/180.d0*pi))**2.0 ) )

            !Calculating the Ambartsumian–Chandrasekhar H function
            call calc_H(w,mu0eff/K,r0,H0e)
            call calc_H(w,mueff/K,r0,He)

            !Calculate phase function (double Henyey-Greenstein function)
            call calc_hgphase(G1,G2,F,g,phase)

            !Calculating the bidirectional reflectance
            BRDF = K * w / (4.d0*pi) * mu0eff / (mu0eff + mueff) * &
                ( phase*(1.d0+Bs) + (H0e*He-1.d0) ) * (1.d0+Bc) * S

            !print*,cg,g,w,mu0eff / (mu0eff + mueff),phase,Bs,phase*(1.d0+Bs),(H0e*He-1.d0),(1.d0+Bc),S

        endif


        return

    end subroutine calc_BRDF
        

    !==================================================================================================
    subroutine calc_E1(x,theta_bar,E1)

        !Calculate the E1 function of the Hapke formalism (Hapke, 2012; p. 333)

        implicit none

        double precision, intent(in) :: x  !Angle (degrees)
        double precision, intent(in) :: theta_bar  !Corrected roughness mean slope angle (degrees)

        double precision, intent(out) :: E1 !Parameter E1 in the Hapke formalism

        E1 = dexp(-2.d0/pi * 1.d0/tan(theta_bar/180.d0*pi) * 1.d0/dtan(x/180.d0*pi))

        return

    end subroutine calc_E1


    !==================================================================================================
    subroutine calc_E2(x,theta_bar,E2)

        !Calculate the E1 function of the Hapke formalism (Hapke, 2012; p. 333)

        implicit none

        double precision, intent(in) :: x  !Angle (degrees)
        double precision, intent(in) :: theta_bar  !Corrected roughness mean slope angle (degrees)

        double precision, intent(out) :: E2 !Parameter E2 in the Hapke formalism

        E2 = dexp(-1.d0/pi * 1.d0/tan(theta_bar/180.d0*pi)**2. * 1.d0/tan(x/180.d0*pi)**2.)

        return

    end subroutine calc_E2


    !==================================================================================================
    subroutine calc_nu(x,theta_bar,E1,E2,chi,nu)

        !Calculate the nu function from the Hapke formalism (Hapke 2012 p.333) 

        implicit none

        double precision, intent(in) :: x  !Angle (degrees)
        double precision, intent(in) :: theta_bar  !Corrected roughness mean slope angle (degrees)
        double precision, intent(in) :: E1,E2,chi  !Parameters from the Hapke formalism evaluated at x

        double precision, intent(out) :: nu !Nu parameter from the Hapke formalism

        nu = chi*(dcos(x/180.d0*pi)+dsin(x/180.d0*pi)*dtan(theta_bar/180.d0*pi)*(E2)/(2.d0-E1))

        return

    end subroutine calc_nu

    !==================================================================================================
    subroutine calc_mueff(i,e,phi,theta_bar,E1e,E1i,E2e,E2i,chi,mueff)

        !Calculate the nu function from the Hapke formalism (Hapke 2012 p.333) 

        implicit none

        double precision, intent(in) :: i,e,phi  !Incident, reflection and azimuth angle (degrees)
        double precision, intent(in) :: theta_bar  !Corrected roughness mean slope angle (degrees)
        double precision, intent(in) :: E1e,E2e,E1i,E2i,chi  !Parameters from the Hapke formalism

        double precision :: irad,erad,phirad,tbarrad

        double precision, intent(out) :: mueff !Effective cos(e)

        !Calculating some initial parameters
        irad = i / 180. * pi  
        erad = e / 180. * pi
        phirad = phi / 180. * pi 
        tbarrad = theta_bar / 180. * pi

        if(i.le.e)then

            mueff = chi * ( dcos(erad) + dsin(erad) * dtan(tbarrad) * &
                (E2e - dsin(phirad/2.d0)**2.d0 * E2i) / (2.d0 - E1e - phirad/pi*E1i)  )

        elseif(i.gt.e)then

            mueff = chi * ( dcos(erad) + dsin(erad) * dtan(tbarrad) * &
                (dcos(phirad) * E2i + dsin(phirad/2.)**2. *E2e) / (2.0 - E1i - phirad/pi*E1e)  )

        endif

        return

    end subroutine calc_mueff

    !==================================================================================================
    subroutine calc_mu0eff(i,e,phi,theta_bar,E1e,E1i,E2e,E2i,chi,mu0eff)

        !Calculate the nu function from the Hapke formalism (Hapke 2012 p.333) 

        implicit none

        double precision, intent(in) :: i,e,phi  !Incident, reflection and azimuth angle (degrees)
        double precision, intent(in) :: theta_bar  !Corrected roughness mean slope angle (degrees)
        double precision, intent(in) :: E1e,E2e,E1i,E2i,chi  !Parameters from the Hapke formalism

        double precision :: irad,erad,phirad,tbarrad

        double precision, intent(out) :: mu0eff !Effective cos(i)

        !Calculating some initial parameters
        irad = i / 180. * pi  
        erad = e / 180. * pi
        phirad = phi / 180. * pi 
        tbarrad = theta_bar / 180. * pi

        if(i.le.e)then

            mu0eff = chi * ( dcos(irad) + dsin(irad) * dtan(tbarrad) * &
                (dcos(phirad) * E2e + dsin(phirad/2.d0)**2. *E2i) / (2.d0 - E1e - phirad/pi*E1i)  )

        elseif(i.gt.e)then

            mu0eff = chi * ( dcos(irad) + dsin(irad) * dtan(tbarrad) * &
                (E2i - dsin(phirad/2.d0)**2. *E2e) / (2.d0 - E1i - phirad/pi*E1e)  )

        endif

        return

    end subroutine calc_mu0eff

    !==================================================================================================
    subroutine calc_H(sglalb,x,r0,H)

        !Calculate the Ambartsumian–Chandrasekhar H function of the Hapke formalism (Hapke, 2012; p. 333)

        implicit none

        double precision, intent(in) :: sglalb !Single scattering albedo
        double precision, intent(in) :: x      !Value at which the H-function must be evaluated
        double precision, intent(in) :: r0     !Diffusive reflectance

        double precision, intent(out) :: H     !Ambartsumian–Chandrasekhar H function

        H = 1.d0/(1.d0-sglalb*x*(r0+(1.d0-2.d0*r0*x)/2.d0*dlog((1.d0+x)/x)) )

        return

    end subroutine calc_H

    !==================================================================================================
    subroutine calc_hgphase(G1,G2,F,g,phase)

        !Calculate the phase function given by a double Henyey-Greenstein function

        implicit none

        double precision, intent(in) :: G1,G2,F !Parameters defining the double H-G function
        double precision, intent(in) :: g       !Scattering angle (degrees)

        double precision :: t1,t2

        double precision, intent(out) :: phase     !Ambartsumian–Chandrasekhar H function

        t1 = (1.d0-G1**2.)/(1.d0 - 2.d0*G1*dcos(g/180.*pi) + G1**2.d0)**1.5
        t2 = (1.d0-G2**2.)/(1.d0 - 2.d0*G2*dcos(g/180.*pi) + G2**2.d0)**1.5

        phase = F * t1 + (1.d0 - F) * t2

        return

    end subroutine calc_hgphase


end module hapke