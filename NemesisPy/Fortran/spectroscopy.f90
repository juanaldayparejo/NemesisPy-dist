module spectroscopy

    integer, parameter :: maxg = 100

contains

    !==================================================================================================
    subroutine k_overlapg(nwave,ng,ngas,npoints,delg,k_gas,dkgasdT,f,k_combined,dk_combined)

        !This subroutine combines the absorption coefficient distributions of
        !several overlapping gases. The overlap is implicitly assumed to be random
        !and the k-distributions are assumed to have NG-1 mean values and NG-1
        !weights. Correspondingly there are NG ordinates in total.

        !Inputs
        !-------

        !nwave :: Number of wavelengths
        !ng :: Number of g-ordinates
        !del_g :: Intervals of g-ordinates
        !ngas :: Number of gases to combine
        !npoints :: Number of p-T points over to run the overlapping routine
        !k_gas(nwave,ng,ngas,npoints) :: K-distributions of the different gases
        !dkgasdT(nwave,ng,ngas,npoints) :: Rate of change of K-distributions of the different gases with temperature
        !f(ngas,npoints) :: Absorber amounts for each of the gases in each of the layers (cm-2)

        !Outputs
        !---------

        !k(nwave,ng,npoints) :: Combined k-distribution
        !dk(nwave,ng,npoints,ngas+1) :: Combined rate of change of k-distribution with the gas VMRs (1 to NGAS) and with temperature (NGAS+1)


        implicit none

        !Inputs
        integer, intent(in) :: nwave,ng,ngas,npoints
        double precision, intent(in) :: delg(ng)
        double precision, intent(in) :: k_gas(nwave,ng,npoints,ngas),dkgasdT(nwave,ng,npoints,ngas)
        double precision, intent(in) :: f(ngas,npoints)

        !Local
        integer :: ip,iwave,ig,igas
        double precision :: k_g_combined(ng),dk_g_combined(ng,ngas+1)

        !Outputs
        double precision, intent(out) :: k_combined(nwave,ng,npoints)
        double precision, intent(out) :: dk_combined(nwave,ng,npoints,ngas+1)


        do ip=1,npoints
            do iwave=1,nwave

                !Combining k-distributions
                call k_overlapg_gas(ng, ngas, k_gas(iwave,:,ip,:), dkgasdT(iwave,:,ip,:), f(:,ip), &
                                    delg, k_g_combined, dk_g_combined)

                do ig=1,ng
                    k_combined(iwave,ig,ip) = k_g_combined(ig)
                    do igas=1,ngas+1
                        dk_combined(iwave,ig,ip,igas) = dk_g_combined(ig,igas)
                    enddo
                enddo

            enddo
        enddo


        return

    end subroutine k_overlapg

    !==================================================================================================

    subroutine k_overlapg_gas(ng,ngas,k_gn,dkgndT,amount,delg,k_g,dkdq)

        !This subroutine combines the absorption coefficient distributions of
        !two overlapping gases. The overlap is implicitly assumed to be random
        !and the k-distributions are assumed to have NG-1 mean values and NG-1
        !weights. Correspondingly there are NG ordinates in total.

        !Inputs
        integer, intent(in) :: ng,ngas
        double precision, intent(in) :: k_gn(ng,ngas),dkgndT(ng,ngas)
        double precision, intent(in) :: amount(ngas)
        double precision, intent(in) :: delg(ng)

        !Local
        integer :: i,j,igas,nloop
        double precision :: k_g1(ng),k_g2(ng),dk1dT(ng),dk2dT(ng)
        double precision :: weight(ng*ng), contri(ng*ng), grad(ng*ng,ngas+1)
        double precision :: a1,a2


        !Outputs
        double precision, intent(out) :: k_g(ng)
        double precision, intent(out) :: dkdq(ng,ngas+1)

        if(ngas.le.1)then

            igas=1

            a1 = amount(igas)
            DO i=1,ng
                k_g1(i) = k_gn(i,igas)
                dk1dT(i) = dkgndT(i,igas)
            ENDDO

            do i=1,ng
                k_g(I) = k_g1(I)*a1
                dkdq(I,igas) = k_g1(I)
                dkdq(I,igas+1) = dk1dT(I)*a1
            enddo

        else

            DO igas=1,ngas-1

                !First pair of gases
                IF(igas.EQ.1)THEN

                    a1 = amount(igas)
                    a2 = amount(igas+1)
                    DO i=1,ng
                        k_g1(i) = k_gn(i,igas)
                        k_g2(i) = k_gn(i,igas+1)
                        dk1dT(i) = dkgndT(i,igas)
                        dk2dT(i) = dkgndT(i,igas+1)
                    ENDDO
            
                    !Skip if first k-distribution = 0.0
                    IF(k_g1(ng).EQ.0.0)THEN
                        DO I=1,ng
                            k_g(I) = k_g2(I)*a2
                            dkdq(I,igas) = 0.0
                            dkdq(I,igas+1) = k_g2(I)
                            dkdq(I,igas+2) = dk2dT(I)*a2
                        ENDDO
                        GOTO 99
                    ENDIF
            
                    !Skip if second k-distribution = 0.0
                    IF(k_g2(ng).EQ.0.0)THEN
                        DO I=1,ng
                            k_g(I) = k_g1(I)*a1
                            dkdq(I,igas) = k_g1(I)
                            dkdq(I,igas+1) = 0.0
                            dkdq(I,igas+2) = dk1dT(I)*a1
                        ENDDO
                        GOTO 99
                    ENDIF
            
                    jtest=ng*ng
            
                    nloop = 0
                    DO I=1,ng
                        DO J=1,ng
                            nloop = nloop + 1
                            weight(nloop) = delg(I)*delg(J)
                            contri(nloop) = k_g1(I)*a1 + k_g2(J)*a2
                            grad(nloop,igas) = k_g1(I)
                            grad(nloop,igas+1) = k_g2(J)
                            grad(nloop,igas+2) = dk1dT(I)*a1 + dk2dT(J)*a2
                        ENDDO
                    ENDDO

                !Subsequeant gases ... add amount*k to previous summed k.
                ELSE

                    a2 = amount(igas+1)
                    DO i=1,ng
                        k_g1(i) = k_g(i) 
                        k_g2(i) = k_gn(i,igas+1)
                        dk1dT(i) = dkdq(i,igas+1)           ! dK/dT of previous sum
                        dk2dT(i) = dkgndT(i,igas+1)         ! dK/dT of new dist
                    ENDDO
            
                    !Skip if first k-distribution = 0.0
                    IF(k_g1(ng).EQ.0.0)THEN
                        DO I=1,ng
                            k_g(I) = k_g2(I)*a2
                            DO jgas=1,igas
                                dkdq(I,jgas) = 0.0
                            ENDDO
                            dkdq(I,igas+1) = k_g2(I)
                            dkdq(I,igas+2) = dk2dT(I)*a2
                        ENDDO
                        GOTO 99
                    ENDIF
            
                    !Skip if second k-distribution = 0.0
                    IF(k_g2(ng).EQ.0.0)THEN
                        DO I=1,ng
                            k_g(I) = k_g1(I)
                            dkdq(I,igas+1) = 0.0
                            dkdq(I,igas+2) = dk1dT(I)
                        ENDDO
                        GOTO 99
                    ENDIF

                    nloop = 0
                    DO I=1,ng
                        DO J=1,ng
                            nloop = nloop + 1
                            weight(nloop) = delg(I)*delg(J)
                            contri(nloop) = k_g1(I) + k_g2(J)*a2
                            DO jgas = 1,igas
                                grad(nloop,jgas) = dkdq(I,jgas)
                            ENDDO
                            grad(nloop,igas+1) = k_g2(J)
                            grad(nloop,igas+2) = dk1dT(I) + dk2dT(J)*a2
                        ENDDO
                    ENDDO

                ENDIF
            
                call rankg(ngas,igas+2,delg,ng,nloop,weight,contri,grad,k_g,dkdq)

    99          CONTINUE
            
            ENDDO

        endif
        
        RETURN

    end subroutine k_overlapg_gas

    !==================================================================================================

    subroutine rankg(ngas,nparam,gw,ng,nloop,weight,cont,grad,k_g,dkdq)
        !***********************************************************************
        !_TILE:	RANKG.f
        !
        !_DESC:	Subroutine to sort randomised k-coefficients, and associated
        !	gradients into the mean k-distribution and gradient.
        !	
        !_ARGS:	Input variables:
        !	nparam			INTEGER	Number of gradients to consider
        !	gw(maxg)		REAL	Required weights of final k-dist.
        !	ng			INTEGER	Number of weights.
        !	nloop			INTEGER	Number of points in randomised
        !					k-distribution.
        !	weight(maxrank)		REAL	Weights of points in random k-dist
        !	cont(maxrank)		REAL	Random k-coeffs.
        !	grad(maxrank,maxgas+1)	REAL	Gradients of random k-coeffs.
        !
        !	Output variables
        !	k_g(maxg)		REAL	Mean k-dist.
        !	dkdq(maxg)		REAL	Mean gradients.
        !
        !_HIST:	23/2/96	Pat Irwin	Original
        !	2/3/96	A.L.Weir
        !	9/5/01	Pat Irwin
        !	31/7/01	Pat Irwin	Commented
        !	29/2/12	Pat Irwin	Updated for Radtrans2.0
        !***************************** VARIABLES *******************************
        
        implicit none

        !Inputs
        integer, intent(in) :: nparam,ng,nloop,ngas
        double precision :: gw(ng),weight(nloop),cont(nloop),grad(nloop,ngas+1)
        
        !Local
        integer :: i,ig,iparam,ico(nloop)
        double precision :: g_ord(ng+1),gdist(nloop),sum,frac,tmp(nloop)


        !Outputs
        double precision, intent(out) :: k_g(ng),dkdq(ng,ngas+1)

        !******************************** CODE *********************************
        
        !=======================================================================
        !
        !	Sum delta gs to get cumulative g ordinate. rank cont and weight
        !	arrays in ascending order of k (i.e. cont values).
        !
        !=======================================================================
        g_ord(1) = 0.0
        DO I=1,ng
            g_ord(I+1) = g_ord(I) + gw(I)
        ENDDO
        !     Make sure g_ord(ng+1)=1. (rounding errors can lead to numbers just
        !                               less than 1.0)

        if(g_ord(ng+1).lt.1.0)g_ord(ng+1)=1.
        
        DO I=1,nloop
            ico(I) = I
        ENDDO
        
        !Sort random k-coeffs into order. Integer array ico records which swaps
        !have been made so that we can also re-order the gradients.
        CALL sort2g(nloop, cont, ico)

        !Resort the weights:
        DO I=1,nloop
            tmp(I) = weight(ico(I))
        ENDDO
        DO I=1,nloop
            weight(I) = tmp(I)
        ENDDO

        !Resort the gradients
        DO iparam=1,nparam
            DO I=1,nloop
                tmp(I) = grad(ico(I),iparam)
            ENDDO
            DO I=1,nloop
                grad(I,iparam) = tmp(I)
            ENDDO
        ENDDO 

        !=======================================================================
        !
        !	Now form new g(k) and gradients by summing over weight. The new
        !       k(g) is then found by ascending the arranged ks and getting the
        !       weighted averages of the k values in the relevant g
        !       interval. Linear interpolation is used to deal with steps crossing
        !       over the required g intervals.
        !
        !=======================================================================
              
        !gdist(0) = 0.0
        gdist(1) = weight(1)
        DO I=2,nloop
            gdist(I) = weight(I) + gdist(I-1) 
        ENDDO
        
        DO I=1,ng
            k_g(I) = 0.0
            DO iparam=1,nparam
                dkdq(I,iparam) = 0.0
            ENDDO
        ENDDO

        ig = 1
        sum = 0.0
        DO I=1,nloop
            IF(gdist(I).LT.g_ord(ig+1))THEN
                k_g(ig) = k_g(ig) + cont(I) * weight(I)
                DO iparam=1,nparam
                    dkdq(ig,iparam) = dkdq(ig,iparam) + grad(I,iparam)*weight(I)
                ENDDO
                sum = sum + weight(I)
            ELSE
                frac = (g_ord(ig+1) - gdist(I-1))/(gdist(I) - gdist(I-1))
                k_g(ig) = k_g(ig) + sngl(frac)*cont(I)*weight(I)
                DO iparam=1,nparam
                    dkdq(ig,iparam) = dkdq(ig,iparam) + sngl(frac)*grad(I,iparam)*weight(I)
                ENDDO
                sum = sum + frac*weight(I)
                k_g(ig) = k_g(ig)/sngl(sum)
                DO iparam=1,nparam
                    dkdq(ig,iparam) = dkdq(ig,iparam)/sngl(sum)
                ENDDO
                ig = ig + 1
                sum = (1.0 - frac)*weight(I)
                k_g(ig) = sngl(1.0 - frac)*cont(I)*weight(I)
                DO iparam=1,nparam
                    dkdq(ig,iparam) = sngl(1.-frac)*grad(I,iparam)*weight(I)
                ENDDO
            ENDIF
        ENDDO

        IF(ig.EQ.ng)THEN
            k_g(ig) = k_g(ig)/sngl(sum)
            DO iparam=1,nparam
                dkdq(ig,iparam) = dkdq(ig,iparam)/sngl(sum)
            ENDDO
        ENDIF
        
        RETURN
        
    end subroutine rankg
    
    !==================================================================================================

    subroutine SORT2G(N,RA,IB)
        !******************************************************************
        !Modified numerical recipes routine to sort a vector RA of length N
        !into ascending order. Integer vector IB is initially set to
        !1,2,3,... and on output keeps a record of how RA has been sorted.
        !
        !Pat Irwin	31/7/01	Original version
        !Pat Irwin	29/2/12	Updated for Radtrans2.0
        ! 
        !******************************************************************

        implicit none

        integer, intent(in) :: N
        double precision :: RA(N)
        integer :: IB(N)

        integer :: L,IR,I,J,IRB
        double precision :: RRA
              
        L=N/2+1
        IR=N
10      CONTINUE
        IF(L.GT.1)THEN
            L=L-1
            RRA=RA(L)
            IRB=IB(L)
        ELSE
            RRA=RA(IR)
            IRB=IB(IR)
            RA(IR)=RA(1)
            IB(IR)=IB(1)
            IR=IR-1
            IF(IR.EQ.1)THEN
                RA(1)=RRA
                IB(1)=IRB
                RETURN
            ENDIF
        ENDIF
        I=L
        J=L+L
20      IF(J.LE.IR)THEN
            IF(J.LT.IR)THEN
                IF(RA(J).LT.RA(J+1))J=J+1
            ENDIF
            IF(RRA.LT.RA(J))THEN
                RA(I)=RA(J)
                IB(I)=IB(J)
                I=J
                J=J+J
            ELSE
                J=IR+1
            ENDIF
            GO TO 20
        ENDIF
        RA(I)=RRA
        IB(I)=IRB
        GO TO 10

    end subroutine
        


end module spectroscopy