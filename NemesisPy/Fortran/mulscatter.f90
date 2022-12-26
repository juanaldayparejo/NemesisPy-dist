module mulscatter

    real, parameter :: pi = 3.1415926536
    integer, parameter :: IPOW0 = 16

contains

    !==================================================================================================
    subroutine calc_RTF_matrix(nwave,ng,nlay,nf,nmu,mu,wtmu,TAUTOT,OMEGAS,TAURAY,BNU,PPLPL,PPLMI,&
                                RL,TL,JL,iscl)

        !Subroutine to calculate the diffuse reflection, transmission and source matrices in a 
        !multiple scattering atmosphere

        implicit none

        !Inputs
        integer, intent(in) :: nwave,ng,nlay,nf,nmu   !Matrix sizes
        double precision, intent(in) :: mu(nmu),wtmu(nmu) 
        double precision, intent(in) :: TAUTOT(nwave,ng,nlay)  !Total optical depth
        double precision, intent(in) :: OMEGAS(nwave,ng,nlay)  !Single scattering albedo
        double precision, intent(in) :: TAURAY(nwave,nlay)     !Rayleigh scattering optical depth
        double precision, intent(in) :: BNU(nwave,nlay)        !Thermal emission of each layer
        double precision, intent(in) :: PPLPL(nwave,nlay,nf+1,nmu,nmu)  !Phase matrix in + direction (i.e. downwards)
        double precision, intent(in) :: PPLMI(nwave,nlay,nf+1,nmu,nmu)  !Phase matrix in - direction (i.e. upwards)


        !Local
        double precision :: ACOM(nmu,nmu),BCOM(nmu,nmu),CCOM(nmu,nmu)
        double precision :: MM(nmu,nmu),MMINV(nmu,nmu),CC(nmu,nmu),CCINV(nmu,nmu)
        double precision :: XFAC,TAUT,BC,OMEGA,TAUSCATOT,TAUSCAT,TAUR,TAUL,TEX,TAU0
        double precision :: DEL01,CON,E(nmu,nmu)
        double precision :: GPLMI(nmu,nmu), GPLPL(nmu,nmu)
        double precision :: R1(nmu,nmu),T1(nmu,nmu),J1(nmu,1)
        double precision :: RINIT(nmu,nmu),TINIT(nmu,nmu),JINIT(nmu,1)
        integer :: NN,N,I,J,ic,iwave,ig,ilay,imu,jmu


        !Outputs
        double precision, intent(out) :: RL(nwave,ng,nlay,nf+1,nmu,nmu)  !Diffuse reflection matrix
        double precision, intent(out) :: TL(nwave,ng,nlay,nf+1,nmu,nmu)  !Diffuse transmission matrix
        double precision, intent(out) :: JL(nwave,ng,nlay,nf+1,nmu,1)    !Diffuse source matrix
        integer, intent(out) :: iscl(nwave,ng,nlay)                      !Flag indicating if it is a scattering layer



        !Calculating constant matrices
        !---------------------------------

        DO J = 1,NMU
            DO I = 1,NMU
                E(I,J) = 0.0D0
                IF(I.EQ.J) E(I,J) = 1.0D0
                MM(I,J) = 0.0D0
                IF(I.EQ.J) MM(I,J) = MU(I)
                MMINV(I,J) = 0.0D0
                IF(I.EQ.J) MMINV(I,J) = 1.0D0/MU(I)
                CC(I,J) = 0.0D0
                IF(I.EQ.J) CC(I,J) = WTMU(I)
                CCINV(I,J) = 0.0D0
                IF(I.EQ.J) CCINV(I,J) = 1.0D0/WTMU(I)
            ENDDO
        ENDDO

        !MAIN LOOP
        !----------

        !Looping over Fourier expansion
        do ic=0,nf

            !Looping over wavelength
            do iwave=1,nwave
                !Looping over g-ordinate
                do ig=1,ng

                    !Looping over layer
                    do ilay=1,nlay

                        !Defining parameters
                        TAUT = 1.0D0*TAUTOT(iwave,ig,ilay)
                        BC = 1.0D0*BNU(iwave,ilay)
                        OMEGA = OMEGAS(iwave,ig,ilay)
                
                        !Calculating total scattering (aerosol + Rayleigh)
                        TAUSCATOT = TAUT*OMEGA
                        TAUR = TAURAY(iwave,ilay)

                        !Calculating only aerosol scattering
                        TAUSCAT = TAUSCATOT - TAUR

                        !print*,TAUT,OMEGA,TAUR
                        !pause

                        !Add in an error trap to counter single-double subtraction overflows 
                        IF(TAUSCAT.LT.0.)TAUSCAT=0.
                        IF(TAUT.LT.0.)TAUT=0.
                        IF(OMEGA.GT.1.0)THEN
                            OMEGA = 1.D0
                        ELSE IF(OMEGA.LT.0.0)THEN
                            OMEGA = 0.D0
                        ENDIF

                        !Calculating the matrices
                        IF(TAUT.EQ.0)THEN

                            do imu=1,nmu
                                do jmu=1,nmu
                                    RL(iwave,ig,ilay,ic+1,imu,jmu)=0.0
                                    TL(iwave,ig,ilay,ic+1,imu,jmu)=0.0
                                end do
                                JL(iwave,ig,ilay,ic+1,imu,1) =0.0
                                TL(iwave,ig,ilay,ic+1,imu,imu)=1.0
                            end do
                            iscl(iwave,ig,ilay) = 0

                        ELSE IF(OMEGA.EQ.0)THEN

                            do imu=1,nmu
                                do jmu=1,nmu
                                    RL(iwave,ig,ilay,ic+1,imu,jmu)=0.0
                                    TL(iwave,ig,ilay,ic+1,imu,jmu)=0.0
                                end do
                                TEX = -MMINV(imu,imu)*TAUT
                                if(TEX.GT.-200.0D0)THEN
                                    TL(iwave,ig,ilay,ic+1,imu,imu)=DEXP(TEX)
                                ELSE
                                    TL(iwave,ig,ilay,ic+1,imu,imu)=0.D0
                                ENDIF
                                JL(iwave,ig,ilay,ic+1,imu,1)= BC*(1.0 - TL(iwave,ig,ilay,ic+1,imu,imu))
                            end do
                            iscl(iwave,ig,ilay) = 0

                        ELSE

                            iscl(iwave,ig,ilay) = 1

                            !****************************************************************
                            !First calculate properties of initial very thin layer.
                            !******************************
                            !COMPUTATION OF GAMMA++	(Plass et al. 1973)
                            !GPLPL = MMINV*(E - CON*PPLPL*CC)
                            !******************************
                            CON=OMEGA*PI
                            
                            DEL01 = 0.0D0
                            IF(IC.EQ.0)DEL01 = 1.0D0
                            
                            CON=CON*(1.0D0 + DEL01)
                    
                            call MMUL(CON,NMU,NMU,NMU,PPLPL(iwave,ilay,ic+1,:,:),CC,ACOM)
                            call MADD(-1.0D0,NMU,NMU,E,ACOM,BCOM)
                            call MMUL(1.0D0,NMU,NMU,NMU,MMINV,BCOM,GPLPL)
                            

                            !******************************
                            !COMPUTATION OF GAMMA+-   
                            !GPLMI = MMINV*CON*PPLMI*CC
                            !******************************

                            call MMUL(CON,NMU,NMU,NMU,PPLMI(iwave,ilay,ic+1,:,:),CC,ACOM)
                            call MMUL(1.0D0,NMU,NMU,NMU,MMINV,ACOM,GPLMI)



                            !Define the layer in terms of its thickness
                            NN=INT(DLOG(TAUT)/DLOG(2.0D0))+IPOW0

                            IF(NN.GE.1)THEN
                             XFAC=1.0/(2.0D0**NN)
                            ELSE
                             XFAC=1.0
                            ENDIF
                            TAU0 = TAUT*XFAC


                            !**********************************************************************
                            !     COMPUTATION OF R, T AND J FOR INITIAL LAYER (SINGLE SCATTERING)
                            !**********************************************************************
                            call MADD(-TAU0,NMU,NMU,E,GPLPL,TINIT)
                            call MMUL(TAU0,NMU,NMU,NMU,E,GPLMI,RINIT)

                            DO J=1,NMU
                                J1(J,1)=0.0
                            ENDDO
                            IF(IC.EQ.0)THEN
                                DO J=1,NMU
                                    J1(J,1)=(1.0D0-OMEGA)*BC*TAU0*MMINV(J,J)
                                ENDDO
                            ENDIF

                            !If single scattering we fix the values
                            IF(NN.LT.1)THEN
                                
                                CALL MEQU(NMU,TINIT,TL(iwave,ig,ilay,ic+1,:,:))
                                CALL MEQU(NMU,RINIT,RL(iwave,ig,ilay,ic+1,:,:))
                                DO J=1,NMU
                                    JL(iwave,ig,ilay,ic+1,J,1)=J1(J,1)
                                ENDDO


                            !If multiple scattering we double the layer
                            ELSE
                                
                                CALL MEQU(NMU,TINIT,T1)
                                CALL MEQU(NMU,RINIT,R1)


                                !**************************************************************
                                !     COMPUTATION OF R AND T FOR SUBSEQUENT LAYERS: DOUBLING    
                                !**************************************************************
                                DO 100 N=1,NN

                                    call add_layer(NMU,E,R1,T1,J1,R1,T1,J1,&
                                        RL(iwave,ig,ilay,ic+1,:,:),TL(iwave,ig,ilay,ic+1,:,:),JL(iwave,ig,ilay,ic+1,:,:))
                                    TAUL=TAU0*(2.0D0**N)
                                
                                    IF(TAUL.EQ.TAUT) GOTO 100
                                    CALL MEQU(NMU,RL(iwave,ig,ilay,ic+1,:,:),R1)
                                    CALL MEQU(NMU,TL(iwave,ig,ilay,ic+1,:,:),T1)

                                    DO J=1,NMU
                                        J1(J,1)=JL(iwave,ig,ilay,ic+1,J,1)
                                    ENDDO

                                100 ENDDO

                            END IF

                        END IF

                    enddo
                enddo
            enddo
        enddo

    return


    end subroutine calc_RTF_matrix


    !==================================================================================================

    subroutine add_layer(NMU,E,R1,T1,J1,RSUB,TSUB,JSUB,RANS,TANS,JANS)
        !$Id: add.f,v 1.2 2011-06-17 15:57:52 irwin Exp $
        !*****************************************************************
        
        !Subroutine to add the diffuse reflection, transmission and reflection
        !matrices for two adjacent atmospheric layers
        
        !   Input variables:
        !	R1(NMU,NMU)	DOUBLE	Diffuse reflection operator for 1st layer
        !	T1(NMU,NMU)	DOUBLE	Diffuse transmission operator for 1st layer
        !	J1(NMU,1)	DOUBLE	Diffuse source function for 1st layer
        !	RSUB(NMU,NMU)	DOUBLE	Diffuse reflection operator for 2nd layer
        !	TSUB(NMU,NMU)	DOUBLE	Diffuse transmission operator for 2nd layer
        !	JSUB(NMU,1)	DOUBLE	Diffuse source function for 2nd layer
        !	NMU		INTEGER	Number of elements used
        
        !   Output variables
        !	RANS(NMU,NMU)	DOUBLE	Combined diffuse reflection operator
        !	TANS(NMU,NMU)	DOUBLE	Combined diffuse transmission operator
        !	JANS(NMU,1)	DOUBLE	Combined diffuse source function
        !
        !     Pat Irwin		17/9/96
        !
        !**********************************************************************
        
        implicit none

        !Inputs
        integer, intent(in) :: NMU
        double precision, intent(in) :: E(NMU,NMU)
        double precision, intent(in) :: R1(NMU,NMU),T1(NMU,NMU),J1(NMU,1)
        double precision, intent(in) :: RSUB(NMU,NMU),TSUB(NMU,NMU),JSUB(NMU,1)

        !Local
        double precision :: ACOM(NMU,NMU),BCOM(NMU,NMU),CCOM(NMU,NMU),JCOM(NMU,NMU)
        
        !Outputs
        double precision, intent(out) :: RANS(NMU,NMU),TANS(NMU,NMU),JANS(NMU,1)
        
        call MMUL(-1.0D0,NMU,NMU,NMU,RSUB,R1,BCOM)
        call MADD(1.0D0,NMU,NMU,E,BCOM,BCOM)
        call MATINV8(NMU,BCOM,ACOM)
        call MEQU(NMU,ACOM,BCOM)
        call MMUL(1.0D0,NMU,NMU,NMU,T1,BCOM,CCOM)
        call MMUL(1.0D0,NMU,NMU,NMU,CCOM,RSUB,RANS)
        call MMUL(1.0D0,NMU,NMU,NMU,RANS,T1,ACOM)
        call MADD(1.0D0,NMU,NMU,R1,ACOM,RANS)
        call MMUL(1.0D0,NMU,NMU,NMU,CCOM,TSUB,TANS)
        call MMUL(1.0D0,NMU,NMU,1,RSUB,J1,JCOM)
        call MADD(1.0D0,NMU,1,JSUB,JCOM,JCOM)
        call MMUL(1.0D0,NMU,NMU,1,CCOM,JCOM,JANS)
        call MADD(1.0D0,NMU,1,J1,JANS,JANS)

        RETURN

    end subroutine

    !==================================================================================================

    subroutine addp_layer(NMU,E,R1,T1,J1,ISCAT1,RSUB,TSUB,JSUB,RANS,TANS,JANS)
        !$Id: addp.f,v 1.2 2011-06-17 15:57:52 irwin Exp $
        !*****************************************************************
        !Subroutine to add the diffuse reflection, transmission and reflection
        !matrices for two adjacent atmospheric layers
   
        !Input variables:
        !R1(JDIM,JDIM)	DOUBLE	Diffuse reflection operator for 1st layer
        !T1(JDIM,JDIM)	DOUBLE	Diffuse transmission operator for 1st layer
        !J1(JDIM,1)	DOUBLE	Diffuse source function for 1st layer
        !ISCAT1		INTEGER Flag to indicate if 2nd layer is scattering
        !RSUB(JDIM,JDIM)	DOUBLE	Diffuse reflection operator for 2nd layer
        !TSUB(JDIM,JDIM)	DOUBLE	Diffuse transmission operator for 2nd layer
        !JSUB(JDIM,1)	DOUBLE	Diffuse source function for 2nd layer
        !NMU		INTEGER	Number of elements used
   
        !Output variables
        !RANS(JDIM,JDIM)	DOUBLE	Combined diffuse reflection operator
        !TANS(JDIM,JDIM)	DOUBLE	Combined diffuse transmission operator
        !JANS(JDIM,1)	DOUBLE	Combined diffuse source function
   
        !Pat Irwin		17/9/96
        !**********************************************************************
   
        implicit none

        !Inputs
        integer, intent(in) :: NMU,ISCAT1
        double precision, intent(in) :: E(NMU,NMU)
        double precision, intent(in) :: R1(NMU,NMU),T1(NMU,NMU),J1(NMU,1)
        double precision, intent(in) :: RSUB(NMU,NMU),TSUB(NMU,NMU),JSUB(NMU,1)

        !Local
        double precision :: ACOM(NMU,NMU),BCOM(NMU,NMU),CCOM(NMU,NMU),JCOM(NMU,NMU)
        double precision :: TA, TB
        integer :: I,J

        !Outputs
        double precision, intent(out) :: RANS(NMU,NMU),TANS(NMU,NMU),JANS(NMU,1)


        !Subroutine solves Eq. 7b,8b,9b of Plass et al. (1973) 
        !Here :
            !R1 is R10 for the homogenous layer 1 being added (and thus equal to R01)
            !T1 is T10 for the homogenous layer 1 being added (and thus equal to T10)
            !J1 is the source function for the homegenous layer 1 (and thus same in +ve and -ve directions)
            !   +ve is going in direction from layer 1 to 2nd layer
            !   -ve is going in direction from 2nd layer to layer 1
            !RSUB is R12 for the 2nd layer (homegenous or composite)
            !TSUB is T21 for the 2nd layer (homegenous or composite)
            !JSUB is source function for 2nd layer (homegenous or composite) going in -ve direction (i.e. JM21)
            !RANS is R02 for combined layers
            !TANS is T20 for combineds layers
            !JANS is JM20 for combined layers (i.e. in -ve direction)
   
        IF(ISCAT1.EQ.1)THEN
            !2nd layer is scattering. Solve Eq. 7b,8b,9b of Plass et al. (1973)

            call MMUL(-1.0D0,NMU,NMU,NMU,RSUB,R1,BCOM) !BCOM=-RSUB*R1
            call MADD(1.0D0,NMU,NMU,E,BCOM,BCOM)  !BCOM=E-RSUB*R1
            call MATINV8(NMU,BCOM,ACOM) !ACOM = INV(E-RSUB*R1)
            call MEQU(NMU,ACOM,BCOM) !BCOM=INV(E-RSUB*R1)
            call MMUL(1.0D0,NMU,NMU,NMU,T1,BCOM,CCOM) !CCOM=T1*INV(E-RSUB*R1)
            call MMUL(1.0D0,NMU,NMU,NMU,CCOM,RSUB,RANS) !RANS=T1*INV(E-RSUB*R1)*RSUB
            call MMUL(1.0D0,NMU,NMU,NMU,RANS,T1,ACOM) !ACOM=T1*INV(E-RSUB*R1)*RSUB*T1
            call MADD(1.0D0,NMU,NMU,R1,ACOM,RANS) !RANS = R1+T1*INV(E-RSUB*R1)*RSUB*T1
            call MMUL(1.0D0,NMU,NMU,NMU,CCOM,TSUB,TANS) !TANS=T1*INV(E-RSUB*R1)*TSUB
            call MMUL(1.0D0,NMU,NMU,1,RSUB,J1,JCOM)  !JCOM=RSUB*J1
            call MADD(1.0D0,NMU,1,JSUB,JCOM,JCOM) !JCOM=JSUB+RSUB*J1
            call MMUL(1.0D0,NMU,NMU,1,CCOM,JCOM,JANS) !JANS=T1*INV(E-RSUB*R1)*(JSUB+RSUB*J1)
            call MADD(1.0D0,NMU,1,J1,JANS,JANS) !JANS = J1+T1*INV(E-RSUB*R1)*(JSUB+RSUB*J1) 

        ELSE
            !2nd layer is non-scattering
            call MMUL(1.0D0,NMU,NMU,1,RSUB,J1,JCOM)
            call MADD(1.0D0,NMU,1,JSUB,JCOM,JCOM)
   
            DO I=1,NMU
                TA = T1(I,I)
                DO J=1,NMU
                    TB = T1(J,J)
                    TANS(I,J) = TSUB(I,J)*TA
                    RANS(I,J) = RSUB(I,J)*TA*TB
                END DO
                JANS(I,1) = J1(I,1) + TA*JCOM(I,1)
            END DO
   
        ENDIF
   
   
        RETURN
    end subroutine
   

    !==================================================================================================

    subroutine MEQU(N,AMAT2,AMAT1)
        !$Id: matrices.f,v 1.2 2011-06-17 15:57:53 irwin Exp $
        !*************************************************************
        !Subroutine to transfer an NxN matrix contained in AMAT2 into AMAT1
        !*************************************************************

        implicit none

        integer,intent(in) :: N
        double precision, intent(in) :: AMAT2(N,N)
        double precision, intent(out) :: AMAT1(N,N)

        integer :: I,J

        DO J=1,N
            DO I=1,N
                AMAT1(I,J)=AMAT2(I,J)
            ENDDO
        ENDDO

        RETURN

    end subroutine
        
    !==================================================================================================

    subroutine MADD(CONST,N1,N2,AM1,AM2,ANS)
        !*************************************************************
        !Subroutine to add one matrix to the multiple of another:
        !   ANS = AM1 + CONST*AM2
        
        implicit none

        integer, intent(in) :: N1,N2
        double precision, intent(in) :: AM1(N1,N2),AM2(N1,N2)
        double precision, intent(in) :: CONST

        integer :: I,J

        double precision, intent(out) :: ANS(N1,N2)

        DO J=1,N2
            DO I=1,N1
                ANS(I,J)=AM1(I,J)+CONST*AM2(I,J)
            ENDDO
        ENDDO


        RETURN

    end subroutine
        
    !==================================================================================================

    subroutine MMUL(CONST,N1,N2,N3,AM1,AM2,ANS)

        !*************************************************************
        !Subroutine to multiply two  matrices together
        !   ANS = CONST*AM1*AM2

        implicit none

        integer, intent(in) :: N1,N2,N3
        double precision, intent(in) :: CONST
        double precision, intent(in) :: AM1(N1,N2), AM2(N2,N3)

        double precision :: AIJ
        integer :: I,J,K

        double precision, intent(out) :: ANS(N1,N3)

        DO J=1,N3
            DO I=1,N1
                AIJ = 0.0D0
                DO K=1,N2
                    AIJ = AIJ + AM1(I,K)*AM2(K,J)
                ENDDO
                ANS(I,J)=CONST*AIJ
            ENDDO
        ENDDO

        RETURN

    end subroutine
        

    !==================================================================================================

	subroutine MATINV8(N,A,AINV)
        !-----------------------------------------------------------------------
        !TITLE: MATINV8: matrix inversion routine
        !
        !_ARGS:  A id NxN matrix stored in a NDIMxNDIM array. AINV id also a NDIMxNDIM
        !        array.
        !
        !_KEYS:
        !
        !-----------------------------------------------------------------------

        implicit none

        !Inputs
        integer, intent(in) :: N
        double precision, intent(in) :: A(N,N)

        !Local
        integer :: I,J,INDX(N)
        double precision :: D

        !Outputs
        double precision, intent(out) :: AINV(N,N)

        !Initialising matrix
        DO J=1,N
            DO I=1,N
                AINV(I,J)=0.D0
            ENDDO
            AINV(J,J)=1.D0
        ENDDO

        CALL LUDCMP8(N,A,INDX,D)
        DO J=1,N
            call LUBKSB8(N,A,INDX,AINV(1,J))
        ENDDO
            
        RETURN

    end subroutine
             
    !==================================================================================================
    
    subroutine LUDCMP8(N,A,INDX,D)
        !  3/5/88 ...LWK... FROM [ATMRJW.RECIPES], CONVERTED TO DOUBLE PRECISION

        implicit none

        !Inputs
        integer, intent(in) :: N
        double precision :: A(N,N)

        !Local
        integer :: I,J,K,IMAX
        integer, parameter :: NMAX = 100
        double precision :: VV(NMAX),AAMAX,SUM,DUM
        double precision, parameter :: TINY = 1.0D-20

        !Outputs
        double precision, intent(out) :: D
        integer, intent(out) :: INDX(N)

        IF(N.GT.NMAX)THEN
            PRINT*,'Error in matinv8:ludcmp8. N>NMAX'
            PRINT*,N,NMAX
            STOP 
        ENDIF

        D=1.0D0
        DO 12 I=1,N
            AAMAX=0.D0
            DO J=1,N
                IF (ABS(A(I,J)).GT.AAMAX) AAMAX=ABS(A(I,J))
            ENDDO
            IF (AAMAX.EQ.0.) THEN
                PRINT*, 'Singular matrix.'
            STOP
            ENDIF
            VV(I)=1.D0/AAMAX
12      CONTINUE

        DO 19 J=1,N
            IF (J.GT.1) THEN
                DO I=1,J-1
                    SUM=A(I,J)
                    IF (I.GT.1)THEN
                        DO K=1,I-1
                        SUM=SUM-A(I,K)*A(K,J)
                        ENDDO
                        A(I,J)=SUM
                    ENDIF
                ENDDO   
            ENDIF
            AAMAX=0.D0
            DO I=J,N
                SUM=A(I,J)
                IF (J.GT.1)THEN
                    DO K=1,J-1
                        SUM=SUM-A(I,K)*A(K,J)
                    ENDDO
                    A(I,J)=SUM
                ENDIF
                DUM=VV(I)*ABS(SUM)
                IF (DUM.GE.AAMAX) THEN
                    IMAX=I
                    AAMAX=DUM
                ENDIF
            ENDDO
            IF (J.NE.IMAX)THEN
                DO K=1,N
                    DUM=A(IMAX,K)
                    A(IMAX,K)=A(J,K)
                    A(J,K)=DUM
                ENDDO
                D=-D
                VV(IMAX)=VV(J)
            ENDIF
            INDX(J)=IMAX
            IF(J.NE.N)THEN
                IF(A(J,J).EQ.0.)A(J,J)=TINY
                DUM=1.D0/A(J,J)
                DO I=J+1,N
                    A(I,J)=A(I,J)*DUM
                ENDDO
            ENDIF
19      CONTINUE

        IF(A(N,N).EQ.0.D0)A(N,N)=TINY
        RETURN
    end subroutine

    !==================================================================================================

    subroutine LUBKSB8(N,A,INDX,B)
        !3/5/88 ...LWK...  FROM [ATMRJW.RECIPES], CONVERTED TO DOUBLE PRECISION
              
        implicit none

        !Inputs
        integer, intent(in) :: N
        double precision, intent(in) :: A(N,N)
        integer, intent(in) :: INDX(N)

        !Local
        integer :: I,J,II,LL
        double precision :: SUM

        !Outputs
        double precision, intent(out) :: B(N)

        II=0
        DO I=1,N
            LL=INDX(I)
            SUM=B(LL)
            B(LL)=B(I)
            IF (II.NE.0)THEN
                DO J=II,I-1
                    SUM=SUM-A(I,J)*B(J)
                ENDDO
            ELSE IF (SUM.NE.0.) THEN
                II=I
            ENDIF
            B(I)=SUM
        ENDDO
        DO I=N,1,-1
            SUM=B(I)
            IF(I.LT.N)THEN
                DO J=I+1,N
                    SUM=SUM-A(I,J)*B(J)
                ENDDO
            ENDIF
            B(I)=SUM/A(I,I)
        ENDDO

        RETURN
    end subroutine
        


end module mulscatter