      PROGRAM DSYTRF_PEAK
C     LAPACK blocked Bunch-Kaufman (DSYTRF) rate from the same OpenBLAS
C     MA57 links against. This is the honest ceiling for a *pivoted*
C     symmetric indefinite factorization built on tuned BLAS3 -- the
C     same algorithm class as feral's frontal factor, unlike DGEMM.
C     Counted in sum_{k<n}(n-k)^2 multiply-adds, the feral convention.
      IMPLICIT NONE
      INTEGER, PARAMETER :: NS = 7
      INTEGER :: SIZES(NS), I, J, K, N, R, REPS, INFO, LWORK
      DOUBLE PRECISION, ALLOCATABLE :: A(:,:), A0(:,:), WORK(:)
      INTEGER, ALLOCATABLE :: IPIV(:)
      INTEGER(8) :: T0, T1, RATE, BEST
      DOUBLE PRECISION :: MACS, US, MRATE
      DATA SIZES /32,64,96,128,192,256,512/
      CALL SYSTEM_CLOCK(COUNT_RATE=RATE)
      WRITE(*,'(A6,A14,A12,A12,A8)') "n","macs","min_us","MMac/s","info"
      DO I = 1, NS
        N = SIZES(I)
        LWORK = 64*N
        ALLOCATE(A(N,N), A0(N,N), IPIV(N), WORK(LWORK))
        DO J = 1, N
          DO K = 1, N
            IF (K .EQ. J) THEN
              A0(K,J) = DBLE(N) + 1.0D0
            ELSE
              A0(K,J) = DBLE(MOD(K*7+J*3, 17)) / 17.0D0
            END IF
          END DO
        END DO
        MACS = 0.0D0
        DO K = 0, N-1
          MACS = MACS + DBLE(N-K)*DBLE(N-K)
        END DO
        REPS = MAX(5, MIN(20000, INT(2.0D8 / MACS)))
        A = A0
        CALL DSYTRF('L', N, A, N, IPIV, WORK, LWORK, INFO)
        BEST = HUGE(BEST)
        DO R = 1, REPS
          A = A0
          CALL SYSTEM_CLOCK(T0)
          CALL DSYTRF('L', N, A, N, IPIV, WORK, LWORK, INFO)
          CALL SYSTEM_CLOCK(T1)
          IF (T1-T0 .LT. BEST) BEST = T1-T0
        END DO
        US = DBLE(BEST) * 1.0D6 / DBLE(RATE)
        MRATE = MACS / MAX(US, 1.0D-9)
        WRITE(*,'(I6,F14.0,F12.2,F12.0,I8)') N, MACS, US, MRATE, INFO
        DEALLOCATE(A,A0,IPIV,WORK)
      END DO
      END
