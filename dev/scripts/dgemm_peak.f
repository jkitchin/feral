      PROGRAM DGEMM_PEAK
C     Peak DGEMM rate for the OpenBLAS that MA57 links against, in
C     multiply-adds per microsecond (== MMac/s), so it is directly
C     comparable to feral's diag_153_dense_peak column.
      IMPLICIT NONE
      INTEGER, PARAMETER :: NS = 10
      INTEGER :: SIZES(NS), I, J, K, N, R, REPS
      DOUBLE PRECISION, ALLOCATABLE :: A(:,:), B(:,:), C(:,:)
      INTEGER(8) :: T0, T1, RATE, BEST
      DOUBLE PRECISION :: MACS, US, MRATE
      DATA SIZES /16,32,48,64,96,128,192,256,384,512/
      CALL SYSTEM_CLOCK(COUNT_RATE=RATE)
      WRITE(*,'(A6,A14,A12,A12)') "n", "macs", "min_us", "MMac/s"
      DO I = 1, NS
        N = SIZES(I)
        ALLOCATE(A(N,N), B(N,N), C(N,N))
        DO J = 1, N
          DO K = 1, N
            A(K,J) = DBLE(MOD(K*7+J*3, 17)) / 17.0D0
            B(K,J) = DBLE(MOD(K*5+J*11, 13)) / 13.0D0
            C(K,J) = 0.0D0
          END DO
        END DO
        MACS = DBLE(N) * DBLE(N) * DBLE(N)
        REPS = MAX(5, MIN(20000, INT(2.0D8 / MACS)))
        CALL DGEMM('N','N',N,N,N,1.0D0,A,N,B,N,0.0D0,C,N)
        BEST = HUGE(BEST)
        DO R = 1, REPS
          CALL SYSTEM_CLOCK(T0)
          CALL DGEMM('N','N',N,N,N,1.0D0,A,N,B,N,0.0D0,C,N)
          CALL SYSTEM_CLOCK(T1)
          IF (T1-T0 .LT. BEST) BEST = T1-T0
        END DO
        US = DBLE(BEST) * 1.0D6 / DBLE(RATE)
        MRATE = MACS / US
        WRITE(*,'(I6,F14.0,F12.2,F12.0)') N, MACS, US, MRATE
        DEALLOCATE(A,B,C)
      END DO
      END
