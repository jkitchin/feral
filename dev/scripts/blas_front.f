      PROGRAM BLAS_FRONT
C     What does a BLAS-based partial frontal LDL^T cost on the exact
C     (nrow, ncol) shapes feral's corpus produces?  This is the kernel
C     MA57 runs: factor the leading ncol x ncol block with pivoting
C     (DSYTRF), solve the off-diagonal block (DTRSM + D scaling), then
C     one DSYRK Schur update of the (nrow-ncol) trailing block.
C     Counted in sum_{k<ncol}(nrow-k)^2 multiply-adds -- feral's model.
      IMPLICIT NONE
      INTEGER, PARAMETER :: NSH = 10
      INTEGER :: SHR(NSH), SHC(NSH), I, J, K, NR, NC, R, REPS, INFO
      INTEGER :: LWORK, M
      DOUBLE PRECISION, ALLOCATABLE :: A(:,:), A0(:,:), WORK(:), D(:)
      INTEGER, ALLOCATABLE :: IPIV(:)
      INTEGER(8) :: T0, T1, RATE, BEST
      DOUBLE PRECISION :: MACS, US, MRATE
C     nrow / ncol pairs taken from diag_153_kernel_headroom "top shapes"
      DATA SHR /3,4,11,18,36,44,56,76,345,365/
      DATA SHC /1,2, 3,16,16,16,16,16, 16, 22/
      CALL SYSTEM_CLOCK(COUNT_RATE=RATE)
      WRITE(*,'(A6,A6,A14,A12,A12)') "nrow","ncol","macs","min_us",
     &     "MMac/s"
      DO I = 1, NSH
        NR = SHR(I)
        NC = SHC(I)
        M  = NR - NC
        LWORK = 64*NR
        ALLOCATE(A(NR,NR), A0(NR,NR), IPIV(NR), WORK(LWORK), D(NR))
        DO J = 1, NR
          DO K = 1, NR
            IF (K .EQ. J) THEN
              A0(K,J) = DBLE(NR) + 1.0D0
            ELSE
              A0(K,J) = DBLE(MOD(K*7+J*3, 17)) / 17.0D0
            END IF
          END DO
        END DO
        MACS = 0.0D0
        DO K = 0, NC-1
          MACS = MACS + DBLE(NR-K)*DBLE(NR-K)
        END DO
        REPS = MAX(200, MIN(200000, INT(2.0D8 / MACS)))
C       Batch-time REPS iterations: SYSTEM_CLOCK has 1 us granularity,
C       which is larger than a whole small front.  A copy-only loop of
C       the same length is timed and subtracted so the A = A0 restore
C       is not charged to the kernel.
        A = A0
        CALL DSYTRF('L', NC, A, NR, IPIV, WORK, LWORK, INFO)
        CALL SYSTEM_CLOCK(T0)
        DO R = 1, REPS
          A = A0
          CALL DSYTRF('L', NC, A, NR, IPIV, WORK, LWORK, INFO)
          IF (M .GT. 0) THEN
            CALL DTRSM('R','L','T','U', M, NC, 1.0D0, A, NR,
     &                 A(NC+1,1), NR)
            DO J = 1, NC
              D(J) = 1.0D0 / A(J,J)
            END DO
            DO J = 1, NC
              DO K = 1, M
                A(NC+K,J) = A(NC+K,J) * D(J)
              END DO
            END DO
            CALL DSYRK('L','N', M, NC, -1.0D0, A(NC+1,1), NR,
     &                 1.0D0, A(NC+1,NC+1), NR)
          END IF
        END DO
        CALL SYSTEM_CLOCK(T1)
        BEST = T1 - T0
        CALL SYSTEM_CLOCK(T0)
        DO R = 1, REPS
          A = A0
          CALL DUMMY(A, NR)
        END DO
        CALL SYSTEM_CLOCK(T1)
        BEST = BEST - (T1 - T0)
        IF (BEST .LT. 1) BEST = 1
        US = DBLE(BEST) * 1.0D6 / DBLE(RATE) / DBLE(REPS)
        MRATE = MACS / MAX(US, 1.0D-9)
        WRITE(*,'(I6,I6,F14.0,F12.2,F12.0)') NR, NC, MACS, US, MRATE
        DEALLOCATE(A,A0,IPIV,WORK,D)
      END DO
      END

      SUBROUTINE DUMMY(A, N)
C     Opaque sink so the copy-only calibration loop is not elided.
      IMPLICIT NONE
      INTEGER N
      DOUBLE PRECISION A(N,N)
      IF (A(1,1) .EQ. -12345.0D0) A(1,1) = 0.0D0
      RETURN
      END
