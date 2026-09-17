C     MA57 analysis-only probe for feral issue #203.
C
C     Reads a symmetric Matrix Market file (lower triangle, coordinate
C     real), runs MA57ID + MA57AD (analysis only, no numeric factor),
C     and prints the forecast factor size and elimination flop count:
C
C       INFO(5)  forecast number of reals to hold the factors
C       INFO(6)  forecast number of integers to hold the factors
C       INFO(12) forecast maximum front size
C       RINFO(1) forecast number of flops for the elimination
C
C     Only the pattern matters for these, so the unit values written by
C     kkt_pattern_repro.py are fine.
C
C     Usage: ma57_analyse MATRIX.mtx [ICNTL6]
C       ICNTL6 selects the MA57 ordering: 1 = user perm, 2 = AMD/MC47,
C       3 = MA27 min degree, 4 = METIS, 5 = automatic (default).

      PROGRAM MA57_ANALYSE
      IMPLICIT NONE
      INTEGER, PARAMETER :: MAXNZ = 20000000
      INTEGER, ALLOCATABLE :: IRN(:), JCN(:), KEEP(:), IWORK(:), PERM(:)
      INTEGER :: ICNTL(20), INFO(40)
      DOUBLE PRECISION :: CNTL(5), RINFO(20)
      INTEGER :: N, M, NZ, I, J, K, IOS, LKEEP, ORD
      DOUBLE PRECISION :: V
      CHARACTER(LEN=512) :: PATH, ARG2
      CHARACTER(LEN=1024) :: LINE

      IF (COMMAND_ARGUMENT_COUNT() .LT. 1) THEN
        WRITE(0,*) "usage: ma57_analyse MATRIX.mtx [ICNTL6]"
        STOP 2
      END IF
      CALL GET_COMMAND_ARGUMENT(1, PATH)
      ORD = 5
      IF (COMMAND_ARGUMENT_COUNT() .GE. 2) THEN
        CALL GET_COMMAND_ARGUMENT(2, ARG2)
        READ(ARG2,*) ORD
      END IF

      OPEN(UNIT=10, FILE=TRIM(PATH), STATUS='OLD', ACTION='READ',
     &     IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "cannot open ", TRIM(PATH)
        STOP 1
      END IF
C     Skip banner and comment lines.
   10 READ(10,'(A)',IOSTAT=IOS) LINE
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "unexpected eof in header"
        STOP 1
      END IF
      IF (LINE(1:1) .EQ. '%') GOTO 10
      READ(LINE,*) N, M, NZ
      IF (NZ .GT. MAXNZ) THEN
        WRITE(0,*) "nnz too large: ", NZ
        STOP 1
      END IF

      ALLOCATE(IRN(NZ), JCN(NZ))
      DO K = 1, NZ
        READ(10,*,IOSTAT=IOS) I, J, V
        IF (IOS .NE. 0) THEN
          WRITE(0,*) "read failed at entry ", K
          STOP 1
        END IF
        IRN(K) = I
        JCN(K) = J
      END DO
      CLOSE(10)

      LKEEP = 5*N + NZ + MAX(N, NZ) + 42
      ALLOCATE(KEEP(LKEEP), IWORK(5*N), PERM(N))
      DO I = 1, N
        PERM(I) = I
      END DO

      CALL MA57ID(CNTL, ICNTL)
      ICNTL(5) = 0
      ICNTL(6) = ORD
      IF (ORD .EQ. 1) THEN
        DO I = 1, N
          KEEP(I) = PERM(I)
        END DO
      END IF
      CALL MA57AD(N, NZ, IRN, JCN, LKEEP, KEEP, IWORK, ICNTL, INFO,
     &            RINFO)
      IF (INFO(1) .LT. 0) THEN
        WRITE(0,*) "ma57ad failed INFO(1)=", INFO(1)
        STOP 1
      END IF

      WRITE(*,'(A,I0)')    "n ", N
      WRITE(*,'(A,I0)')    "nnz ", NZ
      WRITE(*,'(A,I0)')    "icntl6 ", ORD
      WRITE(*,'(A,I0)')    "ordering_used ", INFO(36)
      WRITE(*,'(A,I0)')    "real_forecast ", INFO(5)
      WRITE(*,'(A,I0)')    "int_forecast ", INFO(6)
      WRITE(*,'(A,I0)')    "maxfront ", INFO(7)
      WRITE(*,'(A,I0)')    "treenodes ", INFO(8)
      WRITE(*,'(A,ES14.6)') "flops_asm ", RINFO(1)
      WRITE(*,'(A,ES14.6)') "flops_elim ", RINFO(2)

C     Dump KEEP(1:N), the pivot order chosen by the analysis, so the
C     same ordering can be replayed through another solver.
      IF (COMMAND_ARGUMENT_COUNT() .GE. 3) THEN
        CALL GET_COMMAND_ARGUMENT(3, ARG2)
        OPEN(UNIT=11, FILE=TRIM(ARG2), STATUS='REPLACE', ACTION='WRITE')
        DO I = 1, N
          WRITE(11,'(I0)') KEEP(I)
        END DO
        CLOSE(11)
      END IF
      END
