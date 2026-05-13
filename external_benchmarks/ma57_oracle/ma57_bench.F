C
C   ma57_bench.F — HSL MA57 oracle for the feral comparison benchmark.
C
C   Reads a manifest file of (mtx_path, rhs_path, out_path) triples.
C   For each triple, factors the matrix with MA57AD/BD and solves
C   with MA57DD (iterative refinement enabled), then writes a
C   plain-text sidecar mirroring the schema used by mumps_bench.F
C   and hsl_bench.c.
C
C   Configuration (matches the comparison harness "best-effort" mode):
C     * ICNTL(6) = 5  — automatic AMD / METIS ordering
C     * ICNTL(7) = 1  — numerical pivoting with threshold (BK-style)
C     * ICNTL(15) = 1 — automatic scaling on (MC64-equivalent)
C     * CNTL(1)  = 1e-8 — pivot threshold (matches MA27 / Ipopt default)
C     * MA57DD JOB=2 with ICNTL(9)=10 — residual-based iterative
C       refinement, up to 10 steps. Mirrors what MUMPS + ICNTL(10)=2
C       and MA97 + Richardson loop provide; gives an apples-to-apples
C       accuracy comparison.
C
C   Inertia:
C     * INFO(24) — number of negative eigenvalues
C     * INFO(25) — rank of the matrix; zero count = N - INFO(25)
C     * positive count = N - neg - zero
C
C   Usage:  ./ma57_bench <manifest.txt>
C
C   Manifest format (one matrix per line):
C       /abs/path/to/MAT.mtx /abs/path/MAT.rhs.txt /abs/path/out.txt
C
C   Output file format (one key per line):
C       solver ma57-2023.11.17
C       n <int>
C       nnz <int>
C       inertia_pos <int>
C       inertia_neg <int>
C       inertia_zero <int>
C       factor_us <int>
C       solve_us <int>
C       rel_res <f64>
C       info1 <int>
C       info24 <int>     (number of 2x2 pivots)
C       info25 <int>     (rank)
C       status ok|fail
C
      PROGRAM MA57_BENCH
      IMPLICIT NONE

      INTERFACE
        SUBROUTINE READ_MTX(PATH, NR, NC, NNZ, IRN, JCN, VALS, IOS)
          CHARACTER(LEN=*), INTENT(IN) :: PATH
          INTEGER, INTENT(OUT) :: NR, NC, NNZ
          INTEGER, ALLOCATABLE, INTENT(OUT) :: IRN(:), JCN(:)
          DOUBLE PRECISION, ALLOCATABLE, INTENT(OUT) :: VALS(:)
          INTEGER, INTENT(OUT) :: IOS
        END SUBROUTINE READ_MTX
        SUBROUTINE PARSE_LINE(LINE, P1, P2, P3)
          CHARACTER(LEN=*), INTENT(IN) :: LINE
          CHARACTER(LEN=*), INTENT(OUT) :: P1, P2, P3
        END SUBROUTINE PARSE_LINE
      END INTERFACE

      CHARACTER(LEN=4096) :: MANIFEST_PATH, LINE
      CHARACTER(LEN=4096) :: MTX_PATH, RHS_PATH, OUT_PATH
      INTEGER :: IOS, NMAT, NDONE
      INTEGER :: I, MROWS, MCOLS, MNNZ
      DOUBLE PRECISION :: VAL
      INTEGER :: ROW, COL

      INTEGER, ALLOCATABLE :: IRA(:), JCA(:)
      DOUBLE PRECISION, ALLOCATABLE :: AA(:), BB(:), XX(:), RESBUF(:)
      DOUBLE PRECISION, ALLOCATABLE :: AXBUF(:)

C     MA57 workspace
      INTEGER :: LKEEP, LFACT, LIFACT, LWORK
      INTEGER, ALLOCATABLE :: KEEP(:), IFACT(:), IWORK(:)
      DOUBLE PRECISION, ALLOCATABLE :: FACT(:), WORK(:)
      INTEGER :: ICNTL(20), INFO(40)
      DOUBLE PRECISION :: CNTL(5), RINFO(20)
      INTEGER :: JOB, INERT_POS, INERT_NEG, INERT_ZERO

      DOUBLE PRECISION :: RES_NORM, RHS_NORM, REL_RES
      INTEGER(8) :: T_FAC_START, T_FAC_END, T_SOL_START, T_SOL_END
      INTEGER(8) :: CLOCK_RATE, CLOCK_MAX
      INTEGER(8) :: FAC_US, SOL_US
      LOGICAL :: HAVE_FAILURE

      IF (COMMAND_ARGUMENT_COUNT() .LT. 1) THEN
        WRITE(0,*) "Usage: ma57_bench <manifest.txt>"
        STOP 2
      END IF
      CALL GET_COMMAND_ARGUMENT(1, MANIFEST_PATH)

      CALL SYSTEM_CLOCK(COUNT_RATE=CLOCK_RATE, COUNT_MAX=CLOCK_MAX)

      OPEN(UNIT=10, FILE=TRIM(MANIFEST_PATH), STATUS='OLD',
     &     ACTION='READ', IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot open manifest: ", TRIM(MANIFEST_PATH)
        STOP 4
      END IF

      NMAT = 0
      NDONE = 0
 100  CONTINUE
      READ(10, '(A)', IOSTAT=IOS) LINE
      IF (IOS .NE. 0) GOTO 999
      IF (LEN_TRIM(LINE) .EQ. 0) GOTO 100

      NMAT = NMAT + 1
      CALL PARSE_LINE(LINE, MTX_PATH, RHS_PATH, OUT_PATH)

      HAVE_FAILURE = .FALSE.

C     ---- Read MTX ----
      CALL READ_MTX(MTX_PATH, MROWS, MCOLS, MNNZ, IRA, JCA, AA, IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "READ_MTX failed: ", TRIM(MTX_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 800
      END IF

C     ---- Read RHS ----
      ALLOCATE(BB(MROWS))
      OPEN(UNIT=11, FILE=TRIM(RHS_PATH), STATUS='OLD',
     &     ACTION='READ', IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot open rhs: ", TRIM(RHS_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 700
      END IF
      DO I = 1, MROWS
        READ(11, *, IOSTAT=IOS) BB(I)
        IF (IOS .NE. 0) THEN
          WRITE(0,*) "rhs read failed at row ", I
          HAVE_FAILURE = .TRUE.
          CLOSE(11)
          GOTO 700
        END IF
      END DO
      CLOSE(11)

C     ---- MA57 init: defaults, then override with "best-effort" cfg ----
      CALL MA57ID(CNTL, ICNTL)
      ICNTL(1) = 0
      ICNTL(2) = 0
      ICNTL(3) = 0
      ICNTL(4) = 0
      ICNTL(5) = 0
      ICNTL(6) = 5
      ICNTL(7) = 1
      ICNTL(9) = 1
      ICNTL(15) = 1
      CNTL(1)  = 1.0D-8

C     ---- Analysis ----
      LKEEP = 5 * MROWS + MNNZ + MAX(MROWS, MNNZ) + 42
      ALLOCATE(KEEP(LKEEP))
      ALLOCATE(IWORK(5 * MROWS))

      CALL MA57AD(MROWS, MNNZ, IRA, JCA, LKEEP, KEEP, IWORK,
     &            ICNTL, INFO, RINFO)
      IF (INFO(1) .LT. 0) THEN
        WRITE(0,'(A,A,A,I6)') "ma57ad failed on ", TRIM(MTX_PATH),
     &        " INFO(1)=", INFO(1)
        HAVE_FAILURE = .TRUE.
        GOTO 650
      END IF

C     INFO(9)  = suggested real workspace length (LFACT)
C     INFO(10) = suggested integer workspace length (LIFACT)
C     Pad by 3x for safety against delayed pivots on tough KKT systems.
      LFACT  = INT(DBLE(INFO(9))  * 3.0D0) + 100
      LIFACT = INT(DBLE(INFO(10)) * 3.0D0) + 100
      ALLOCATE(FACT(LFACT))
      ALLOCATE(IFACT(LIFACT))

C     ---- Factorization (with workspace-grow retry loop) ----
      CALL SYSTEM_CLOCK(T_FAC_START)
 710  CONTINUE
      CALL MA57BD(MROWS, MNNZ, AA, FACT, LFACT, IFACT, LIFACT,
     &            LKEEP, KEEP, IWORK, ICNTL, CNTL, INFO, RINFO)
      IF (INFO(1) .EQ. -3 .OR. INFO(1) .EQ. -4) THEN
C       INFO(17): min LFACT;  INFO(18): min LIFACT. Grow by 2x for slack.
        DEALLOCATE(FACT)
        DEALLOCATE(IFACT)
        LFACT  = MAX(INT(DBLE(INFO(17)) * 2.0D0), 2 * LFACT)
        LIFACT = MAX(INT(DBLE(INFO(18)) * 2.0D0), 2 * LIFACT)
        ALLOCATE(FACT(LFACT))
        ALLOCATE(IFACT(LIFACT))
        GOTO 710
      END IF
      CALL SYSTEM_CLOCK(T_FAC_END)
      IF (INFO(1) .LT. 0) THEN
        WRITE(0,'(A,A,A,I6)') "ma57bd failed on ", TRIM(MTX_PATH),
     &        " INFO(1)=", INFO(1)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF

C     ---- Solve with iterative refinement (JOB=2) ----
C     MA57DD needs WORK(3*N) for JOB<=2 (no error analysis).
      LWORK = 3 * MROWS
      ALLOCATE(WORK(LWORK))
      ALLOCATE(XX(MROWS))
      ALLOCATE(RESBUF(MROWS))
      JOB = 1
      CALL SYSTEM_CLOCK(T_SOL_START)
      CALL MA57DD(JOB, MROWS, MNNZ, AA, IRA, JCA, FACT, LFACT,
     &            IFACT, LIFACT, BB, XX, RESBUF, WORK, IWORK,
     &            ICNTL, CNTL, INFO, RINFO)
      CALL SYSTEM_CLOCK(T_SOL_END)
      IF (INFO(1) .LT. 0) THEN
        WRITE(0,'(A,A,A,I6)') "ma57dd failed on ", TRIM(MTX_PATH),
     &        " INFO(1)=", INFO(1)
        HAVE_FAILURE = .TRUE.
        GOTO 500
      END IF

C     ---- Compute residual ||A·x - b||_2 / ||b||_2 ----
      ALLOCATE(AXBUF(MROWS))
      DO I = 1, MROWS
        AXBUF(I) = 0.0D0
      END DO
      DO I = 1, MNNZ
        ROW = IRA(I)
        COL = JCA(I)
        VAL = AA(I)
        AXBUF(ROW) = AXBUF(ROW) + VAL * XX(COL)
        IF (ROW .NE. COL) THEN
          AXBUF(COL) = AXBUF(COL) + VAL * XX(ROW)
        END IF
      END DO
      RES_NORM = 0.0D0
      RHS_NORM = 0.0D0
      DO I = 1, MROWS
        RES_NORM = RES_NORM + (AXBUF(I) - BB(I))**2
        RHS_NORM = RHS_NORM + BB(I)**2
      END DO
      IF (RHS_NORM .GT. 0.0D0) THEN
        REL_RES = SQRT(RES_NORM / RHS_NORM)
      ELSE
        REL_RES = SQRT(RES_NORM)
      END IF

      FAC_US = ((T_FAC_END - T_FAC_START) * 1000000_8) / CLOCK_RATE
      SOL_US = ((T_SOL_END - T_SOL_START) * 1000000_8) / CLOCK_RATE

C     Inertia: INFO(24) = #neg, INFO(25) = rank.
      INERT_NEG  = INFO(24)
      INERT_ZERO = MAX(0, MROWS - INFO(25))
      INERT_POS  = MAX(0, MROWS - INERT_NEG - INERT_ZERO)

C     ---- Write result ----
      OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &     ACTION='WRITE', IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot write out: ", TRIM(OUT_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 450
      END IF
      WRITE(12,'(A)') "solver ma57-2023.11.17"
      WRITE(12,'(A,I0)') "n ", MROWS
      WRITE(12,'(A,I0)') "nnz ", MNNZ
      WRITE(12,'(A,I0)') "inertia_pos ", INERT_POS
      WRITE(12,'(A,I0)') "inertia_neg ", INERT_NEG
      WRITE(12,'(A,I0)') "inertia_zero ", INERT_ZERO
      WRITE(12,'(A,I0)') "factor_us ", FAC_US
      WRITE(12,'(A,I0)') "solve_us ", SOL_US
      WRITE(12,'(A,ES24.16E3)') "rel_res ", REL_RES
      WRITE(12,'(A,I0)') "info1 ", INFO(1)
      WRITE(12,'(A,I0)') "info24 ", INFO(24)
      WRITE(12,'(A,I0)') "info25 ", INFO(25)
      WRITE(12,'(A)') "status ok"
      CLOSE(12)
      NDONE = NDONE + 1

 450  CONTINUE
      IF (ALLOCATED(AXBUF)) DEALLOCATE(AXBUF)
 500  CONTINUE
      DEALLOCATE(WORK)
      IF (ALLOCATED(XX)) DEALLOCATE(XX)
      IF (ALLOCATED(RESBUF)) DEALLOCATE(RESBUF)
 600  CONTINUE
      DEALLOCATE(FACT)
      DEALLOCATE(IFACT)
 650  CONTINUE
      DEALLOCATE(KEEP)
      DEALLOCATE(IWORK)
 700  CONTINUE
      IF (ALLOCATED(BB)) DEALLOCATE(BB)
 800  CONTINUE
      IF (ALLOCATED(IRA)) DEALLOCATE(IRA)
      IF (ALLOCATED(JCA)) DEALLOCATE(JCA)
      IF (ALLOCATED(AA)) DEALLOCATE(AA)

      IF (HAVE_FAILURE) THEN
        OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &       ACTION='WRITE')
        WRITE(12,'(A)') "solver ma57-2023.11.17"
        WRITE(12,'(A)') "status fail"
        CLOSE(12)
      END IF

      GOTO 100
 999  CONTINUE
      CLOSE(10)

      WRITE(0,'(A,I0,A,I0,A)') "ma57_bench done: ", NDONE, " / ",
     &      NMAT, " matrices succeeded"
      STOP 0
      END PROGRAM MA57_BENCH

C     ---------------------------------------------------------------
      SUBROUTINE PARSE_LINE(LINE, P1, P2, P3)
      CHARACTER(LEN=*), INTENT(IN) :: LINE
      CHARACTER(LEN=*), INTENT(OUT) :: P1, P2, P3
      INTEGER :: I, S, L
      L = LEN_TRIM(LINE)
      I = 1
      DO WHILE (I .LE. L .AND. LINE(I:I) .EQ. ' ')
        I = I + 1
      END DO
      S = I
      DO WHILE (I .LE. L .AND. LINE(I:I) .NE. ' ')
        I = I + 1
      END DO
      P1 = LINE(S:I-1)
      DO WHILE (I .LE. L .AND. LINE(I:I) .EQ. ' ')
        I = I + 1
      END DO
      S = I
      DO WHILE (I .LE. L .AND. LINE(I:I) .NE. ' ')
        I = I + 1
      END DO
      P2 = LINE(S:I-1)
      DO WHILE (I .LE. L .AND. LINE(I:I) .EQ. ' ')
        I = I + 1
      END DO
      S = I
      DO WHILE (I .LE. L .AND. LINE(I:I) .NE. ' ')
        I = I + 1
      END DO
      P3 = LINE(S:I-1)
      RETURN
      END SUBROUTINE PARSE_LINE

C     ---------------------------------------------------------------
C     Read a Matrix Market symmetric coordinate file (1-indexed).
C     Header: %%MatrixMarket matrix coordinate real symmetric
C     Then a single "rows cols nnz" line, then nnz "i j v" lines.
C     ---------------------------------------------------------------
      SUBROUTINE READ_MTX(PATH, NR, NC, NNZ, IRN, JCN, VALS, IOS)
      CHARACTER(LEN=*), INTENT(IN) :: PATH
      INTEGER, INTENT(OUT) :: NR, NC, NNZ
      INTEGER, ALLOCATABLE, INTENT(OUT) :: IRN(:), JCN(:)
      DOUBLE PRECISION, ALLOCATABLE, INTENT(OUT) :: VALS(:)
      INTEGER, INTENT(OUT) :: IOS
      CHARACTER(LEN=1024) :: LINE
      INTEGER :: K

      OPEN(UNIT=20, FILE=TRIM(PATH), STATUS='OLD',
     &     ACTION='READ', IOSTAT=IOS)
      IF (IOS .NE. 0) RETURN

      READ(20, '(A)', IOSTAT=IOS) LINE
      IF (IOS .NE. 0) THEN
        CLOSE(20)
        RETURN
      END IF

 200  CONTINUE
      READ(20, '(A)', IOSTAT=IOS) LINE
      IF (IOS .NE. 0) THEN
        CLOSE(20)
        RETURN
      END IF
      IF (LINE(1:1) .EQ. '%') GOTO 200

      READ(LINE, *, IOSTAT=IOS) NR, NC, NNZ
      IF (IOS .NE. 0) THEN
        CLOSE(20)
        RETURN
      END IF

      ALLOCATE(IRN(NNZ))
      ALLOCATE(JCN(NNZ))
      ALLOCATE(VALS(NNZ))

      DO K = 1, NNZ
        READ(20, *, IOSTAT=IOS) IRN(K), JCN(K), VALS(K)
        IF (IOS .NE. 0) THEN
          CLOSE(20)
          RETURN
        END IF
      END DO

      CLOSE(20)
      IOS = 0
      RETURN
      END SUBROUTINE READ_MTX
