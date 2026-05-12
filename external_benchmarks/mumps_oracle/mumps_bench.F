C
C   mumps_bench.F — Native MUMPS oracle for the feral consensus benchmark.
C
C   Reads a manifest file of (mtx_path, rhs_path, out_path) triples.
C   For each triple, factors the matrix with DMUMPS (SYM=2 indefinite,
C   sequential, double precision), computes the residual against the
C   given RHS, and writes a plain-text result to out_path.
C
C   MUMPS init/free happens once for the whole manifest to avoid
C   per-matrix MPI/struct overhead.
C
C   Usage:  ./mumps_bench manifest.txt
C
C   Manifest format (one matrix per line, whitespace-separated):
C       /abs/path/to/MGH10S_0000.mtx /abs/path/MGH10S_0000.rhs.txt /abs/path/out.txt
C
C   RHS file format: one f64 per line, length must match the matrix dim.
C
C   Output file format (one key per line):
C       solver mumps-5.8.2
C       n <int>
C       nnz <int>
C       inertia_pos <int>
C       inertia_neg <int>
C       inertia_zero <int>
C       factor_us <int>
C       solve_us <int>
C       residual <f64>
C       infog1 <int>
C       infog28 <int>
C       infog9 <int>     (real entries effectively used in factors)
C       rinfog4 <f64>    (||A||_inf — populated when ICNTL(11)>=1)
C       rinfog5 <f64>    (||X||_inf)
C       rinfog6 <f64>    (scaled residual ||Ax-b||_inf / (||A||_inf·||x||_inf+||b||_inf))
C       rinfog7 <f64>    (omega_1 componentwise residual)
C       rinfog8 <f64>    (omega_2 componentwise residual)
C       rinfog9 <f64>    (forward error upper bound: omega1*COND1 + omega2*COND2)
C       rinfog10 <f64>   (COND1: componentwise condition number, set 1)
C       rinfog11 <f64>   (COND2: componentwise condition number, set 2)
C       status ok|fail
C
C   Note on ICNTL(11): MUMPS 5.8.2 populates the condition-number
C   fields RINFOG(10)/(11) only when ICNTL(11)==1 (full statistics);
C   ICNTL(11)==2 is the cheaper variant that skips LCOND and leaves
C   RINFOG(9..11) unset. We use ICNTL(11)=1 here so RINFOG(11) is
C   available for cross-validation against feral's
C   estimate_condition_1norm. See dsol_driver.F:5742 in the MUMPS
C   source. The kkt-feature-gaps F2.2 plan note refers to ICNTL(11)=2
C   but the fields it requests (RINFOG(11)) require ICNTL(11)=1.
C
      PROGRAM MUMPS_BENCH
      IMPLICIT NONE
      INCLUDE 'dmumps_struc.h'
      TYPE (DMUMPS_STRUC) MP

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
      INTEGER :: IOS, NMAT, NDONE, IUM, IUR, IUO
      INTEGER(8) :: I8
      INTEGER :: I, MROWS, MCOLS, MNNZ
      DOUBLE PRECISION :: VAL
      INTEGER :: ROW, COL
      DOUBLE PRECISION, ALLOCATABLE :: AA(:), BB(:), XCOPY(:), AXBUF(:)
      INTEGER, ALLOCATABLE :: IRA(:), JCA(:)
      DOUBLE PRECISION :: RES_NORM, RHS_NORM, REL_RES
      INTEGER(8) :: T_FAC_START, T_FAC_END, T_SOL_START, T_SOL_END
      INTEGER(8) :: CLOCK_RATE, CLOCK_MAX
      INTEGER(8) :: FAC_US, SOL_US
      INTEGER :: STATUS_CODE
      LOGICAL :: HAVE_FAILURE

C     Read command-line arg
      IF (COMMAND_ARGUMENT_COUNT() .LT. 1) THEN
        WRITE(0,*) "Usage: mumps_bench <manifest.txt>"
        STOP 2
      END IF
      CALL GET_COMMAND_ARGUMENT(1, MANIFEST_PATH)

C     Initialize MUMPS once
      MP%COMM = -987654
      MP%JOB = -1
      MP%SYM = 2
      MP%PAR = 1
      CALL DMUMPS(MP)
      IF (MP%INFOG(1) .LT. 0) THEN
        WRITE(0,'(A,I6)') "MUMPS init failed, INFOG(1)=", MP%INFOG(1)
        STOP 3
      END IF

C     Suppress MUMPS console output and request null pivot detection
      MP%ICNTL(1) = 0
      MP%ICNTL(2) = 0
      MP%ICNTL(3) = 0
      MP%ICNTL(4) = 0
      MP%ICNTL(24) = 1
C     ICNTL(11)=1: full error analysis statistics including the two
C     componentwise condition numbers RINFOG(10) (COND1) and
C     RINFOG(11) (COND2). Used by feral's F2.2 cross-validation.
C     ICNTL(11)=2 is cheaper but leaves RINFOG(9..11) unset, so it
C     would not satisfy the cross-validation oracle.
      MP%ICNTL(11) = 1
C     ICNTL(10) > 0: max number of steps of iterative refinement.
C     MUMPS default is 0 (no refinement). The bench harness sets it
C     to 2, matching the feral driver which uses
C     `solve_sparse_refined` (Richardson iteration with a similar
C     step budget). This makes the cross-solver accuracy comparison
C     fair — every solver is configured to refine.
      MP%ICNTL(10) = 2

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

C     ---- Configure MUMPS for this matrix ----
      MP%N = MROWS
      MP%NNZ = INT(MNNZ, 8)
      ALLOCATE(MP%IRN(MNNZ))
      ALLOCATE(MP%JCN(MNNZ))
      ALLOCATE(MP%A(MNNZ))
      ALLOCATE(MP%RHS(MROWS))
      DO I = 1, MNNZ
        MP%IRN(I) = IRA(I)
        MP%JCN(I) = JCA(I)
        MP%A(I) = AA(I)
      END DO
      DO I = 1, MROWS
        MP%RHS(I) = BB(I)
      END DO

C     ---- Analyze + Factor ----
      CALL SYSTEM_CLOCK(T_FAC_START)
      MP%JOB = 4
      CALL DMUMPS(MP)
      CALL SYSTEM_CLOCK(T_FAC_END)
      IF (MP%INFOG(1) .LT. 0) THEN
        WRITE(0,'(A,A,A,I6)') "factor failed on ", TRIM(MTX_PATH),
     &        " INFOG(1)=", MP%INFOG(1)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF

C     ---- Solve ----
      CALL SYSTEM_CLOCK(T_SOL_START)
      MP%JOB = 3
      CALL DMUMPS(MP)
      CALL SYSTEM_CLOCK(T_SOL_END)
      IF (MP%INFOG(1) .LT. 0) THEN
        WRITE(0,'(A,A,A,I6)') "solve failed on ", TRIM(MTX_PATH),
     &        " INFOG(1)=", MP%INFOG(1)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF

C     ---- Compute residual ||A·x - b||_2 / ||b||_2 ----
      ALLOCATE(AXBUF(MROWS))
      ALLOCATE(XCOPY(MROWS))
      DO I = 1, MROWS
        XCOPY(I) = MP%RHS(I)
        AXBUF(I) = 0.0D0
      END DO
      DO I = 1, MNNZ
        ROW = IRA(I)
        COL = JCA(I)
        VAL = AA(I)
        AXBUF(ROW) = AXBUF(ROW) + VAL * XCOPY(COL)
        IF (ROW .NE. COL) THEN
          AXBUF(COL) = AXBUF(COL) + VAL * XCOPY(ROW)
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

C     ---- Write result ----
      OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &     ACTION='WRITE', IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot write out: ", TRIM(OUT_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 500
      END IF
      WRITE(12,'(A)') "solver mumps-5.8.2"
      WRITE(12,'(A,I0)') "n ", MROWS
      WRITE(12,'(A,I0)') "nnz ", MNNZ
      WRITE(12,'(A,I0)') "inertia_pos ", MROWS - MP%INFOG(12)
     &      - MP%INFOG(28)
      WRITE(12,'(A,I0)') "inertia_neg ", MP%INFOG(12)
      WRITE(12,'(A,I0)') "inertia_zero ", MP%INFOG(28)
      WRITE(12,'(A,I0)') "factor_us ", FAC_US
      WRITE(12,'(A,I0)') "solve_us ", SOL_US
      WRITE(12,'(A,ES24.16E3)') "residual ", REL_RES
      WRITE(12,'(A,I0)') "infog1 ", MP%INFOG(1)
      WRITE(12,'(A,I0)') "infog28 ", MP%INFOG(28)
C     INFOG(9) = real entries effectively used in factors; "factor_nnz"
C     for cross-solver fill comparison. Field 20 (estimated) and field 9
C     (effective) can disagree when delayed pivots inflate storage; the
C     effective value is the right reference for feral.
      WRITE(12,'(A,I0)') "infog9 ", MP%INFOG(9)
C     ICNTL(11)=1 error-analysis output. RINFOG(4)=||A||_inf,
C     RINFOG(5)=||X||_inf, RINFOG(6)=scaled residual, RINFOG(7)=omega1,
C     RINFOG(8)=omega2, RINFOG(9)=forward-error upper bound,
C     RINFOG(10)=COND1, RINFOG(11)=COND2. Definitions: dsol_aux.F:935
C     (DMUMPS_SOL_LCOND). For perfectly singular factors MUMPS leaves
C     these at 0; downstream code must treat 0 as "unavailable".
      WRITE(12,'(A,ES24.16E3)') "rinfog4 ", MP%RINFOG(4)
      WRITE(12,'(A,ES24.16E3)') "rinfog5 ", MP%RINFOG(5)
      WRITE(12,'(A,ES24.16E3)') "rinfog6 ", MP%RINFOG(6)
      WRITE(12,'(A,ES24.16E3)') "rinfog7 ", MP%RINFOG(7)
      WRITE(12,'(A,ES24.16E3)') "rinfog8 ", MP%RINFOG(8)
      WRITE(12,'(A,ES24.16E3)') "rinfog9 ", MP%RINFOG(9)
      WRITE(12,'(A,ES24.16E3)') "rinfog10 ", MP%RINFOG(10)
      WRITE(12,'(A,ES24.16E3)') "rinfog11 ", MP%RINFOG(11)
      WRITE(12,'(A)') "status ok"
      CLOSE(12)
      NDONE = NDONE + 1

 500  CONTINUE
      DEALLOCATE(AXBUF)
      DEALLOCATE(XCOPY)
 600  CONTINUE
      DEALLOCATE(MP%IRN)
      DEALLOCATE(MP%JCN)
      DEALLOCATE(MP%A)
      DEALLOCATE(MP%RHS)
 700  CONTINUE
      IF (ALLOCATED(BB)) DEALLOCATE(BB)
 800  CONTINUE
      IF (ALLOCATED(IRA)) DEALLOCATE(IRA)
      IF (ALLOCATED(JCA)) DEALLOCATE(JCA)
      IF (ALLOCATED(AA)) DEALLOCATE(AA)

      IF (HAVE_FAILURE) THEN
        OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &       ACTION='WRITE')
        WRITE(12,'(A)') "solver mumps-5.8.2"
        WRITE(12,'(A)') "status fail"
        CLOSE(12)
      END IF

      GOTO 100
 999  CONTINUE
      CLOSE(10)

      WRITE(0,'(A,I0,A,I0,A)') "mumps_bench done: ", NDONE, " / ",
     &      NMAT, " matrices succeeded"

C     Free MUMPS instance
      MP%JOB = -2
      CALL DMUMPS(MP)
      STOP 0
      END PROGRAM MUMPS_BENCH

C     ---------------------------------------------------------------
      SUBROUTINE PARSE_LINE(LINE, P1, P2, P3)
      CHARACTER(LEN=*), INTENT(IN) :: LINE
      CHARACTER(LEN=*), INTENT(OUT) :: P1, P2, P3
      INTEGER :: I, J, S, L
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
C     Header line: %%MatrixMarket matrix coordinate real symmetric
C     Comment lines start with %.
C     Then a single "rows cols nnz" line, then nnz "i j v" lines.
C     Returns 1-indexed IRN/JCN. The lower triangle is what's stored.
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

C     Read and discard the %%MatrixMarket header
      READ(20, '(A)', IOSTAT=IOS) LINE
      IF (IOS .NE. 0) THEN
        CLOSE(20)
        RETURN
      END IF

C     Skip any additional comment lines starting with '%', then
C     read the size line "rows cols nnz"
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
