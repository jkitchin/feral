C
C   mumps_schur_bench.F -- MUMPS Schur-complement oracle for feral F3.3.
C
C   Reads a manifest of (mtx_path, schur_idx_path, out_txt, out_bin)
C   tuples. For each, runs DMUMPS with ICNTL(19)=1 (centralized dense
C   Schur, full square symmetric) on the user-supplied Schur index
C   list and emits the resulting SIZE_SCHUR x SIZE_SCHUR column-major
C   buffer as a raw little-endian f64 binary file plus a key/value
C   metadata sidecar.
C
C   No solve is performed -- only analyze + factor + Schur extraction.
C   The cross-validation oracle for feral's
C   factorize_multifrontal_with_schur compares the dense Schur block
C   entry-wise; per dev/research/schur-complement.md D7, acceptance
C   is max relative error <= 1e-10 on N >= 100 corpus matrices.
C
C   Manifest format (one matrix per line, four whitespace-separated
C   absolute paths):
C       /abs/MGH10S_0000.mtx /abs/MGH10S_0000.schur_idx.txt
C       /abs/MGH10S_0000.out.txt /abs/MGH10S_0000.schur.bin
C
C   schur_idx_path format: first line is SIZE_SCHUR, subsequent lines
C   are 1-indexed original column indices (one per line). Length must
C   match SIZE_SCHUR; the Python driver shifts feral's 0-indexed list
C   on write.
C
C   Output file format (out_txt, one key per line):
C       solver mumps-5.8.2-schur
C       icntl19 1
C       n <int>
C       nnz <int>
C       n_schur <int>
C       factor_us <int>
C       infog1 <int>
C       infog28 <int>
C       schur_bin <path>          (echoes the bin path the driver passed)
C       status ok|fail
C
C   The binary file (out_bin) contains exactly SIZE_SCHUR^2 doubles
C   in native (little-endian on this platform) byte order, column-
C   major, full symmetric.  Layout matches feral's SchurBlock.data.
C
      PROGRAM MUMPS_SCHUR_BENCH
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
        SUBROUTINE PARSE_LINE4(LINE, P1, P2, P3, P4)
          CHARACTER(LEN=*), INTENT(IN) :: LINE
          CHARACTER(LEN=*), INTENT(OUT) :: P1, P2, P3, P4
        END SUBROUTINE PARSE_LINE4
      END INTERFACE

      CHARACTER(LEN=4096) :: MANIFEST_PATH, LINE
      CHARACTER(LEN=4096) :: MTX_PATH, IDX_PATH, OUT_PATH, BIN_PATH
      INTEGER :: IOS, NMAT, NDONE
      INTEGER :: I, J, MROWS, MCOLS, MNNZ, NSCHUR
      DOUBLE PRECISION, ALLOCATABLE :: AA(:)
      INTEGER, ALLOCATABLE :: IRA(:), JCA(:)
      INTEGER, ALLOCATABLE :: SCHUR_IDX(:)
      INTEGER(8) :: T_FAC_START, T_FAC_END
      INTEGER(8) :: CLOCK_RATE, CLOCK_MAX
      INTEGER(8) :: FAC_US
      LOGICAL :: HAVE_FAILURE

      IF (COMMAND_ARGUMENT_COUNT() .LT. 1) THEN
        WRITE(0,*) "Usage: mumps_schur_bench <manifest.txt>"
        STOP 2
      END IF
      CALL GET_COMMAND_ARGUMENT(1, MANIFEST_PATH)

C     Initialize MUMPS once. SYM=2 (general symmetric indefinite),
C     PAR=1 (host participates in computation).
      MP%COMM = -987654
      MP%JOB = -1
      MP%SYM = 2
      MP%PAR = 1
      CALL DMUMPS(MP)
      IF (MP%INFOG(1) .LT. 0) THEN
        WRITE(0,'(A,I6)') "MUMPS init failed, INFOG(1)=", MP%INFOG(1)
        STOP 3
      END IF

C     Suppress MUMPS console output.  Request Schur, centralized dense
C     full-square symmetric (ICNTL(19)=1).  ICNTL(24)=1 enables null
C     pivot detection so we still see legitimate failures rather than
C     spurious infinities in the Schur.
      MP%ICNTL(1) = 0
      MP%ICNTL(2) = 0
      MP%ICNTL(3) = 0
      MP%ICNTL(4) = 0
      MP%ICNTL(19) = 1
      MP%ICNTL(24) = 1

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
      CALL PARSE_LINE4(LINE, MTX_PATH, IDX_PATH, OUT_PATH, BIN_PATH)

      HAVE_FAILURE = .FALSE.

C     ---- Read MTX ----
      CALL READ_MTX(MTX_PATH, MROWS, MCOLS, MNNZ, IRA, JCA, AA, IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "READ_MTX failed: ", TRIM(MTX_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 800
      END IF

C     ---- Read Schur index list ----
      OPEN(UNIT=11, FILE=TRIM(IDX_PATH), STATUS='OLD',
     &     ACTION='READ', IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot open schur idx: ", TRIM(IDX_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 800
      END IF
      READ(11, *, IOSTAT=IOS) NSCHUR
      IF (IOS .NE. 0 .OR. NSCHUR .LT. 1 .OR. NSCHUR .GE. MROWS) THEN
        WRITE(0,*) "bad schur size for ", TRIM(MTX_PATH),
     &             " nschur=", NSCHUR, " n=", MROWS
        HAVE_FAILURE = .TRUE.
        CLOSE(11)
        GOTO 800
      END IF
      ALLOCATE(SCHUR_IDX(NSCHUR))
      DO I = 1, NSCHUR
        READ(11, *, IOSTAT=IOS) SCHUR_IDX(I)
        IF (IOS .NE. 0
     &      .OR. SCHUR_IDX(I) .LT. 1
     &      .OR. SCHUR_IDX(I) .GT. MROWS) THEN
          WRITE(0,*) "bad schur idx at ", I, " in ", TRIM(IDX_PATH)
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
      DO I = 1, MNNZ
        MP%IRN(I) = IRA(I)
        MP%JCN(I) = JCA(I)
        MP%A(I) = AA(I)
      END DO

C     Schur configuration (per dini_defaults.F:471-485,
C     dmumps_driver.F:1462-1479).  SIZE_SCHUR is the count;
C     LISTVAR_SCHUR is the 1-indexed user index list (preserved order
C     of the Schur block returned to the user); SCHUR is the host
C     output buffer of length SIZE_SCHUR^2 (dense column-major full
C     square for ICNTL(19)=1 + SYM=2; full-symmetric output is
C     guaranteed by DMUMPS_SYMMETRIZE in dtype3_root.F:1258-1294).
      MP%SIZE_SCHUR = NSCHUR
      ALLOCATE(MP%LISTVAR_SCHUR(NSCHUR))
      DO I = 1, NSCHUR
        MP%LISTVAR_SCHUR(I) = SCHUR_IDX(I)
      END DO
      ALLOCATE(MP%SCHUR(INT(NSCHUR, 8) * INT(NSCHUR, 8)))
      DO J = 1, NSCHUR * NSCHUR
        MP%SCHUR(J) = 0.0D0
      END DO

C     ---- Analyze + Factor (no solve) ----
      CALL SYSTEM_CLOCK(T_FAC_START)
      MP%JOB = 4
      CALL DMUMPS(MP)
      CALL SYSTEM_CLOCK(T_FAC_END)
      IF (MP%INFOG(1) .LT. 0) THEN
        WRITE(0,'(A,A,A,I6)') "schur factor failed on ",
     &        TRIM(MTX_PATH), " INFOG(1)=", MP%INFOG(1)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF

      FAC_US = ((T_FAC_END - T_FAC_START) * 1000000_8) / CLOCK_RATE

C     ---- Symmetrize: MUMPS centralized-dense (ICNTL(19)=1) writes the
C     upper triangle (column-major, rows 0..=j of column j) and leaves
C     the strict lower triangle zero. The documented
C     DMUMPS_SYMMETRIZE pass at dtype3_root.F:1258-1294 only mirrors
C     for the distributed-block-cyclic modes (ICNTL(19)>=2). Mirror
C     here so the on-disk layout matches feral's full-square
C     SchurBlock convention.
      DO J = 1, NSCHUR
        DO I = J + 1, NSCHUR
          MP%SCHUR((INT(J - 1, 8) * INT(NSCHUR, 8)) + INT(I, 8))
     &        = MP%SCHUR((INT(I - 1, 8) * INT(NSCHUR, 8)) + INT(J, 8))
        END DO
      END DO

C     ---- Write Schur block as raw little-endian f64 binary ----
C     STREAM access produces a flat byte stream with no Fortran
C     record markers, so feral's Rust reader can mmap or read
C     n_schur^2 * 8 bytes directly.
      OPEN(UNIT=13, FILE=TRIM(BIN_PATH), STATUS='REPLACE',
     &     ACTION='WRITE', FORM='UNFORMATTED', ACCESS='STREAM',
     &     IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot open schur bin: ", TRIM(BIN_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF
      WRITE(13) MP%SCHUR
      CLOSE(13)

C     ---- Write metadata sidecar ----
      OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &     ACTION='WRITE', IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot write out: ", TRIM(OUT_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF
      WRITE(12,'(A)') "solver mumps-5.8.2-schur"
      WRITE(12,'(A)') "icntl19 1"
      WRITE(12,'(A,I0)') "n ", MROWS
      WRITE(12,'(A,I0)') "nnz ", MNNZ
      WRITE(12,'(A,I0)') "n_schur ", NSCHUR
      WRITE(12,'(A,I0)') "factor_us ", FAC_US
      WRITE(12,'(A,I0)') "infog1 ", MP%INFOG(1)
      WRITE(12,'(A,I0)') "infog28 ", MP%INFOG(28)
      WRITE(12,'(A,A)') "schur_bin ", TRIM(BIN_PATH)
      WRITE(12,'(A)') "status ok"
      CLOSE(12)
      NDONE = NDONE + 1

 600  CONTINUE
      DEALLOCATE(MP%IRN)
      DEALLOCATE(MP%JCN)
      DEALLOCATE(MP%A)
      IF (ASSOCIATED(MP%LISTVAR_SCHUR)) DEALLOCATE(MP%LISTVAR_SCHUR)
      IF (ASSOCIATED(MP%SCHUR)) DEALLOCATE(MP%SCHUR)
      MP%SIZE_SCHUR = 0
 700  CONTINUE
      IF (ALLOCATED(SCHUR_IDX)) DEALLOCATE(SCHUR_IDX)
 800  CONTINUE
      IF (ALLOCATED(IRA)) DEALLOCATE(IRA)
      IF (ALLOCATED(JCA)) DEALLOCATE(JCA)
      IF (ALLOCATED(AA)) DEALLOCATE(AA)

      IF (HAVE_FAILURE) THEN
        OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &       ACTION='WRITE')
        WRITE(12,'(A)') "solver mumps-5.8.2-schur"
        WRITE(12,'(A)') "status fail"
        CLOSE(12)
      END IF

      GOTO 100
 999  CONTINUE
      CLOSE(10)

      WRITE(0,'(A,I0,A,I0,A)') "mumps_schur_bench done: ", NDONE,
     &      " / ", NMAT, " matrices succeeded"

C     Free MUMPS instance
      MP%JOB = -2
      CALL DMUMPS(MP)
      STOP 0
      END PROGRAM MUMPS_SCHUR_BENCH

C     ---------------------------------------------------------------
      SUBROUTINE PARSE_LINE4(LINE, P1, P2, P3, P4)
      CHARACTER(LEN=*), INTENT(IN) :: LINE
      CHARACTER(LEN=*), INTENT(OUT) :: P1, P2, P3, P4
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
      DO WHILE (I .LE. L .AND. LINE(I:I) .EQ. ' ')
        I = I + 1
      END DO
      S = I
      DO WHILE (I .LE. L .AND. LINE(I:I) .NE. ' ')
        I = I + 1
      END DO
      P4 = LINE(S:I-1)
      RETURN
      END SUBROUTINE PARSE_LINE4

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
