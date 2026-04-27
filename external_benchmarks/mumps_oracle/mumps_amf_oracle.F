C
C   mumps_amf_oracle.F -- MUMPS HAMF4 ordering oracle for feral-amf.
C
C   Reads a manifest file of (mtx_path, out_path) pairs. For each,
C   runs MUMPS analyze with ICNTL(7) = 2 (force AMF / approximate
C   minimum fill) and dumps the symmetric permutation plus the
C   estimated nnz_L (INFOG(20)) to the output text file.
C
C   Analyze-only -- no factor or solve. The corresponding nnz_L
C   measurement that feral cross-checks against is the symbolic
C   factor entry estimate INFOG(20). For Phase D's runtime
C   comparison, the canonical post-factor INFOG(9) lives in the
C   companion mumps_bench harness; the AMF oracle is concerned with
C   the pre-numerical fill prediction only.
C
C   Usage:  ./mumps_amf_oracle manifest.txt
C
C   Manifest format (one matrix per line, whitespace-separated):
C       /abs/path/to/MGH10S_0000.mtx /abs/path/to/MGH10S_0000.hamf4.txt
C
C   Output file format (one key per line, then sym_perm block):
C       solver mumps-5.8.2-amf
C       icntl7 2
C       n <int>
C       nnz <int>
C       nnz_l_estimated <int>     (INFOG(20))
C       analyze_us <int>
C       infog1 <int>
C       infog4 <int>              (estimated total real workspace)
C       infog20 <int>             (alias of nnz_l_estimated)
C       infog24 <int>             (estimated total integer workspace)
C       status ok|fail
C       sym_perm
C       <i1>
C       <i2>
C       ...
C       <iN>
C
C   The sym_perm block is 1-indexed (MUMPS convention). The Python
C   driver translates this to a 0-indexed array in the canonical
C   .hamf4.json sidecar.
C
      PROGRAM MUMPS_AMF_ORACLE
      IMPLICIT NONE
      INCLUDE 'dmumps_struc.h'
      TYPE (DMUMPS_STRUC) MP

      INTERFACE
        SUBROUTINE READ_MTX_NOVAL(PATH, NR, NC, NNZ, IRN, JCN, IOS)
          CHARACTER(LEN=*), INTENT(IN) :: PATH
          INTEGER, INTENT(OUT) :: NR, NC, NNZ
          INTEGER, ALLOCATABLE, INTENT(OUT) :: IRN(:), JCN(:)
          INTEGER, INTENT(OUT) :: IOS
        END SUBROUTINE READ_MTX_NOVAL
        SUBROUTINE PARSE_LINE2(LINE, P1, P2)
          CHARACTER(LEN=*), INTENT(IN) :: LINE
          CHARACTER(LEN=*), INTENT(OUT) :: P1, P2
        END SUBROUTINE PARSE_LINE2
      END INTERFACE

      CHARACTER(LEN=4096) :: MANIFEST_PATH, LINE
      CHARACTER(LEN=4096) :: MTX_PATH, OUT_PATH
      INTEGER :: IOS, NMAT, NDONE
      INTEGER :: I, MROWS, MCOLS, MNNZ
      INTEGER, ALLOCATABLE :: IRA(:), JCA(:)
      INTEGER(8) :: T_AN_START, T_AN_END
      INTEGER(8) :: CLOCK_RATE, CLOCK_MAX
      INTEGER(8) :: ANALYZE_US
      LOGICAL :: HAVE_FAILURE

C     Read command-line arg
      IF (COMMAND_ARGUMENT_COUNT() .LT. 1) THEN
        WRITE(0,*) "Usage: mumps_amf_oracle <manifest.txt>"
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

C     Suppress MUMPS console output. Force AMF ordering.
      MP%ICNTL(1) = 0
      MP%ICNTL(2) = 0
      MP%ICNTL(3) = 0
      MP%ICNTL(4) = 0
C     ICNTL(7) = 2 forces approximate minimum fill (HAMF4) ordering.
C     ana_set_ordering.F:43 dispatches to ana_amf when ICNTL(7) is 2.
      MP%ICNTL(7) = 2

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
      CALL PARSE_LINE2(LINE, MTX_PATH, OUT_PATH)

      HAVE_FAILURE = .FALSE.

C     ---- Read MTX ----
      CALL READ_MTX_NOVAL(MTX_PATH, MROWS, MCOLS, MNNZ, IRA, JCA, IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "READ_MTX failed: ", TRIM(MTX_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 800
      END IF

C     ---- Configure MUMPS for this matrix ----
C     We do not need numerical values for analyze-only; allocate A
C     anyway because some MUMPS versions sanity-check the pointer.
      MP%N = MROWS
      MP%NNZ = INT(MNNZ, 8)
      ALLOCATE(MP%IRN(MNNZ))
      ALLOCATE(MP%JCN(MNNZ))
      ALLOCATE(MP%A(MNNZ))
      DO I = 1, MNNZ
        MP%IRN(I) = IRA(I)
        MP%JCN(I) = JCA(I)
        MP%A(I) = 0.0D0
      END DO

C     ---- Analyze (job=1) with ICNTL(7) = 2 ----
      CALL SYSTEM_CLOCK(T_AN_START)
      MP%JOB = 1
      CALL DMUMPS(MP)
      CALL SYSTEM_CLOCK(T_AN_END)
      IF (MP%INFOG(1) .LT. 0) THEN
        WRITE(0,'(A,A,A,I6)') "analyze failed on ", TRIM(MTX_PATH),
     &        " INFOG(1)=", MP%INFOG(1)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF

      ANALYZE_US = ((T_AN_END - T_AN_START) * 1000000_8) / CLOCK_RATE

C     ---- Write result ----
      OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &     ACTION='WRITE', IOSTAT=IOS)
      IF (IOS .NE. 0) THEN
        WRITE(0,*) "Cannot write out: ", TRIM(OUT_PATH)
        HAVE_FAILURE = .TRUE.
        GOTO 600
      END IF
      WRITE(12,'(A)') "solver mumps-5.8.2-amf"
      WRITE(12,'(A)') "icntl7 2"
      WRITE(12,'(A,I0)') "n ", MROWS
      WRITE(12,'(A,I0)') "nnz ", MNNZ
      WRITE(12,'(A,I0)') "nnz_l_estimated ", MP%INFOG(20)
      WRITE(12,'(A,I0)') "analyze_us ", ANALYZE_US
      WRITE(12,'(A,I0)') "infog1 ", MP%INFOG(1)
      WRITE(12,'(A,I0)') "infog4 ", MP%INFOG(4)
      WRITE(12,'(A,I0)') "infog20 ", MP%INFOG(20)
      WRITE(12,'(A,I0)') "infog24 ", MP%INFOG(24)
      WRITE(12,'(A)') "status ok"
      WRITE(12,'(A)') "sym_perm"
      DO I = 1, MROWS
        WRITE(12,'(I0)') MP%SYM_PERM(I)
      END DO
      CLOSE(12)
      NDONE = NDONE + 1

 600  CONTINUE
      DEALLOCATE(MP%IRN)
      DEALLOCATE(MP%JCN)
      DEALLOCATE(MP%A)
 800  CONTINUE
      IF (ALLOCATED(IRA)) DEALLOCATE(IRA)
      IF (ALLOCATED(JCA)) DEALLOCATE(JCA)

      IF (HAVE_FAILURE) THEN
        OPEN(UNIT=12, FILE=TRIM(OUT_PATH), STATUS='REPLACE',
     &       ACTION='WRITE')
        WRITE(12,'(A)') "solver mumps-5.8.2-amf"
        WRITE(12,'(A)') "status fail"
        CLOSE(12)
      END IF

      GOTO 100
 999  CONTINUE
      CLOSE(10)

      WRITE(0,'(A,I0,A,I0,A)') "mumps_amf_oracle done: ", NDONE,
     &      " / ", NMAT, " matrices succeeded"

C     Free MUMPS instance
      MP%JOB = -2
      CALL DMUMPS(MP)
      STOP 0
      END PROGRAM MUMPS_AMF_ORACLE

C     ---------------------------------------------------------------
      SUBROUTINE PARSE_LINE2(LINE, P1, P2)
      CHARACTER(LEN=*), INTENT(IN) :: LINE
      CHARACTER(LEN=*), INTENT(OUT) :: P1, P2
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
      RETURN
      END SUBROUTINE PARSE_LINE2

C     ---------------------------------------------------------------
C     Read a Matrix Market symmetric coordinate file (1-indexed).
C     Pattern only -- numerical values are discarded.
C     ---------------------------------------------------------------
      SUBROUTINE READ_MTX_NOVAL(PATH, NR, NC, NNZ, IRN, JCN, IOS)
      CHARACTER(LEN=*), INTENT(IN) :: PATH
      INTEGER, INTENT(OUT) :: NR, NC, NNZ
      INTEGER, ALLOCATABLE, INTENT(OUT) :: IRN(:), JCN(:)
      INTEGER, INTENT(OUT) :: IOS
      CHARACTER(LEN=1024) :: LINE
      DOUBLE PRECISION :: VAL_DUMMY
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

C     Skip any additional comment lines, then read "rows cols nnz"
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

      DO K = 1, NNZ
        READ(20, *, IOSTAT=IOS) IRN(K), JCN(K), VAL_DUMMY
        IF (IOS .NE. 0) THEN
          CLOSE(20)
          RETURN
        END IF
      END DO

      CLOSE(20)
      IOS = 0
      RETURN
      END SUBROUTINE READ_MTX_NOVAL
