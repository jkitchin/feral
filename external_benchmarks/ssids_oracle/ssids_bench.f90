! ssids_bench.f90 — Native SPRAL/SSIDS oracle for the feral consensus benchmark.
!
! Reads a manifest file of (mtx_path, rhs_path, out_path) triples.
! For each triple, factors the matrix with SSIDS (posdef=.false. for
! indefinite KKTs), computes the residual against the given RHS, and
! writes a plain-text result to out_path.
!
! Each matrix gets its own akeep/fkeep — SSIDS shared state is per-matrix
! because the structure (akeep) depends on the specific sparsity pattern.
!
! Usage:  ./ssids_bench manifest.txt
!
! Manifest format (one matrix per line, whitespace-separated):
!     /abs/path/to/MGH10S_0000.mtx /abs/path/MGH10S_0000.rhs.txt /abs/path/out.txt
!
! Output file format (one key per line):
!     solver ssids
!     n <int>
!     nnz <int>
!     inertia_pos <int>
!     inertia_neg <int>
!     inertia_zero <int>
!     factor_us <int>
!     solve_us <int>
!     residual <f64>
!     ssids_flag <int>
!     matrix_rank <int>
!     num_delay <int>
!     num_factor <int>     (entries in factors after factorization)
!     status ok|fail

program ssids_bench
   use spral_ssids
   implicit none

   integer, parameter :: long = selected_int_kind(18)
   integer, parameter :: wp = kind(0.0d0)

   type (ssids_akeep) :: akeep
   type (ssids_fkeep) :: fkeep
   type (ssids_options) :: options
   type (ssids_inform) :: inform

   character(len=4096) :: manifest_path, line
   character(len=4096) :: mtx_path, rhs_path, out_path
   integer :: ios, nmat, ndone
   integer :: i, k, mrows, mcols, mnnz, ncols
   integer :: cuda_error
   integer :: row, col
   double precision :: vv
   integer, allocatable :: ira(:), jca(:)        ! coordinate (1-indexed, lower triangle)
   double precision, allocatable :: aa(:)
   integer(long), allocatable :: ptr(:)          ! CSC col_ptr (1-indexed, length n+1)
   integer, allocatable :: row_idx(:)            ! CSC row indices (1-indexed)
   double precision, allocatable :: val(:)       ! CSC values
   integer, allocatable :: col_count(:)
   double precision, allocatable :: bb(:), xx(:), axbuf(:)
   double precision :: res_norm, rhs_norm, rel_res
   integer(8) :: t_fac_start, t_fac_end, t_sol_start, t_sol_end
   integer(8) :: clock_rate, clock_max
   integer(8) :: fac_us, sol_us
   logical :: have_failure
   integer :: pos_count, neg_count, zero_count

   if (command_argument_count() < 1) then
      write(0,*) "Usage: ssids_bench <manifest.txt>"
      stop 2
   end if
   call get_command_argument(1, manifest_path)

   call system_clock(count_rate=clock_rate, count_max=clock_max)

   ! Suppress SSIDS console output (we capture results via inform)
   options%print_level = -1
   options%action = .true.   ! continue on singular matrices, like ForceAccept

   open(unit=10, file=trim(manifest_path), status='old', &
        action='read', iostat=ios)
   if (ios /= 0) then
      write(0,*) "Cannot open manifest: ", trim(manifest_path)
      stop 4
   end if

   nmat = 0
   ndone = 0
100 continue
   read(10, '(A)', iostat=ios) line
   if (ios /= 0) goto 999
   if (len_trim(line) == 0) goto 100

   nmat = nmat + 1
   call parse_line(line, mtx_path, rhs_path, out_path)
   have_failure = .false.
   write(0,'(A,A)') "BEGIN ", trim(mtx_path)
   flush(0)

   ! ---- Read MTX ----
   call read_mtx(mtx_path, mrows, mcols, mnnz, ira, jca, aa, ios)
   if (ios /= 0) then
      write(0,*) "READ_MTX failed: ", trim(mtx_path)
      have_failure = .true.
      goto 800
   end if

   ! ---- Read RHS ----
   allocate(bb(mrows))
   open(unit=11, file=trim(rhs_path), status='old', action='read', iostat=ios)
   if (ios /= 0) then
      write(0,*) "Cannot open rhs: ", trim(rhs_path)
      have_failure = .true.
      goto 700
   end if
   do i = 1, mrows
      read(11, *, iostat=ios) bb(i)
      if (ios /= 0) then
         write(0,*) "rhs read failed at row ", i
         have_failure = .true.
         close(11)
         goto 700
      end if
   end do
   close(11)

   ! ---- Convert COO (1-indexed lower triangle) to CSC ----
   ncols = mrows
   allocate(col_count(ncols))
   col_count = 0
   do k = 1, mnnz
      col_count(jca(k)) = col_count(jca(k)) + 1
   end do
   allocate(ptr(ncols + 1))
   ptr(1) = 1
   do i = 1, ncols
      ptr(i + 1) = ptr(i) + col_count(i)
   end do
   allocate(row_idx(mnnz))
   allocate(val(mnnz))
   ! Re-use col_count as a per-column write cursor
   col_count = 0
   do k = 1, mnnz
      col = jca(k)
      i = int(ptr(col) + col_count(col))
      row_idx(i) = ira(k)
      val(i) = aa(k)
      col_count(col) = col_count(col) + 1
   end do
   deallocate(col_count)

   ! ---- Analyse + Factor ----
   call system_clock(t_fac_start)
   call ssids_analyse(.true., ncols, ptr, row_idx, akeep, options, inform)
   if (inform%flag < 0) then
      write(0,*) "ssids_analyse failed on ", trim(mtx_path), " flag=", inform%flag
      call hint_omp_cancellation(inform%flag)
      have_failure = .true.
      goto 600
   end if
   call ssids_factor(.false., val, akeep, fkeep, options, inform)
   call system_clock(t_fac_end)
   if (inform%flag < 0) then
      write(0,*) "ssids_factor failed on ", trim(mtx_path), " flag=", inform%flag
      call hint_omp_cancellation(inform%flag)
      have_failure = .true.
      goto 600
   end if

   ! ---- Solve ----
   allocate(xx(mrows))
   xx(:) = bb(:)
   call system_clock(t_sol_start)
   call ssids_solve(xx, akeep, fkeep, options, inform)
   call system_clock(t_sol_end)
   if (inform%flag < 0) then
      write(0,*) "ssids_solve failed on ", trim(mtx_path), " flag=", inform%flag
      have_failure = .true.
      goto 550
   end if

   ! ---- Compute residual ||A·x − b|| / ||b|| using the original COO ----
   allocate(axbuf(mrows))
   axbuf = 0.0d0
   do k = 1, mnnz
      row = ira(k)
      col = jca(k)
      vv = aa(k)
      axbuf(row) = axbuf(row) + vv * xx(col)
      if (row /= col) then
         axbuf(col) = axbuf(col) + vv * xx(row)
      end if
   end do
   res_norm = 0.0d0
   rhs_norm = 0.0d0
   do i = 1, mrows
      res_norm = res_norm + (axbuf(i) - bb(i))**2
      rhs_norm = rhs_norm + bb(i)**2
   end do
   if (rhs_norm > 0.0d0) then
      rel_res = sqrt(res_norm / rhs_norm)
   else
      rel_res = sqrt(res_norm)
   end if
   deallocate(axbuf)

   fac_us = ((t_fac_end - t_fac_start) * 1000000_8) / clock_rate
   sol_us = ((t_sol_end - t_sol_start) * 1000000_8) / clock_rate

   ! Inertia derivation:
   !   negative = inform%num_neg
   !   zero     = n - inform%matrix_rank   (rank deficiency)
   !   positive = matrix_rank - num_neg
   neg_count = inform%num_neg
   zero_count = mrows - inform%matrix_rank
   pos_count = inform%matrix_rank - neg_count

   ! ---- Write result ----
   open(unit=12, file=trim(out_path), status='replace', action='write', iostat=ios)
   if (ios /= 0) then
      write(0,*) "Cannot write out: ", trim(out_path)
      have_failure = .true.
      goto 500
   end if
   write(12,'(A)') "solver ssids"
   write(12,'(A,I0)') "n ", mrows
   write(12,'(A,I0)') "nnz ", mnnz
   write(12,'(A,I0)') "inertia_pos ", pos_count
   write(12,'(A,I0)') "inertia_neg ", neg_count
   write(12,'(A,I0)') "inertia_zero ", zero_count
   write(12,'(A,I0)') "factor_us ", fac_us
   write(12,'(A,I0)') "solve_us ", sol_us
   write(12,'(A,ES24.16E3)') "residual ", rel_res
   write(12,'(A,I0)') "ssids_flag ", inform%flag
   write(12,'(A,I0)') "matrix_rank ", inform%matrix_rank
   write(12,'(A,I0)') "num_delay ", inform%num_delay
   ! num_factor = entries in factors after numerical factorization;
   ! source of truth for feral / SSIDS fill parity. Reported as
   ! integer(long) in SPRAL; we let Fortran I0 widen it.
   write(12,'(A,I0)') "num_factor ", inform%num_factor
   write(12,'(A)') "status ok"
   close(12)
   ndone = ndone + 1

500 continue
550 continue
   if (allocated(xx)) deallocate(xx)
600 continue
   call ssids_free(akeep, fkeep, cuda_error)
   if (allocated(ptr)) deallocate(ptr)
   if (allocated(row_idx)) deallocate(row_idx)
   if (allocated(val)) deallocate(val)
700 continue
   if (allocated(bb)) deallocate(bb)
800 continue
   if (allocated(ira)) deallocate(ira)
   if (allocated(jca)) deallocate(jca)
   if (allocated(aa)) deallocate(aa)

   if (have_failure) then
      open(unit=12, file=trim(out_path), status='replace', action='write')
      write(12,'(A)') "solver ssids"
      write(12,'(A)') "status fail"
      close(12)
   end if

   goto 100
999 continue
   close(10)

   write(0,'(A,I0,A,I0,A)') "ssids_bench done: ", ndone, " / ", &
        nmat, " matrices succeeded"
   stop 0

contains

   ! SSIDS flag -53 (SSIDS_ERROR_OMP_CANCELLATION) means the SSIDS CPU
   ! code found OMP cancellation disabled. It can only be enabled via the
   ! OMP_CANCELLATION environment variable, which must be set *before*
   ! the process starts — a Fortran program cannot set it for itself.
   ! run_ssids.py exports it; a direct invocation must too. Print an
   ! actionable hint instead of leaving the bare flag number cryptic.
   subroutine hint_omp_cancellation(flag)
      integer, intent(in) :: flag
      if (flag == -53) then
         write(0,'(A)') "  hint: flag -53 = SSIDS_ERROR_OMP_CANCELLATION; " // &
            "re-run with OMP_CANCELLATION=true set in the environment"
      end if
   end subroutine hint_omp_cancellation

   subroutine parse_line(line_in, p1, p2, p3)
      character(len=*), intent(in) :: line_in
      character(len=*), intent(out) :: p1, p2, p3
      integer :: i, s, l
      l = len_trim(line_in)
      i = 1
      do while (i <= l .and. line_in(i:i) == ' ')
         i = i + 1
      end do
      s = i
      do while (i <= l .and. line_in(i:i) /= ' ')
         i = i + 1
      end do
      p1 = line_in(s:i-1)
      do while (i <= l .and. line_in(i:i) == ' ')
         i = i + 1
      end do
      s = i
      do while (i <= l .and. line_in(i:i) /= ' ')
         i = i + 1
      end do
      p2 = line_in(s:i-1)
      do while (i <= l .and. line_in(i:i) == ' ')
         i = i + 1
      end do
      s = i
      do while (i <= l .and. line_in(i:i) /= ' ')
         i = i + 1
      end do
      p3 = line_in(s:i-1)
   end subroutine parse_line

   subroutine read_mtx(path, nr, nc, nnz_out, irn, jcn, vals, ios_out)
      character(len=*), intent(in) :: path
      integer, intent(out) :: nr, nc, nnz_out
      integer, allocatable, intent(out) :: irn(:), jcn(:)
      double precision, allocatable, intent(out) :: vals(:)
      integer, intent(out) :: ios_out
      character(len=1024) :: hdr
      integer :: kk

      open(unit=20, file=trim(path), status='old', action='read', iostat=ios_out)
      if (ios_out /= 0) return

      read(20, '(A)', iostat=ios_out) hdr
      if (ios_out /= 0) then
         close(20)
         return
      end if

200   continue
      read(20, '(A)', iostat=ios_out) hdr
      if (ios_out /= 0) then
         close(20)
         return
      end if
      if (hdr(1:1) == '%') goto 200

      read(hdr, *, iostat=ios_out) nr, nc, nnz_out
      if (ios_out /= 0) then
         close(20)
         return
      end if

      allocate(irn(nnz_out))
      allocate(jcn(nnz_out))
      allocate(vals(nnz_out))

      do kk = 1, nnz_out
         read(20, *, iostat=ios_out) irn(kk), jcn(kk), vals(kk)
         if (ios_out /= 0) then
            close(20)
            return
         end if
      end do

      close(20)
      ios_out = 0
   end subroutine read_mtx

end program ssids_bench
