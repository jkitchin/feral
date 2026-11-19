# Plan — issue #237 step 3: parallel KR sweeps

Research: `dev/research/issue-237-parallel-kr-sweeps-2026-11-19.md`.

1. Harness reports `scaling_us` (done): baseline 204–284 ms.
2. Tests first: parallel == serial bitwise at T ∈ {1,2,3,4,7} on an
   arrow + banded matrix above the threshold, with zero rows and
   explicit zeros.
3. Split `compute_infnorm` into the serial reference pass and a
   chunked parallel pass behind the rayon-worker + nnz gate.
4. `cargo test --release`; harness digests unchanged at both sizes.
5. Measure scaling_us and factor wall, all driver modes.
