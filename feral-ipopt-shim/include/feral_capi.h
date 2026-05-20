/* feral C ABI — minimum POC surface for Ipopt's
 * SparseSymLinearSolverInterface plug-in shape.
 *
 * Status codes mirror Ipopt's ESymSolverStatus enum
 * (IpSymLinearSolver.hpp:19-33). Matrix format matches
 * Ipopt's CSR_Format_0_Offset (upper-triangle CSR,
 * 0-based, sorted/deduplicated within row), which is
 * byte-identical to feral's CscMatrix layout for
 * symmetric matrices.
 *
 * Hand-written for POC. Will be cbindgen-generated once
 * the ABI stabilizes.
 */
#ifndef FERAL_CAPI_H
#define FERAL_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
#define FERAL_SUCCESS        0
#define FERAL_SINGULAR       1
#define FERAL_WRONG_INERTIA  2
#define FERAL_FATAL          3

/* Opaque handle. */
typedef struct FeralSolver FeralSolver;

/* Lifecycle. */
FeralSolver* feral_new(void);
void         feral_free(FeralSolver* s);

/* Structure phase. ia has length n+1; ja has length nnz. */
int     feral_set_structure(FeralSolver* s, int n, int nnz,
                            const int* ia, const int* ja);
double* feral_values_ptr(FeralSolver* s);

/* Numerical phase. */
int feral_factor(FeralSolver* s, int check_neg, int expected_neg);
int feral_solve(FeralSolver* s, int nrhs, double* rhs);

/* Query. */
int feral_num_neg(const FeralSolver* s);

/* Near-singularity signal — the analog of MA57's CNTL(2) small-pivot
 * threshold. feral_min_pivot returns min|lambda(D)|, the smallest
 * accepted pivot magnitude over every 1x1/2x2 D block; feral_max_pivot
 * returns the largest. Both return -1.0 when no factor is available or
 * s is NULL. A perturbation handler thresholds the scale-free ratio
 * feral_min_pivot / feral_max_pivot to detect a near-singular system
 * even when feral_num_neg reports the correct inertia. */
double feral_min_pivot(const FeralSolver* s);
double feral_max_pivot(const FeralSolver* s);

#ifdef __cplusplus
}
#endif

#endif /* FERAL_CAPI_H */
