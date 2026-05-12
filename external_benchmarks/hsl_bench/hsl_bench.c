/*
 * hsl_bench.c -- HSL MA97 oracle for the feral comparison benchmark.
 *
 * Reads a manifest file of (mtx_path, rhs_path, out_path) triples
 * (whitespace-separated). For each matrix:
 *   1. Reads the MatrixMarket file (symmetric, coordinate, real).
 *   2. Converts to lower-triangle CSC.
 *   3. Reads RHS from rhs_path (one f64 per line).
 *   4. Calls ma97_analyse + ma97_factor + ma97_solve with MC64 scaling
 *      and METIS ordering (control.ordering = 5 = auto AMD/METIS).
 *   5. Computes rel_res = ||A x - b||_2 / ||b||_2.
 *   6. Writes per-key text output to out_path. Schema mirrors
 *      external_benchmarks/mumps_oracle/mumps_bench.F.
 *
 * Build: see Makefile (links libhsl + libgfortran + libopenblas).
 * Usage: ./hsl_bench manifest.txt
 *
 * Manifest line: <mtx_path> <rhs_path> <out_path>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "hsl_ma97d.h"

/* Microsecond wall clock via clock_gettime(MONOTONIC). */
static long long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

/* Triplet (row, col, val). */
typedef struct { int row, col; double val; } trip_t;

static int trip_cmp_col(const void *a, const void *b) {
    const trip_t *ta = (const trip_t *)a;
    const trip_t *tb = (const trip_t *)b;
    if (ta->col != tb->col) return ta->col - tb->col;
    return ta->row - tb->row;
}

/* Read MatrixMarket symmetric coordinate file. Returns 0 on success.
 * Allocates row/col/val arrays; caller frees. Stores only lower
 * triangle (i >= j). */
static int read_mtx_lower(const char *path, int *n_out, int *nnz_out,
                          int **rows_out, int **cols_out, double **vals_out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[4096];
    int is_symmetric = 0;
    int header_seen = 0;
    int n = 0, m = 0, nnz_decl = 0;
    int *rows = NULL, *cols = NULL;
    double *vals = NULL;
    int k = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%') {
            if (strstr(line, "symmetric")) is_symmetric = 1;
            continue;
        }
        if (!header_seen) {
            if (sscanf(line, "%d %d %d", &m, &n, &nnz_decl) != 3) { fclose(f); return -2; }
            if (m != n) { fclose(f); return -3; }
            rows = (int *)malloc(sizeof(int) * (size_t)nnz_decl);
            cols = (int *)malloc(sizeof(int) * (size_t)nnz_decl);
            vals = (double *)malloc(sizeof(double) * (size_t)nnz_decl);
            if (!rows || !cols || !vals) { fclose(f); return -4; }
            header_seen = 1;
            continue;
        }
        int r, c;
        double v;
        if (sscanf(line, "%d %d %lf", &r, &c, &v) != 3) continue;
        r -= 1; c -= 1;  /* 0-indexed */
        /* Keep only lower triangle (or diag). */
        if (r < c) {
            int t = r; r = c; c = t;
        }
        rows[k] = r; cols[k] = c; vals[k] = v;
        k++;
    }
    fclose(f);
    if (!is_symmetric) {
        free(rows); free(cols); free(vals);
        return -5;
    }
    *n_out = n;
    *nnz_out = k;
    *rows_out = rows; *cols_out = cols; *vals_out = vals;
    return 0;
}

/* Convert triplets to CSC (column-major), lower triangle only.
 * Sorts triplets in place. ptr has length n+1; row + val have length nnz. */
static int trips_to_csc(int n, int nnz, int *trip_rows, int *trip_cols, double *trip_vals,
                        int **ptr_out, int **row_out, double **val_out) {
    trip_t *t = (trip_t *)malloc(sizeof(trip_t) * (size_t)nnz);
    if (!t) return -1;
    for (int i = 0; i < nnz; i++) {
        t[i].row = trip_rows[i];
        t[i].col = trip_cols[i];
        t[i].val = trip_vals[i];
    }
    qsort(t, (size_t)nnz, sizeof(trip_t), trip_cmp_col);
    int *ptr = (int *)calloc((size_t)(n + 1), sizeof(int));
    int *row = (int *)malloc(sizeof(int) * (size_t)nnz);
    double *val = (double *)malloc(sizeof(double) * (size_t)nnz);
    if (!ptr || !row || !val) { free(t); return -2; }
    for (int i = 0; i < nnz; i++) {
        ptr[t[i].col + 1]++;
    }
    for (int j = 0; j < n; j++) ptr[j + 1] += ptr[j];
    for (int i = 0; i < nnz; i++) {
        row[i] = t[i].row;
        val[i] = t[i].val;
    }
    free(t);
    *ptr_out = ptr; *row_out = row; *val_out = val;
    return 0;
}

/* Read RHS file: one f64 per line, exactly n values.  Returns 0 on
 * success, nonzero on error. */
static int read_rhs(const char *path, int n, double *b) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int k = 0;
    while (fgets(line, sizeof(line), f)) {
        if (k >= n) { fclose(f); return -2; }
        if (sscanf(line, "%lf", &b[k]) != 1) { fclose(f); return -3; }
        k++;
    }
    fclose(f);
    if (k != n) return -4;
    return 0;
}

/* Residual ||A x - b||_2 / ||b||_2 using lower-triangle CSC. */
static double rel_res_2norm(int n, const int *ptr, const int *row, const double *val,
                            const double *x, const double *b) {
    double *r = (double *)calloc((size_t)n, sizeof(double));
    if (!r) return NAN;
    for (int i = 0; i < n; i++) r[i] = -b[i];
    for (int j = 0; j < n; j++) {
        for (int p = ptr[j]; p < ptr[j + 1]; p++) {
            int i = row[p];
            double a = val[p];
            r[i] += a * x[j];
            if (i != j) r[j] += a * x[i];
        }
    }
    double rn = 0.0, bn = 0.0;
    for (int i = 0; i < n; i++) {
        rn += r[i] * r[i];
        bn += b[i] * b[i];
    }
    free(r);
    if (bn == 0.0) return 0.0;
    return sqrt(rn / bn);
}

static int solve_one(const char *mtx_path, const char *rhs_path, const char *out_path) {
    int n = 0, nnz_trip = 0;
    int *trip_rows = NULL, *trip_cols = NULL;
    double *trip_vals = NULL;
    int rc = read_mtx_lower(mtx_path, &n, &nnz_trip,
                            &trip_rows, &trip_cols, &trip_vals);
    FILE *out = fopen(out_path, "w");
    if (!out) {
        free(trip_rows); free(trip_cols); free(trip_vals);
        return -1;
    }
    fprintf(out, "solver ma97-2.8.1\n");
    if (rc != 0) {
        fprintf(out, "status fail\n");
        fprintf(out, "fail_reason read_mtx_rc_%d\n", rc);
        fclose(out);
        return -1;
    }

    int *ptr = NULL, *cscrow = NULL;
    double *cscval = NULL;
    rc = trips_to_csc(n, nnz_trip, trip_rows, trip_cols, trip_vals,
                      &ptr, &cscrow, &cscval);
    free(trip_rows); free(trip_cols); free(trip_vals);
    if (rc != 0) {
        fprintf(out, "status fail\nfail_reason csc_rc_%d\n", rc);
        fclose(out); return -1;
    }
    int nnz = ptr[n];
    fprintf(out, "n %d\n", n);
    fprintf(out, "nnz %d\n", nnz);

    /* Read RHS. */
    double *b = (double *)malloc(sizeof(double) * (size_t)n);
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    if (!b || !x) {
        fprintf(out, "status fail\nfail_reason alloc\n");
        fclose(out); free(ptr); free(cscrow); free(cscval);
        free(b); free(x);
        return -1;
    }
    int rc_rhs = read_rhs(rhs_path, n, b);
    if (rc_rhs != 0) {
        fprintf(out, "status fail\nfail_reason rhs_rc_%d\n", rc_rhs);
        fclose(out); free(ptr); free(cscrow); free(cscval);
        free(b); free(x);
        return -1;
    }
    memcpy(x, b, sizeof(double) * (size_t)n);  /* x = b for in-place solve */

    /* MA97 control. */
    struct ma97_control_d control;
    struct ma97_info_d info;
    ma97_default_control_d(&control);
    control.f_arrays = 0;          /* 0-indexed C arrays */
    control.ordering = 5;          /* auto AMD/METIS */
    control.scaling = 1;           /* MC64 (best for indefinite KKT) */
    control.print_level = -1;
    control.unit_error = -1;
    control.unit_warning = -1;
    control.unit_diagnostics = -1;
    control.action = 1;            /* continue past singularity */

    void *akeep = NULL, *fkeep = NULL;
    int *order = NULL;             /* MA97 fills if requested */

    /* Analyse */
    long long t0 = now_us();
    ma97_analyse_d(1, n, ptr, cscrow, cscval, &akeep, &control, &info, order);
    long long t1 = now_us();
    if (info.flag < 0) {
        fprintf(out, "status fail\nfail_reason analyse_flag_%d\n", info.flag);
        ma97_finalise_d(&akeep, &fkeep);
        fclose(out);
        free(ptr); free(cscrow); free(cscval);
        free(b); free(x);
        return -1;
    }
    long long analyse_us = t1 - t0;

    /* Factor */
    int matrix_type = 4;  /* real indefinite */
    t0 = now_us();
    ma97_factor_d(matrix_type, ptr, cscrow, cscval,
                  &akeep, &fkeep, &control, &info, NULL);
    t1 = now_us();
    long long factor_us = t1 - t0;
    if (info.flag < 0) {
        fprintf(out, "status fail\nfail_reason factor_flag_%d\n", info.flag);
        fprintf(out, "analyse_us %lld\n", analyse_us);
        ma97_finalise_d(&akeep, &fkeep);
        fclose(out);
        free(ptr); free(cscrow); free(cscval);
        free(b); free(x);
        return -1;
    }
    int num_neg = info.num_neg;
    int matrix_rank = info.matrix_rank;
    long num_factor = info.num_factor;

    /* Solve + iterative refinement.
     *
     * MA97 does not provide a built-in residual-based refinement
     * entry point (ma97_solve_fredholm targets singular systems via
     * the Fredholm alternative, not refinement of a non-singular
     * solve). We implement Richardson iteration in C around
     * ma97_solve_d to match what MUMPS gets from ICNTL(10)=2 and
     * feral gets from solve_sparse_refined: factor-then-refine until
     * residual stops dropping.
     *
     * Stopping rule: max 4 steps; exit early if ||r||/||b|| stops
     * dropping or reaches 10 * machine_eps. Matches feral's
     * stagnation-based termination in spirit.
     */
    int max_refine = 4;
    int n_refine = 0;
    t0 = now_us();
    ma97_solve_d(0, 1, x, n, &akeep, &fkeep, &control, &info);
    if (info.flag < 0) {
        t1 = now_us();
        long long solve_us = t1 - t0;
        fprintf(out, "status fail\nfail_reason solve_flag_%d\n", info.flag);
        fprintf(out, "analyse_us %lld\n", analyse_us);
        fprintf(out, "factor_us %lld\n", factor_us);
        fprintf(out, "solve_us %lld\n", solve_us);
        ma97_finalise_d(&akeep, &fkeep);
        fclose(out);
        free(ptr); free(cscrow); free(cscval);
        free(b); free(x);
        return -1;
    }
    /* Refinement loop */
    double *r = (double *)calloc((size_t)n, sizeof(double));
    double *dx = (double *)calloc((size_t)n, sizeof(double));
    double bn2 = 0.0;
    for (int i = 0; i < n; i++) bn2 += b[i] * b[i];
    double bn = sqrt(bn2);
    double prev_rn = HUGE_VAL;
    if (r && dx && bn > 0.0) {
        for (int step = 0; step < max_refine; step++) {
            /* r = b - A x */
            for (int i = 0; i < n; i++) r[i] = b[i];
            for (int j = 0; j < n; j++) {
                for (int p = ptr[j]; p < ptr[j + 1]; p++) {
                    int i = cscrow[p];
                    double a = cscval[p];
                    r[i] -= a * x[j];
                    if (i != j) r[j] -= a * x[i];
                }
            }
            double rn2 = 0.0;
            for (int i = 0; i < n; i++) rn2 += r[i] * r[i];
            double rn = sqrt(rn2);
            if (rn / bn < 1e-15) break;
            if (rn >= prev_rn) break;   /* stagnation */
            prev_rn = rn;
            /* dx = A^{-1} r */
            for (int i = 0; i < n; i++) dx[i] = r[i];
            ma97_solve_d(0, 1, dx, n, &akeep, &fkeep, &control, &info);
            if (info.flag < 0) break;
            for (int i = 0; i < n; i++) x[i] += dx[i];
            n_refine++;
        }
    }
    free(r); free(dx);
    /* The CSC arrays alias the matrix used by MA97 internally and
     * must not be freed before this point. */
    t1 = now_us();
    long long solve_us = t1 - t0;

    double rel = rel_res_2norm(n, ptr, cscrow, cscval, x, b);
    int num_pos = matrix_rank - num_neg;
    int num_zero = n - matrix_rank;

    fprintf(out, "inertia_pos %d\n", num_pos);
    fprintf(out, "inertia_neg %d\n", num_neg);
    fprintf(out, "inertia_zero %d\n", num_zero);
    fprintf(out, "analyse_us %lld\n", analyse_us);
    fprintf(out, "factor_us %lld\n", factor_us);
    fprintf(out, "solve_us %lld\n", solve_us);
    fprintf(out, "nnz_l %ld\n", num_factor);
    fprintf(out, "rel_res %.17e\n", rel);
    fprintf(out, "refined yes\n");
    fprintf(out, "n_refine_steps %d\n", n_refine);
    fprintf(out, "status ok\n");
    fclose(out);

    ma97_finalise_d(&akeep, &fkeep);
    free(ptr); free(cscrow); free(cscval);
    free(b); free(x);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s manifest.txt\n", argv[0]);
        return 2;
    }
    FILE *m = fopen(argv[1], "r");
    if (!m) { perror("open manifest"); return 2; }
    char line[8192];
    int n_done = 0, n_fail = 0;
    while (fgets(line, sizeof(line), m)) {
        char mtx[4096], rhs[4096], out[4096];
        if (sscanf(line, "%4095s %4095s %4095s", mtx, rhs, out) != 3) continue;
        int rc = solve_one(mtx, rhs, out);
        if (rc != 0) n_fail++;
        n_done++;
        fprintf(stderr, "[%d] %s -> %s (rc=%d)\n", n_done, mtx, out, rc);
    }
    fclose(m);
    fprintf(stderr, "done %d (failures %d)\n", n_done, n_fail);
    return 0;
}
