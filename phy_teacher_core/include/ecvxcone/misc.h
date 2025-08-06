/*
 * Simplified CVXOPT MISC header for pure C use (no Python API)
 * Copyright stripped for simplification.
 */

#ifndef __MISC__
#define __MISC__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cvxopt.h"
#include "math.h"

/*
- dims['l'] = ml, the dimension of the nonnegative orthant C_0.
    (ml >= 0.)
- dims['q'] = mq = [ mq[0], mq[1], ..., mq[N-1] ], a list of N
    integers with the dimensions of the second order cones C_1, ...,
    C_N.  (N >= 0 and mq[k] >= 1.)
- dims['s'] = ms = [ ms[0], ms[1], ..., ms[M-1] ], a list of M
    integers with the orders of the semidefinite cones C_{N+1}, ...,
    C_{N+M}.  (M >= 0 and ms[k] >= 0.)
The default value of dims is {'l': G.size[0], 'q': [], 's': []}.
*/

typedef struct {
    int l;              // 'l' dimension
    int *q;             // 'q' dimension
    int q_size;         // size of q array
    int *s;             // 's' dimension
    int s_size;         // size of s array
} DIMs;

typedef struct {
    matrix *dnl;
    matrix *dnli;
    matrix *d;
    matrix *di;
    matrix **v;      // array of matrix pointers
    double *beta;    // array of doubles
    matrix **r;      // array of matrix pointers
    matrix **rti;    // array of matrix pointers
    int v_count;     // number of v matrices
    int r_count;     // number of r matrices
    int b_count;     // number of beta values
    int dnl_count;   // number of dnl/dnli matrices
    int d_count;     // number of d/di matrices
    int has_dnl;     // flag indicating if dnl/dnli entries exist
} scaling;

// Context structure for closures
typedef struct {
    void *G;
    int mnl;
    int p, n;
    int cdim;
    int cdim_pckd;
    matrix *QA;
    matrix *tauA;
    matrix *Gs;
    matrix *K;
    matrix *bzp;
    matrix *yy;
    scaling *W;  // Scaling structure
} KKTCholContext;

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

/* Error handling macros */
#define ERR(s)                  (fprintf(stderr, "%s\n", s), abort())
#define ERR_INT(s)              (fprintf(stderr, "%s\n", s), -1)
#define ERR_TYPE(s)             (fprintf(stderr, "TypeError: %s\n", s), abort())

#define err_mtrx(s)             ERR_TYPE(s " must be a matrix")
#define err_bool(s)             ERR_TYPE(s " must be true or false")
#define err_conflicting_ids     ERR_TYPE("conflicting types for matrix arguments")
#define err_invalid_id          ERR_TYPE("matrix arguments must have type 'd' or 'z'")
#define err_nz_int(s)           ERR_TYPE(s " must be a nonzero integer")
#define err_nn_int(s)           ERR_TYPE(s " must be a nonnegative integer")
#define err_buf_len(s)          ERR_TYPE("length of " s " is too small")
#define err_type(s)             ERR_TYPE("incompatible type for " s)
#define err_p_int(s)            ERR(s " must be a positive integer")
#define err_char(s1,s2)         ERR("possible values of " s1 " are: " s2)
#define err_ld(s)               ERR("illegal value of " s)
#define err_int_mtrx(s)         ERR_TYPE(s " must be a matrix with typecode 'i'")
#define err_dbl_mtrx(s)         ERR_TYPE(s " must be a matrix with typecode 'd'")
#define err_msk_noparam         "missing options dictionary"
#define err_no_memory           ERR("no memory use for matrix calculation")
#define err_bad_internal_call   ERR("Internal error: bad function call or argument usage.")
#define err_division_by_zero    ERR("division by zero, check if denominator is zero before dividing")

/* C-style cyclic wrap-around for indices */
#define CWRAP(i,m) ((i) >= 0 ? (i) : ((m)+(i)))
#define OUT_RNG(i, dim) ((i) < -(dim) || (i) >= (dim))

#define VALID_TC_MAT(t) ((t)=='i' || (t)=='d' || (t)=='z')
#define VALID_TC_SP(t)  ((t)=='d' || (t)=='z')
#define TC2ID(c) ((c)=='i' ? 0 : ((c)=='d' ? 1 : 2))

#define X_ID(O)    (Matrix_Check(O) ? MAT_ID(O)    : SP_ID(O))
#define X_NROWS(O) (Matrix_Check(O) ? MAT_NROWS(O) : SP_NROWS(O))
#define X_NCOLS(O) (Matrix_Check(O) ? MAT_NCOLS(O) : SP_NCOLS(O))
#define X_Matrix_Check(O) (Matrix_Check(O) || SpMatrix_Check(O))

#define len(x) (Matrix_Check(x) ? MAT_LGT(x) : SP_LGT(x))

/* Declarations of misc functions */
extern double misc_jnrm2(matrix* x, int n, int offset);
extern void misc_scale(matrix *x, scaling *W, char trans, char inverse);
extern void misc_scale2(matrix *lmbda, matrix *x, DIMs *dims, int mnl, char inverse);
extern void misc_symm(matrix *x, int n, int offset);
extern void misc_sprod(matrix *x, matrix *y, DIMs *dims, int mnl, char diag);
extern double misc_sdot(matrix *x, matrix *y, DIMs *dims, int mnl);
extern double misc_max_step(matrix* x, DIMs* dims, int mnl, matrix* sigma);
extern double misc_snrm2(matrix *x, DIMs *dims, int mnl);
extern void misc_ssqr(matrix *x, matrix *y, DIMs *dims, int mnl);
extern double misc_jdot(matrix* x, matrix* y, int n, int offsetx, int offsety);
extern scaling* misc_compute_scaling(matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int mnl);
extern void misc_update_scaling(scaling *W, matrix *lmbda, matrix *s, matrix *z);
extern void misc_compute_scaling2(scaling *W, matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int mnl);
extern void solve_function(matrix *x, matrix *y, matrix *z, KKTCholContext *ctx, DIMs *dims);
extern void factor_function(scaling *W, matrix *H, matrix *Df, KKTCholContext *ctx, DIMs *dims);
extern KKTCholContext* kkt_chol(void *G, DIMs *dims, void *A, int mnl);
extern void misc_pack(matrix *x, matrix *y, DIMs *dims, int mnl, int offsetx, int offsety);
extern void misc_pack2(matrix *x, DIMs *dims, int mnl);
extern void misc_unpack(matrix *x, matrix *y, DIMs *dims, int mnl, int offsetx, int offsety);
extern void misc_sgemv(void *A, matrix *x, matrix *y, DIMs *dims, char trans, double alpha, 
                double beta, int n, int offsetA, int offsetx, int offsety);
extern void misc_trisc(matrix *x, DIMs *dims, int offset);
extern void misc_triusc(matrix *x, DIMs *dims, int offset);
extern void misc_sinv(matrix *x, matrix *y, DIMs *dims, int mnl);

// Function pointer for converting numbers
extern void (*write_num[])(void *, int, void *, int);

// Scaling initialization function
static inline void Scaling_Init(scaling *W) {
    if (!W) err_no_memory;

    // Initialize all pointers to NULL
    W->dnl = NULL;
    W->dnli = NULL;
    W->d = NULL;
    W->di = NULL;
    W->v = NULL;
    W->beta = NULL;
    W->r = NULL;
    W->rti = NULL;

    // Initialize counts and flags
    W->v_count = 0;
    W->r_count = 0;
    W->b_count = 0;
    W->dnl_count = 0;
    W->d_count = 0;
    W->has_dnl = 0;
}

// Scaling cleanup function
static inline void Scaling_Free(scaling *W) {
    if (W) {
        if (W->dnl) Matrix_Free(W->dnl);
        if (W->dnli) Matrix_Free(W->dnli);
        if (W->d) Matrix_Free(W->d);   
        if (W->di) Matrix_Free(W->di);
        if (W->v) {
            for (int i = 0; i < W->v_count; i++) {
                if (W->v[i]) Matrix_Free(W->v[i]);
            }
            free(W->v);
        }
        if (W->beta) free(W->beta);
        if (W->r) {
            for (int i = 0; i < W->r_count; i++) {
                if (W->r[i]) Matrix_Free(W->r[i]);
            }
            free(W->r);
        }
        if (W->rti) {
            for (int i = 0; i < W->r_count; i++) {
                if (W->rti[i]) Matrix_Free(W->rti[i]);
            }
            free(W->rti);
        }
        free(W);
    }
}

// Initialize KKTCholContext structure
static inline void KKTCholContext_Init(KKTCholContext *ctx) {
    if (!ctx) err_no_memory;

    ctx->G = NULL;
    ctx->mnl = 0;
    ctx->p = 0;
    ctx->n = 0;
    ctx->cdim = 0;
    ctx->cdim_pckd = 0;
    ctx->QA = NULL;
    ctx->tauA = NULL;
    ctx->Gs = NULL;
    ctx->K = NULL;
    ctx->bzp = NULL;
    ctx->yy = NULL;
    ctx->W = NULL;  // Initialize scaling structure
}

static inline void KKTCholContext_Free(KKTCholContext *ctx) {
    if (ctx) {
        // if (ctx->G) Matrix_Free(ctx->G);
        // if (ctx->dims) {
        //     if (ctx->dims->q) free(ctx->dims->q);
        //     if (ctx->dims->s) free(ctx->dims->s);
        //     free(ctx->dims);
        // }
        if (ctx->QA) Matrix_Free(ctx->QA);
        if (ctx->tauA) Matrix_Free(ctx->tauA);
        if (ctx->Gs) Matrix_Free(ctx->Gs);
        if (ctx->K) Matrix_Free(ctx->K);
        if (ctx->bzp) Matrix_Free(ctx->bzp);
        if (ctx->yy) Matrix_Free(ctx->yy);
        // if (ctx->W) {
        //     Scaling_Free(ctx->W);  // Free scaling structure
        // }
        free(ctx);
        // Scaling_Free(ctx->W);  // Free scaling structure
    }
}


// Helper function to calculate sum of array elements
static inline int sum_array(int *arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += arr[i];
    }
    return sum;
}

// Helper function to calculate sum of squares of array elements
static inline int sum_square_array(int *arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += arr[i] * arr[i];
    }
    return sum;
}

// Helper function to calculate sum of triangular numbers in an array
static inline int sum_triangular_array(int *arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += arr[i] * (arr[i] + 1) / 2; // Sum of first n natural numbers
    }
    return sum;
}

// Helper function to calculate maximum of array elements
static inline int max_array(int *arr, int len) {
    int max_val = 0;
    for (int i = 0; i < len; ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

// Function to slice matrix like Python s[i:j] style, only supports dense DOUBLE vector
static inline matrix* matrix_slice(const matrix* m, int start, int end) {
    if (!m) {
        fprintf(stderr, "Error: NULL matrix pointer passed to matrix_slice_range.\n");
        return NULL;
    }

    if (m->mat_type != MAT_DENSE || m->id != DOUBLE) {
        fprintf(stderr, "Error: matrix_slice_range only supports dense matrices with DOUBLE type.\n");
        return NULL;
    }

    int total_len = m->nrows * m->ncols;

    if (start < 0 || end > total_len || start > end) {
        fprintf(stderr, "Error: Invalid slice range [%d:%d], total length is %d.\n", start, end, total_len);
        return NULL;
    }

    int len = end - start;

    matrix* result = (matrix*)malloc(sizeof(matrix));
    if (!result) {
        fprintf(stderr, "Error: Failed to allocate matrix struct in matrix_slice_range.\n");
        return NULL;
    }

    result->mat_type = MAT_DENSE;
    result->id = DOUBLE;
    result->nrows = len;
    result->ncols = 1;

    if (len == 0) {
        result->buffer = NULL; // Empty slice
        return result;
    }

    result->buffer = malloc(sizeof(double) * len);
    if (!result->buffer) {
        fprintf(stderr, "Error: Failed to allocate buffer in matrix_slice_range.\n");
        free(result);
        return NULL;
    }

    // Copy data
    double* dst = (double*)result->buffer;
    double* src = (double*)m->buffer;
    for (int i = 0; i < len; ++i)
        dst[i] = src[start + i];

    return result;
}

// Function to transpose a matrix
static inline matrix* matrix_transpose(matrix *m) {

  if (!m) {
    fprintf(stderr, "matrix_transpose: input matrix is NULL.\n");
    return NULL;
  }
  matrix *ret = Matrix_New(m->ncols, m->nrows, m->id);
  if (!ret) {
    fprintf(stderr, "matrix_transpose: failed to allocate transposed matrix.\n");
    return NULL;
  }

  int i, j, cnt = 0;
  for (i=0; i < ret->nrows; i++)
    for (j=0; j < ret->ncols; j++)
      write_num[m->id](ret->buffer, i + j*ret->nrows, m->buffer, cnt++);

  return ret;
}

// Matrix slice assignment
static inline void matrix_slice_assign(matrix *dst, const matrix *src,
                                       int start_row, int end_row,
                                       int start_col, int end_col) 
{
    int nrows = end_row - start_row;
    int ncols = end_col - start_col;

    if (start_row < 0 || end_row > dst->nrows ||
        start_col < 0 || end_col > dst->ncols ||
        start_row >= end_row || start_col >= end_col) {
        fprintf(stderr, "Error: invalid slice range (%d:%d, %d:%d) for dst of size (%d x %d).\n",
                start_row, end_row, start_col, end_col, dst->nrows, dst->ncols);
        return;
    }

    if (src->nrows != nrows || src->ncols != ncols) {
        fprintf(stderr, "Error: source matrix dimensions (%d x %d) do not match target slice size (%d x %d).\n",
                src->nrows, src->ncols, nrows, ncols);
        return;
    }

    if (dst->id != DOUBLE || src->id != DOUBLE) {
        fprintf(stderr, "Error: Only DOUBLE type supported in matrix_slice_assign.\n");
        return;
    }

    for (int j = 0; j < ncols; ++j) {
        for (int i = 0; i < nrows; ++i) {
            int dst_index = (i + start_row) + (j + start_col) * dst->nrows;
            int src_index = i + j * src->nrows;
            MAT_BUFD(dst)[dst_index] = MAT_BUFD(src)[src_index];
        }
    }
}


// Add B into A (in-place), A = A + B
static inline void matrix_add(matrix *A, const matrix *B) {
    if (!A || !B) {
        fprintf(stderr, "Error: NULL pointer passed to matrix_add.\n");
        return;
    }

    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        fprintf(stderr, "Error: matrix_add dimension mismatch (%d x %d vs %d x %d)\n",
                A->nrows, A->ncols, B->nrows, B->ncols);
        return;
    }

    int len = A->nrows * A->ncols;
    for (int i = 0; i < len; ++i) {
        ((double *)A->buffer)[i] += ((double *)B->buffer)[i];
    }
}

#endif
