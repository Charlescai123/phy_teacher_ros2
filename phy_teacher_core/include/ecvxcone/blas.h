#ifndef __BLAS__
#define __BLAS__

#include "cvxopt.h"
#include "misc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define USE_CBLAS_ZDOT 0


/* BLAS 1 prototypes */
extern void dswap_(int *n, double *x, int *incx, double *y, int *incy);
extern void zswap_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void zscal_(int *n, complex_t *alpha, complex_t *x, int *incx);
extern void zdscal_(int *n, double *alpha, complex_t *x, int *incx);
extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern void zcopy_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void daxpy_(int *n, double *alpha, double *x, int *incx,
    double *y, int *incy);
extern void zaxpy_(int *n, complex_t *alpha, complex_t *x, int *incx,
    complex_t *y, int *incy);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
#if USE_CBLAS_ZDOT
extern void cblas_zdotc_sub(int n, void *x, int incx, void *y,
    int incy, void *result);
extern void cblas_zdotu_sub(int n, void *x, int incx, void *y, int incy,
    void *result);
#endif
extern double dnrm2_(int *n, double *x, int *incx);
extern double dznrm2_(int *n, complex_t *x, int *incx);
extern double dasum_(int *n, double *x, int *incx);
extern double dzasum_(int *n, complex_t *x, int *incx);
extern int idamax_(int *n, double *x, int *incx);
extern int izamax_(int *n, complex_t *x, int *incx);


/* BLAS 2 prototypes */
extern void dgemv_(char* trans, int *m, int *n, double *alpha,
    double *A, int *lda, double *x, int *incx, double *beta, double *y,
    int *incy);
extern void zgemv_(char* trans, int *m, int *n, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    double *alpha, double *A, int *lda, double *x, int *incx,
    double *beta, double *y,  int *incy);
extern void zgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    complex_t *alpha, complex_t *A, int *lda, complex_t *x, int *incx,
    complex_t *beta, complex_t *y,  int *incy);
extern void dsymv_(char *uplo, int *n, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhemv_(char *uplo, int *n, complex_t *alpha, complex_t *A,
    int *lda, complex_t *x, int *incx, complex_t *beta, complex_t *y,
    int *incy);
extern void dsbmv_(char *uplo, int *n, int *k, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhbmv_(char *uplo, int *n, int *k, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dtrmv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrmv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtrsv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrsv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx,
    double *y, int *incy, double *A, int *lda);
extern void zgerc_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void zgeru_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void dsyr_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *A, int *lda);
extern void zher_(char *uplo, int *n, double *alpha, complex_t *x,
    int *incx, complex_t *A, int *lda);
extern void dsyr2_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *y, int *incy, double *A, int *lda);
extern void zher2_(char *uplo, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);


/* BLAS 3 prototypes */
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsymm_(char *side, char *uplo, int *m, int *n,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsymm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zhemm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsyrk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *beta, double *B,
    int *ldb);
extern void zsyrk_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *beta, complex_t *B,
    int *ldb);
extern void zherk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, complex_t *A, int *lda, double *beta, complex_t *B,
    int *ldb);
extern void dsyr2k_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsyr2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zher2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    double *beta, complex_t *C, int *ldc);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);
extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);

// Microsoft Visual C++ complex number support
#ifdef _MSC_VER
#define _Cbuild(r,i) ((double complex)((r) + (i)*I))
#endif

void blas_copy(matrix *x, matrix *y, int n, int ix, int iy, int ox, int oy);
void blas_axpy(matrix *x, matrix *y, number *alpha, int n, int incx, int incy, int offsetx, int offsety);
void blas_scal(void* alpha, matrix* x, int n, int inc, int offset);
double blas_nrm2(matrix *x, int n, int inc, int offset);
void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
          int n, int k, int ldA, int incx, int offsetA, int offsetx);
number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, int offsetx, int offsety);
void blas_tbmv(matrix *A, matrix *x, char uplo, char trans, char diag, 
          int n, int k, int ldA, int incx, int offsetA, int offsetx);
void blas_trmm(matrix *A, matrix *B, char side, char uplo, char transA, char diag,
               void* alpha, int m, int n, int ldA, int ldB, int offsetA, int offsetB);
void blas_trsm(matrix *A, matrix *B, char side, char uplo, char transA, char diag,
          void* alpha, int m, int n, int ldA, int ldB, int offsetA, int offsetB);
void blas_trsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
                int n, int ldA, int ix, int offsetA, int offsetx);
void blas_syrk(matrix *A, matrix *C, char uplo, char trans, void* alpha, void* beta, 
              int n, int k, int ldA, int ldC, int offsetA, int offsetC);
void blas_gemm(matrix *A, matrix *B, matrix *C, char transA, char transB, 
              void* alpha, void* beta, int m, int n, int k, int ldA, int ldB, 
              int ldC, int offsetA, int offsetB, int offsetC);
void blas_gemv(matrix *A, matrix *x, matrix *y, char trans, void* alpha, void* beta, 
            int m, int n, int ldA, int incx, int incy, int offsetA, int offsetx, int offsety);
            
#endif  // __BLAS__