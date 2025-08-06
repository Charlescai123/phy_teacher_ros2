#ifndef __LAPACK__
#define __LAPACK__

#include "cvxopt.h"
#include "misc.h"
#include <time.h>

// #ifndef _MSC_VER
// typedef complex double complex_t;
// #else
// typedef _Dcomplex complex_t;
// #endif

#define err_lapack(info) do { \
    if ((info) < 0) \
        fprintf(stderr, "LAPACK error: illegal argument at position %d\n", -(info)); \
    else \
        fprintf(stderr, "LAPACK error: computation failed at leading minor %d (matrix not positive definite?)\n", (info)); \
    return; \
} while (0)

/***********************************************************************/
/****                                                               ****/
/****                      LAPACK prototypes                        ****/
/****                                                               ****/
/***********************************************************************/
extern int ilaenv_(int  *ispec, char **name, char **opts, int *n1,
    int *n2, int *n3, int *n4);

extern void dlarfg_(int *n, double *alpha, double *x, int *incx, 
    double *tau);
extern void zlarfg_(int *n, complex_t *alpha, complex_t *x, 
    int *incx, complex_t *tau);
extern void dlarfx_(char *side, int *m, int *n, double *V, double *tau, 
    double *C, int *ldc, double *work); 
extern void zlarfx_(char *side, int *m, int *n, complex_t *V, 
    complex_t *tau, complex_t *C, int *ldc, 
    complex_t *work); 

extern void dlacpy_(char *uplo, int *m, int *n, double *A, int *lda,
    double *B, int *ldb);
extern void zlacpy_(char *uplo, int *m, int *n, complex_t *A, 
    int *lda, complex_t *B, int *ldb);

extern void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv,
    int *info);
extern void zgetrf_(int *m, int *n, complex_t *A, int *lda, int *ipiv,
    int *info);
extern void dgetrs_(char *trans, int *n, int *nrhs, double *A, int *lda,
    int *ipiv, double *B, int *ldb, int *info);
extern void zgetrs_(char *trans, int *n, int *nrhs, complex_t *A, 
    int *lda, int *ipiv, complex_t *B, int *ldb, int *info);
extern void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work,
    int *lwork, int *info);
extern void zgetri_(int *n, complex_t *A, int *lda, int *ipiv, 
    complex_t *work, int *lwork, int *info);
extern void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv,
    double *B, int *ldb, int *info);
extern void zgesv_(int *n, int *nrhs, complex_t *A, int *lda, 
    int *ipiv, complex_t *B, int *ldb, int *info);

extern void dgbtrf_(int *m, int *n, int *kl, int *ku, double *AB,
    int *ldab, int *ipiv, int *info);
extern void zgbtrf_(int *m, int *n, int *kl, int *ku, complex_t *AB,
    int *ldab, int *ipiv, int *info);
extern void dgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs,
    double *AB, int *ldab, int *ipiv, double *B, int *ldB, int *info);
extern void zgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs,
    complex_t *AB, int *ldab, int *ipiv, complex_t *B, 
    int *ldB, int *info);
extern void dgbsv_(int *n, int *kl, int *ku, int *nrhs, double *ab,
    int *ldab, int *ipiv, double *b, int *ldb, int *info);
extern void zgbsv_(int *n, int *kl, int *ku, int *nrhs, complex_t *ab,
    int *ldab, int *ipiv, complex_t *b, int *ldb, int *info);

extern void dgttrf_(int *n, double *dl, double *d, double *du,
    double *du2, int *ipiv, int *info);
extern void zgttrf_(int *n, complex_t *dl, complex_t *d, 
    complex_t *du, complex_t *du2, int *ipiv, int *info);
extern void dgttrs_(char *trans, int *n, int *nrhs, double *dl, double *d,
    double *du, double *du2, int *ipiv, double *B, int *ldB, int *info);
extern void zgttrs_(char *trans, int *n, int *nrhs, complex_t *dl,
    complex_t *d, complex_t *du, complex_t *du2, 
    int *ipiv, complex_t *B, int *ldB, int *info);
extern void dgtsv_(int *n, int *nrhs, double *dl, double *d, double *du,
    double *B, int *ldB, int *info);
extern void zgtsv_(int *n, int *nrhs, complex_t *dl, 
    complex_t *d, complex_t *du, complex_t *B, int *ldB, 
    int *info);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
extern void zpotrf_(char *uplo, int *n, complex_t *A, int *lda, 
    int *info);
extern void dpotrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    double *B, int *ldb, int *info);
extern void zpotrs_(char *uplo, int *n, int *nrhs, complex_t *A, 
    int *lda, complex_t *B, int *ldb, int *info);
extern void dpotri_(char *uplo, int *n, double *A, int *lda, int *info);
extern void zpotri_(char *uplo, int *n, complex_t *A, int *lda, 
    int *info);
extern void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    double *B, int *ldb, int *info);
extern void zposv_(char *uplo, int *n, int *nrhs, complex_t *A, 
    int *lda, complex_t *B, int *ldb, int *info);

extern void dpbtrf_(char *uplo, int *n, int *kd, double *AB, int *ldab,
    int *info);
extern void zpbtrf_(char *uplo, int *n, int *kd, complex_t *AB, 
    int *ldab, int *info);
extern void dpbtrs_(char *uplo, int *n, int *kd, int *nrhs, double *AB,
    int *ldab, double *B, int *ldb, int *info);
extern void zpbtrs_(char *uplo, int *n, int *kd, int *nrhs, 
    complex_t *AB, int *ldab, complex_t *B, int *ldb, int *info);
extern void dpbsv_(char *uplo, int *n, int *kd, int *nrhs, double *A,
    int *lda, double *B, int *ldb, int *info);
extern void zpbsv_(char *uplo, int *n, int *kd, int *nrhs, 
    complex_t *A, int *lda, complex_t *B, int *ldb, int *info);

extern void dpttrf_(int *n, double *d, double *e, int *info);
extern void zpttrf_(int *n, double *d, complex_t *e, int *info);
extern void dpttrs_(int *n, int *nrhs, double *d, double *e, double *B,
    int *ldB, int *info);
extern void zpttrs_(char *uplo, int *n, int *nrhs, double *d, 
    complex_t *e, complex_t *B, int *ldB, int *info);
extern void dptsv_(int *n, int *nrhs, double *d, double *e, double *B,
    int *ldB, int *info);
extern void zptsv_(int *n, int *nrhs, double *d, complex_t *e, 
    complex_t *B, int *ldB, int *info);

extern void dsytrf_(char *uplo, int *n, double *A, int *lda, int *ipiv,
    double *work, int *lwork, int *info);
extern void zsytrf_(char *uplo, int *n, complex_t *A, int *lda, 
    int *ipiv, complex_t *work, int *lwork, int *info);
extern void zhetrf_(char *uplo, int *n, complex_t *A, int *lda, 
    int *ipiv, complex_t *work, int *lwork, int *info);
extern void dsytrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    int *ipiv, double *B, int *ldb, int *info);
extern void zsytrs_(char *uplo, int *n, int *nrhs, complex_t *A, 
    int *lda, int *ipiv, complex_t *B, int *ldb, int *info);
extern void zhetrs_(char *uplo, int *n, int *nrhs, complex_t *A, 
    int *lda, int *ipiv, complex_t *B, int *ldb, int *info);
extern void dsytri_(char *uplo, int *n, double *A, int *lda, int *ipiv,
    double *work, int *info);
extern void zsytri_(char *uplo, int *n, complex_t *A, int *lda, 
    int *ipiv, complex_t *work, int *info);
extern void zhetri_(char *uplo, int *n, complex_t *A, int *lda, 
    int *ipiv, complex_t *work, int *info);
extern void dsysv_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    int *ipiv, double *B, int *ldb, double *work, int *lwork,
    int *info);
extern void zsysv_(char *uplo, int *n, int *nrhs, complex_t *A, 
    int *lda, int *ipiv, complex_t *B, int *ldb, 
    complex_t *work, int *lwork, int *info);
extern void zhesv_(char *uplo, int *n, int *nrhs, complex_t *A, 
    int *lda, int *ipiv, complex_t *B, int *ldb, 
    complex_t *work, int *lwork, int *info);

extern void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs,
    double  *a, int *lda, double *b, int *ldb, int *info);
extern void ztrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs,
    complex_t  *a, int *lda, complex_t *b, int *ldb, int *info);
extern void dtrtri_(char *uplo, char *diag, int *n, double  *a, int *lda,
    int *info);
extern void ztrtri_(char *uplo, char *diag, int *n, complex_t  *a, 
    int *lda, int *info);
extern void dtbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd,
    int *nrhs, double *ab, int *ldab, double *b, int *ldb, int *info);
extern void ztbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd,
    int *nrhs, complex_t *ab, int *ldab, complex_t *b, 
    int *ldb, int *info);

extern void dgels_(char *trans, int *m, int *n, int *nrhs, double *a,
    int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
extern void zgels_(char *trans, int *m, int *n, int *nrhs, 
    complex_t *a, int *lda, complex_t *b, int *ldb, 
    complex_t *work, int *lwork, int *info);
extern void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau,
    double *work, int *lwork, int *info);
extern void zgeqrf_(int *m, int *n, complex_t *a, int *lda, 
    complex_t *tau, complex_t *work, int *lwork, int *info);
extern void dormqr_(char *side, char *trans, int *m, int *n, int *k,
    double *a, int *lda, double *tau, double *c, int *ldc, double *work,
    int *lwork, int *info);
extern void zunmqr_(char *side, char *trans, int *m, int *n, int *k,
    complex_t *a, int *lda, complex_t *tau, complex_t *c, 
    int *ldc, complex_t *work, int *lwork, int *info);
extern void dorgqr_(int *m, int *n, int *k, double *A, int *lda,
    double *tau, double *work, int *lwork, int *info);
extern void zungqr_(int *m, int *n, int *k, complex_t *A, int *lda,
    complex_t *tau, complex_t *work, int *lwork, int *info);
extern void dorglq_(int *m, int *n, int *k, double *A, int *lda,
    double *tau, double *work, int *lwork, int *info);
extern void zunglq_(int *m, int *n, int *k, complex_t *A, int *lda,
    complex_t *tau, complex_t *work, int *lwork, int *info);

extern void dgelqf_(int *m, int *n, double *a, int *lda, double *tau,
    double *work, int *lwork, int *info);
extern void zgelqf_(int *m, int *n, complex_t *a, int *lda, 
    complex_t *tau, complex_t *work, int *lwork, int *info);
extern void dormlq_(char *side, char *trans, int *m, int *n, int *k,
    double *a, int *lda, double *tau, double *c, int *ldc, double *work,
    int *lwork, int *info);
extern void zunmlq_(char *side, char *trans, int *m, int *n, int *k,
    complex_t *a, int *lda, complex_t *tau, complex_t *c, 
    int *ldc, complex_t *work, int *lwork, int *info);

extern void dgeqp3_(int *m, int *n, double *a, int *lda, int *jpvt,
    double *tau, double *work, int *lwork, int *info);
extern void zgeqp3_(int *m, int *n, complex_t *a, int *lda, int *jpvt,
    complex_t *tau, complex_t *work, int *lwork, double *rwork, 
    int *info);

extern void dsyev_(char *jobz, char *uplo, int *n, double *A, int *lda,
    double *W, double *work, int *lwork, int *info);
extern void zheev_(char *jobz, char *uplo, int *n, complex_t *A, 
    int *lda, double *W, complex_t *work, int *lwork, double *rwork, 
    int *info);
extern void dsyevx_(char *jobz, char *range, char *uplo, int *n, double *A,
    int *lda, double *vl, double *vu, int *il, int *iu, double *abstol,
    int *m, double *W, double *Z, int *ldz, double *work, int *lwork,
    int *iwork, int *ifail, int *info);
extern void zheevx_(char *jobz, char *range, char *uplo, int *n,
    complex_t *A, int *lda, double *vl, double *vu, int *il, int *iu,
    double *abstol, int *m, double *W, complex_t *Z, int *ldz, 
    complex_t *work, int *lwork, double *rwork, int *iwork, 
    int *ifail, int *info);
extern void dsyevd_(char *jobz, char *uplo, int *n, double *A, int *ldA,
    double *W, double *work, int *lwork, int *iwork, int *liwork,
    int *info);
extern void zheevd_(char *jobz, char *uplo, int *n, complex_t *A, 
    int *ldA, double *W, complex_t *work, int *lwork, double *rwork, 
    int *lrwork, int *iwork, int *liwork, int *info);
extern void dsyevr_(char *jobz, char *range, char *uplo, int *n, double *A,
    int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol,
    int *m, double *W, double *Z, int *ldZ, int *isuppz, double *work,
    int *lwork, int *iwork, int *liwork, int *info);
extern void zheevr_(char *jobz, char *range, char *uplo, int *n,
    complex_t *A, int *ldA, double *vl, double *vu, int *il, int *iu,
    double *abstol, int *m, double *W, complex_t *Z, int *ldZ, 
    int *isuppz, complex_t *work, int *lwork, double *rwork, 
    int *lrwork, int *iwork, int *liwork, int *info);

extern void dsygv_(int *itype, char *jobz, char *uplo, int *n, double *A,
    int *lda, double *B, int *ldb, double *W, double *work, int *lwork,
    int *info);
extern void zhegv_(int *itype, char *jobz, char *uplo, int *n, 
    complex_t *A, int *lda, complex_t *B, int *ldb, double *W, 
    complex_t *work, int *lwork, double *rwork, int *info);

extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A,
    int *ldA, double *S, double *U, int *ldU, double *Vt, int *ldVt,
    double *work, int *lwork, int *info);
extern void dgesdd_(char *jobz, int *m, int *n, double *A, int *ldA,
    double *S, double *U, int *ldU, double *Vt, int *ldVt, double *work,
    int *lwork, int *iwork, int *info);
extern void zgesvd_(char *jobu, char *jobvt, int *m, int *n, 
    complex_t *A, int *ldA, double *S, complex_t *U, int *ldU, 
    complex_t *Vt, int *ldVt, complex_t *work, int *lwork, 
    double *rwork, int *info);
extern void zgesdd_(char *jobz, int *m, int *n, complex_t *A, 
    int *ldA, double *S, complex_t *U, int *ldU, complex_t *Vt, 
    int *ldVt, complex_t *work, int *lwork, double *rwork, 
    int *iwork, int *info);

extern void dgees_(char *jobvs, char *sort, int (*select)(double *, double *), int *n,
    double *A, int *ldA, int *sdim, double *wr, double *wi, double *vs,
    int *ldvs, double *work, int *lwork, int *bwork, int *info);
extern void zgees_(char *jobvs, char *sort, int (*select)(complex_t *), int *n,
    complex_t *A, int *ldA, int *sdim, complex_t *w, 
    complex_t *vs, int *ldvs, complex_t *work, int *lwork, 
    complex_t *rwork, int *bwork, int *info);
extern void dgges_(char *jobvsl, char *jobvsr, char *sort, int (*delctg)(double *, double *, double *),
    int *n, double *A, int *ldA, double *B, int *ldB, int *sdim,
    double *alphar, double *alphai, double *beta, double *vsl, int *ldvsl,
    double *vsr, int *ldvsr, double *work, int *lwork, int *bwork,
    int *info);
extern void zgges_(char *jobvsl, char *jobvsr, char *sort, int (*delctg)(complex_t *, double *),
    int *n, complex_t *A, int *ldA, complex_t *B, int *ldB, 
    int *sdim, complex_t *alpha, complex_t *beta, 
    complex_t *vsl, int *ldvsl, complex_t *vsr, int *ldvsr, 
    complex_t *work, int *lwork, double *rwork, int *bwork, 
    int *info);

/***********************************************************************/
/****                                                               ****/
/****                      LAPACK functions                         ****/
/****                                                               ****/
/***********************************************************************/

void lapack_getrf(matrix *A, matrix *ipiv, int m, int n, int ldA, int offsetA);
void lapack_getrs(matrix *A, matrix *ipiv, matrix *B, char trans, int n, int nrhs, 
                  int ldA, int ldB, int offsetA, int offsetB);
void lapack_getri(matrix *A, matrix *ipiv, int n, int ldA, int offsetA);
void lapack_gesv(matrix *A, matrix *B, matrix *ipiv, int n, int nrhs, int ldA, 
                 int ldB, int offsetA, int offsetB);
void lapack_gbtrf(matrix *A, int m, int kl, matrix *ipiv, int n, int ku, int ldA, int offsetA);
void lapack_gbtrs(matrix *A, int kl, matrix *ipiv, matrix *B, char trans, int n, 
                  int ku, int nrhs, int ldA, int ldB, int offsetA, int offsetB);
void lapack_gbsv(matrix *A, int kl, matrix *B, matrix *ipiv, int ku, int n, 
                 int nrhs, int ldA, int ldB, int offsetA, int offsetB);
void lapack_gttrf(matrix *dl, matrix *d, matrix *du, matrix *du2, matrix *ipiv, 
                 int n, int offsetdl, int offsetd, int offsetdu);
void lapack_gttrs(matrix *dl, matrix *d, matrix *du, matrix *du2, matrix *ipiv, matrix *B, char trans, 
                  int n, int nrhs, int ldB, int offsetdl, int offsetd, int offsetdu, int offsetB);
void lapack_gtsv(matrix *dl, matrix *d, matrix *du, matrix *B, int n, int nrhs, 
                int ldB, int offsetdl, int offsetd, int offsetdu, int offsetB);
void lapack_potrf(matrix* A, char uplo, int n, int ldA, int offsetA);
void lapack_potrs(matrix *A, matrix *B, char uplo, int n, int nrhs, int ldA, int ldB, 
                  int offsetA, int offsetB);
void lapack_potri(matrix* A, char uplo, int n, int ldA, int offsetA);
void lapack_trtrs(matrix *A, matrix *B, char uplo, char trans, char diag, 
                int n, int nrhs, int ldA, int ldB, int offsetA, int offsetB);
void lapack_geqrf(matrix *A, matrix *tau, int m, int n, int ldA, int offsetA);
void lapack_ormqr(matrix *A, matrix *tau, matrix *C, char side, char trans, int m, 
                int n, int k, int ldA, int ldC, int offsetA, int offsetC);
void lapack_gesvd(matrix *A, matrix *S, char jobu, char jobvt, matrix *U, matrix *Vt, int m, int n, 
                int ldA, int ldU, int ldVt, int offsetA, int offsetS, int offsetU, int offsetVt);



#endif