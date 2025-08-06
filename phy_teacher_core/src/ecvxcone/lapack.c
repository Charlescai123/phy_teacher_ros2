/*
 * Copyright 2012-2023 M. Andersen and L. Vandenberghe.
 * Copyright 2010-2011 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT.
 *
 * CVXOPT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * CVXOPT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "lapack.h"

/**
 * @brief LU factorization of a general real or complex m-by-n matrix
 * 
 * lapack_getrf(A, ipiv, m=A.size[0], n=A.size[1], ldA=max(1,A.size[0]), offsetA=0)
 * 
 * @details
 * Computes the LU decomposition with partial pivoting:
 * - P·A = L·U
 * where:
 * - P is a permutation matrix,
 * - L is lower triangular with unit diagonal elements,
 * - U is upper triangular.
 * 
 * On exit:
 * - The matrix A is overwritten with L and U (L is stored below the diagonal with implied ones, U on and above the diagonal).
 * - The pivot vector `ipiv` contains the row swap information to form P.
 *
 * @param[in,out] A     Input matrix to factor ('d' or 'z' type)
 * @param[out]    ipiv  Pivot indices ('i' type), length at least min(m,n)
 * @param[in]     m     Number of rows of A (default: A.size[0]) (default = -1)
 * @param[in]     n     Number of columns of A (default: A.size[1]) (default = -1)
 * @param[in]     ldA   Leading dimension of A (≥ max(1,m)) (default = 0)
 * @param[in]     offsetA Matrix offset (nonnegative) (default = 0)
 *
 * @note
 * - Implements the LAPACK GETRF operation
 * - Suitable for general dense matrices (real or complex)
 * - The factorization is used in solving linear systems or computing determinants
 * - Uses partial pivoting with row interchanges for numerical stability
 *
 * @warning
 * - Matrix must be of type 'd' (real) or 'z' (complex)
 * - ldA must be ≥ max(1,m)
 * - offsetA must be ≥ 0
 * - If U(k,k) = 0, the matrix is exactly singular and factorization fails at step k
 * - Input matrix A will be overwritten
 *
 * @see LAPACK GETRF documentation
 */
void lapack_getrf(matrix *A, matrix *ipiv, int m, int n, int ldA, int offsetA)
{
    // int m=-1, n=-1, ldA=0, oA=0, info;
    int oA = offsetA, info;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv ->id != INT) err_int_mtrx("ipiv");
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (len(ipiv) < MIN(n,m)) err_buf_len("ipiv");

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int *ipiv_ptr = malloc(MIN(m,n)*sizeof(int));
    if (!ipiv_ptr) return;
#else
    int *ipiv_ptr = MAT_BUFI(ipiv);
#endif

    switch (MAT_ID(A)) {
        case DOUBLE:
            dgetrf_(&m, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr, &info);
            break;

        case COMPLEX:
            zgetrf_(&m, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr, &info);
            break;

        default:
#if (SIZEOF_INT < SIZEOF_SIZE_T)
            free(ipiv_ptr);
#endif
            err_invalid_id;
    }

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int i;  for (i=0; i<MIN(m,n); i++) MAT_BUFI(ipiv)[i] = ipiv_ptr[i];
    free(ipiv_ptr);
#endif

    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Solves a system of linear equations using LU factorization.
 * 
 * lapack_getrs(A, ipiv, B, trans='N', n=A.size[0], nrhs=B.size[1],
 *              ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,
 *              offsetB=0)
 * 
 * @details
 * Solves a system of linear equations A*X = B, A^T*X = B, or A^H*X = B
 * using the LU factorization computed by getrf() or gesv().
 * On entry, A and ipiv must contain the LU factorization of the matrix A.
 * On exit, B is overwritten with the solution matrix X.
 * 
 * @param[in]  A        Coefficient matrix ('d' or 'z' type), containing LU factors
 * @param[in]  ipiv     Pivot indices matrix ('i' type), from LU factorization
 * @param[in,out] B     Right-hand side matrix, replaced by the solution ('d' or 'z' type, same as A)
 * @param[in]  trans    Specifies the system to solve:
 *                      - 'N': A * X = B (No transpose)
 *                      - 'T': A^T * X = B (Transpose)
 *                      - 'C': A^H * X = B (Conjugate transpose)
 *                      (default = 'N')
 * @param[in]  n        Order of the matrix A (default = A.nrows, use -1 for default)
 * @param[in]  nrhs     Number of right-hand sides (default = B.ncols, use -1 for default)
 * @param[in]  ldA      Leading dimension of A (≥ max(1,n), use 0 for default)
 * @param[in]  ldB      Leading dimension of B (≥ max(1,n), use 0 for default)
 * @param[in]  offsetA  Offset into A matrix (nonnegative, default = 0)
 * @param[in]  offsetB  Offset into B matrix (nonnegative, default = 0)
 * 
 */
void lapack_getrs(matrix *A, matrix *ipiv, matrix *B, char trans, int n, int nrhs, 
                  int ldA, int ldB, int offsetA, int offsetB)
{
    // int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
    int oA = offsetA, oB = offsetB, info;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            ERR_TYPE("A must be square");
            return;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (len(ipiv) < n) err_buf_len("ipiv");

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int *ipiv_ptr = malloc(n*sizeof(int));
    if (!ipiv_ptr) err_no_memory;
    int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
#else
    int *ipiv_ptr = MAT_BUFI(ipiv);
#endif

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            dgetrs_(&trans, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                MAT_BUFD(B)+oB, &ldB, &info);
            break;

        case COMPLEX:
            zgetrs_(&trans, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                MAT_BUFZ(B)+oB, &ldB, &info);
            break;

	default:
#if (SIZEOF_INT < SIZEOF_SIZE_T)
            free(ipiv_ptr);
#endif
            err_invalid_id;
    }

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    free(ipiv_ptr);
#endif
    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Inverse of a real or complex matrix using LU factorization
 * 
 * lapack_getri(A, ipiv, n=A.size[0], ldA=max(1,A.size[0]), offsetA=0)
 * 
 * @details
 * Computes the inverse of a general real or complex matrix A of order n,
 * using its LU decomposition from `lapack_getrf` or `lapack_gesv`:
 * - A⁻¹ = inv(U)·inv(L)·P
 * where:
 * - A = P⁻¹·L·U (LU factorization with pivoting)
 *
 * On exit:
 * - The input matrix A is overwritten with its inverse.
 *
 * @param[in,out] A     LU-factored matrix to invert ('d' or 'z' type)
 * @param[in]     ipiv  Pivot indices from `getrf` ('i' type)
 * @param[in]     n     Order of matrix A (default: A.size[0]) (default = -1)
 * @param[in]     ldA   Leading dimension of A (≥ max(1,n)) (default = 0)
 * @param[in]     offsetA Matrix offset (nonnegative) (default = 0)
 */
void lapack_getri(matrix *A, matrix *ipiv, int n, int ldA, int offsetA)
{
    // int n=-1, ldA=0, oA=0, info, lwork;
    int oA = offsetA, info, lwork;
    void *work;
    number wl;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            ERR("A must be square");
            return;
        }
    }
    if (n == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (len(ipiv) < n) err_buf_len("ipiv");

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int *ipiv_ptr = malloc(n*sizeof(int));
    if (!ipiv_ptr) err_no_memory;
    int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
#else
    int *ipiv_ptr = MAT_BUFI(ipiv);
#endif

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            dgetri_(&n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            lwork = (int) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double)))) {
#if (SIZEOF_INT < SIZEOF_SIZE_T)
                free(ipiv_ptr);
#endif
                err_no_memory;
            }
            dgetri_(&n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr, (double *) work,
                &lwork, &info);
            free(work);
            break;

        case COMPLEX:
            lwork = -1;
            zgetri_(&n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            lwork = (int) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t)))){
#if (SIZEOF_INT < SIZEOF_SIZE_T)
                free(ipiv_ptr);
#endif
                err_no_memory;
            }
            zgetri_(&n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                (complex_t *) work, &lwork, &info);
            free(work);
            break;

        default:
#if (SIZEOF_INT < SIZEOF_SIZE_T)
            free(ipiv_ptr);
#endif
            err_invalid_id;
    }

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    free(ipiv_ptr);
#endif
    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Solves a general real or complex system of linear equations.
 * 
 * lapack_dgesv(A, B, ipiv=None, n=A.size[0], nrhs=B.size[1], 
 *              ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, 
 *              offsetB=0)
 * 
 * @details
 * Solves the system A * X = B, where A is an n-by-n real or complex matrix.
 * Computes the LU factorization of A (if ipiv is provided) and uses it to
 * solve for X. On exit, B is overwritten with the solution.
 * 
 * If `ipiv` is provided, the LU factorization is stored in A and the pivot indices
 * in `ipiv`. If `ipiv` is not provided, A is not modified and the LU factors are not returned.
 * 
 * @param[in,out] A       Coefficient matrix ('d' or 'z' type). May be overwritten with LU factors.
 * @param[in,out] B       Right-hand side matrix, replaced by the solution X. Must have same type as A.
 * @param[out]    ipiv    Pivot index vector ('i' type). Must have length at least n. If NULL, LU factors are not returned.
 * @param[in]     n       Order of matrix A (default: A.nrows, use -1 for default).
 * @param[in]     nrhs    Number of right-hand sides (default: B.ncols, use -1 for default).
 * @param[in]     ldA     Leading dimension of A (≥ max(1,n), use 0 for default).
 * @param[in]     ldB     Leading dimension of B (≥ max(1,n), use 0 for default).
 * @param[in]     offsetA Offset into A matrix (nonnegative, default = 0).
 * @param[in]     offsetB Offset into B matrix (nonnegative, default = 0).
 * 
 */
void lapack_gesv(matrix *A, matrix *B, matrix *ipiv, int n, int nrhs, int ldA, int ldB, int offsetA, int offsetB)
{
    // int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, k;
    int oA = offsetA, oB = offsetB, info, k;

    void *Ac=NULL;
    int *ipivc=NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
        err_int_mtrx("ipiv");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            ERR_TYPE("A must be square");
            return;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

    if (ipiv) {
#if (SIZEOF_INT < SIZEOF_SIZE_T)
        if (!(ipivc = (int *) calloc(n, sizeof(int))))
            err_no_memory;
#else
        ipivc = MAT_BUFI(ipiv);
#endif
    }
    else if (!(ipivc = (int *) calloc(n, sizeof(int))))
        err_no_memory;

    switch (MAT_ID(A)){
        case DOUBLE:
            if (ipiv)
                dgesv_(&n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
            else {
                if (!(Ac = (void *) calloc(n*n, sizeof(double)))){
                    free(ipivc);
                    err_no_memory;
                }
                for (k=0; k<n; k++) memcpy((double *) Ac + k*n,
                    MAT_BUFD(A)+oA+k*ldA, n*sizeof(double));
                dgesv_(&n, &nrhs, (double *) Ac, &n, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
                free(Ac);
            }
            break;

        case COMPLEX:
            if (ipiv)
                zgesv_(&n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, &info);
            else {
                if (!(Ac = (void *) calloc(n*n, sizeof(complex_t)))){
                    free(ipivc);
                    err_no_memory;
                }
                for (k=0; k<n; k++) memcpy((complex_t *) Ac + k*n,
                    MAT_BUFZ(A)+oA+k*ldA, n*sizeof(complex_t));
                zgesv_(&n, &nrhs, (complex_t *) Ac, &n, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, &info);
                free(Ac);
            }
            break;

        default:
            if (ipiv){
#if (SIZEOF_INT < SIZEOF_SIZE_T)
                free(ipivc);
#endif
            }
            else free(ipivc);
            err_invalid_id;
    }

    if (ipiv){
#if (SIZEOF_INT < SIZEOF_SIZE_T)
        for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
        free(ipivc);
#endif
    }
    else free(ipivc);

    if (info) err_lapack(info);
    else return;
}



/**
 * @brief LU factorization of a real or complex band matrix.
 * 
 * lapack_gbtrf(A, m, kl, ipiv, n=A.size[1], ku=A.size[0]-2*kl-1,
 *              ldA=max(1,A.size[0]), offsetA=0)
 * 
 * @details
 * Computes the LU factorization of an m-by-n real or complex band matrix
 * with `kl` subdiagonals and `ku` superdiagonals using partial pivoting.
 * The matrix is stored in BLAS banded format: the diagonals are located in
 * rows `kl+1` to `2*kl+ku+1` of the array A.
 * 
 * On exit, the matrix `A` is overwritten with the LU factors, and the pivot
 * indices are stored in `ipiv`.
 * 
 * @param[in,out] A       Input/output matrix ('d' or 'z' type) in banded storage format.
 *                        On exit, contains details of the LU factorization.
 * @param[in]     m       Number of rows of the matrix A (nonnegative).
 * @param[in]     kl      Number of subdiagonals (nonnegative).
 * @param[out]    ipiv    Pivot indices array ('i' type) of length at least min(m, n).
 * @param[in]     n       Number of columns of the matrix A (default = A.ncols, use -1 for default).
 * @param[in]     ku      Number of superdiagonals (default = A.nrows - 2*kl - 1, use -1 for default).
 * @param[in]     ldA     Leading dimension of A (≥ 2*kl+ku+1). Use 0 for default.
 * @param[in]     offsetA Offset into A (nonnegative, default = 0).
 * 
 */
void lapack_gbtrf(matrix *A, int m, int kl, matrix *ipiv, int n, int ku, int ldA, int offsetA)
{
    // int m, kl, n=-1, ku=-1, ldA=0, oA=0, info;
    int oA = offsetA, info;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (m < 0) err_nn_int("m");
    if (kl < 0) err_nn_int("kl");
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return;
    if (ku < 0) ku = A->nrows - 2*kl - 1;
    if (ku < 0) err_nn_int("kl");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < 2*kl + ku + 1) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + 2*kl + ku + 1 > len(A)) err_buf_len("A");
    if (!Matrix_Check(ipiv) || ipiv ->id != INT) err_int_mtrx("ipiv");
    if (len(ipiv) < MIN(n,m)) err_buf_len("ipiv");

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int *ipiv_ptr = malloc(MIN(m,n)*sizeof(int));
    if (!ipiv_ptr) err_no_memory;
#else
    int *ipiv_ptr = MAT_BUFI(ipiv);
#endif

    switch (MAT_ID(A)) {
        case DOUBLE:
            dgbtrf_(&m, &n, &kl, &ku, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                &info);
            break;

        case COMPLEX:
            zgbtrf_(&m, &n, &kl, &ku, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                &info);
            break;

        default:
#if (SIZEOF_INT < SIZEOF_SIZE_T)
            free(ipiv_ptr);
#endif
            err_invalid_id;
    }

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int i;  for (i=0; i<MIN(m,n); i++) MAT_BUFI(ipiv)[i] = ipiv_ptr[i];
    free(ipiv_ptr);
#endif

    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Solves a linear system with a real or complex band matrix using its LU factorization.
 * 
 * lapack_gbtrs(A, kl, ipiv, B, trans='N', n=A.size[1], ku=A.size[0]-2*kl-1,
 *              nrhs=B.size[1], ldA=max(1,A.size[0]), ldB=max(1,B.size[0]),
 *              offsetA=0, offsetB=0)
 * 
 * @details
 * Solves a system of linear equations A * X = B, A^T * X = B, or A^H * X = B
 * using the LU factorization of a banded matrix `A` computed by gbtrf() or gbsv().
 * 
 * On entry, matrix `A` and the pivot indices `ipiv` contain the LU factorization
 * of an n-by-n band matrix. On exit, the right-hand side matrix `B` is overwritten
 * with the solution `X`.
 * 
 * This function corresponds to LAPACK routine `lapack_dgbtrs` or `lapack_zgbtrs`
 * depending on the matrix type.
 * 
 * @param[in]  A        LU-factorized coefficient matrix ('d' or 'z' type) in banded format.
 * @param[in]  kl       Number of subdiagonals (nonnegative).
 * @param[in]  ipiv     Pivot indices ('i' type), as returned by gbtrf() or gbsv().
 * @param[in,out] B     Right-hand side matrix, replaced by solution X. Must have same type as A.
 * @param[in]  trans    Specifies the system to solve:
 *                     - 'N': A * X = B
 *                     - 'T': A^T * X = B
 *                     - 'C': A^H * X = B
 *                     (default = 'N')
 * @param[in]  n        Order of the matrix A (default = A.ncols, use -1 for default).
 * @param[in]  ku       Number of superdiagonals (default = A.nrows - 2*kl - 1, use -1 for default).
 * @param[in]  nrhs     Number of right-hand sides (default = B.ncols, use -1 for default).
 * @param[in]  ldA      Leading dimension of A (≥ 2*kl+ku+1, use 0 for default).
 * @param[in]  ldB      Leading dimension of B (≥ max(1,n), use 0 for default).
 * @param[in]  offsetA  Offset into A matrix (nonnegative, default = 0).
 * @param[in]  offsetB  Offset into B matrix (nonnegative, default = 0).
 * 
 */
void lapack_gbtrs(matrix *A, int kl, matrix *ipiv, matrix *B, char trans, int n, 
                  int ku, int nrhs, int ldA, int ldB, int offsetA, int offsetB)
{
    // int kl, n=-1, ku=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;

    int oA = offsetA, oB = offsetB, info;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (kl < 0) err_nn_int("kl");
    if (ku < 0) ku = A->nrows - 2*kl - 1;
    if (ku < 0) err_nn_int("kl");
    if (n < 0) n = A->ncols;
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < 2*kl+ku+1) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + 2*kl + ku + 1 > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (len(ipiv) < n) err_buf_len("ipiv");

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int *ipiv_ptr = malloc(n*sizeof(int));
    if (!ipiv_ptr) err_no_memory;
    int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
#else
    int *ipiv_ptr = MAT_BUFI(ipiv);
#endif

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            dgbtrs_(&trans, &n, &kl, &ku, &nrhs, MAT_BUFD(A)+oA, &ldA,
                ipiv_ptr, MAT_BUFD(B)+oB, &ldB, &info);
            break;

        case COMPLEX:
            zgbtrs_(&trans, &n, &kl, &ku, &nrhs, MAT_BUFZ(A)+oA, &ldA,
                ipiv_ptr, MAT_BUFZ(B)+oB, &ldB, &info);
            break;

	default:
#if (SIZEOF_INT < SIZEOF_SIZE_T)
            free(ipiv_ptr);
#endif
            err_invalid_id;
    }

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    free(ipiv_ptr);
#endif
    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Solves a system of linear equations with a real or complex banded matrix.
 * 
 * lapack_gbsv(A, kl, B, ipiv=None, ku=None, n=A.size[1], nrhs=B.size[1],
 *             ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, 
 *             offsetB=0)
 * 
 * @details
 * Solves the linear system A * X = B, where A is an n-by-n real or complex
 * banded matrix with `kl` subdiagonals and `ku` superdiagonals.
 * 
 * If `ipiv` is provided, A is assumed to be stored in full banded BLAS format 
 * (with diagonals in rows kl+1 to 2*kl+ku+1), and the function returns the LU
 * factorization and pivot indices.
 * 
 * If `ipiv` is not provided, A must be stored in compact format (rows 1 to kl+ku+1)
 * and will not be modified. In this case, the factorization is not returned.
 * 
 * On exit, B is overwritten with the solution matrix X.
 * 
 * This function corresponds to LAPACK routines `lapack_dgbsv` or `lapack_zgbsv`.
 * 
 * @param[in,out] A       Banded coefficient matrix ('d' or 'z' type). May be overwritten with LU factors.
 * @param[in]     kl      Number of subdiagonals (nonnegative).
 * @param[in,out] B       Right-hand side matrix, replaced by solution X. Must match A's type.
 * @param[out]    ipiv    Pivot indices ('i' type) of length at least n. Optional; if NULL, factorization is not returned.
 * @param[in]     ku      Number of superdiagonals. If negative, a default is used:
 *                        - If ipiv is NULL: ku = A.nrows - kl - 1
 *                        - Otherwise: ku = A.nrows - 2*kl - 1
 * @param[in]     n       Order of matrix A (default: A.ncols, use -1 for default).
 * @param[in]     nrhs    Number of right-hand sides (default: B.ncols, use -1 for default).
 * @param[in]     ldA     Leading dimension of A. Must satisfy:
 *                        - If ipiv is NULL: ldA ≥ kl+ku+1
 *                        - Otherwise: ldA ≥ 2*kl+ku+1
 *                        (Use 0 for default.)
 * @param[in]     ldB     Leading dimension of B (≥ max(1,n), use 0 for default).
 * @param[in]     offsetA Offset into A matrix (nonnegative, default = 0).
 * @param[in]     offsetB Offset into B matrix (nonnegative, default = 0).
 * 
 */
void lapack_gbsv(matrix *A, int kl, matrix *B, matrix *ipiv, int ku, int n, 
                int nrhs, int ldA, int ldB, int offsetA, int offsetB)
{
    void *Ac;
    // int kl, ku=-1, n=-1, nrhs=-1, ldA=0, oA=0, ldB=0, oB=0, info, k;
    int oA = offsetA, oB = offsetB, info, k;
    int *ipivc=NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
        err_int_mtrx("ipiv");
    if (n < 0) n = A->ncols;
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (kl < 0) err_nn_int("kl");
    if (ku < 0) ku = A->nrows - kl - 1 - (ipiv ? kl : 0);
    if (ku < 0) err_nn_int("ku");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA < ( ipiv ? 2*kl+ku+1 : kl+ku+1)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + (ipiv ? 2*kl+ku+1 : kl+ku+1) > len(A))
        err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

    if (ipiv) {
#if (SIZEOF_INT < SIZEOF_SIZE_T)
        if (!(ipivc = (int *) calloc(n, sizeof(int))))
            err_no_memory;
#else
        ipivc = MAT_BUFI(ipiv);
#endif
    }
    else if (!(ipivc = (int *) calloc(n, sizeof(int))))
        err_no_memory;

    switch (MAT_ID(A)) {
        case DOUBLE:
            if (ipiv)
                dgbsv_(&n, &kl, &ku, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
            else {
                if (!(Ac = (void *) calloc((2*kl+ku+1)*n,
                    sizeof(double)))){
                    free(ipivc);
                    err_no_memory;
                }
                for (k=0; k<n; k++)
                    memcpy((double *) Ac + kl + k*(2*kl+ku+1),
                        MAT_BUFD(A) + oA + k*ldA,
                        (kl+ku+1)*sizeof(double));
                ldA = 2*kl+ku+1;
                dgbsv_(&n, &kl, &ku, &nrhs, (double *) Ac, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
                free(Ac);
            }
            break;

        case COMPLEX:
            if (ipiv)
                zgbsv_(&n, &kl, &ku, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, &info);
            else {
                if (!(Ac = (void *) calloc((2*kl+ku+1)*n,
                    sizeof(complex_t)))){
                    free(ipivc);
                    err_no_memory;
                }
                for (k=0; k<n; k++)
                    memcpy((complex_t *) Ac + kl + k*(2*kl+ku+1),
                        MAT_BUFZ(A) + oA + k*ldA,
                        (kl+ku+1)*sizeof(complex_t));
                ldA = 2*kl+ku+1;
                zgbsv_(&n, &kl, &ku, &nrhs, (complex_t *) Ac, &ldA, 
                    ipivc, MAT_BUFZ(B)+oB, &ldB, &info);
                free(Ac);
            }
            break;

        default:
            if (ipiv){
#if (SIZEOF_INT < SIZEOF_SIZE_T)
                free(ipivc);
#endif
            }
            else free(ipivc);
            err_invalid_id;
    }

    if (ipiv){
#if (SIZEOF_INT < SIZEOF_SIZE_T)
        for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
        free(ipivc);
#endif
    }
    else free(ipivc);

    if (info) err_lapack(info);
    else return;

}

/**
 * @brief LU factorization of a real or complex tridiagonal matrix.
 * 
 * lapack_gttrf(dl, d, du, du2, ipiv, n=len(d)-offsetd, 
 *              offsetdl=0, offsetd=0, offsetdu=0)
 * 
 * @details
 * Computes the LU factorization of an n-by-n real or complex tridiagonal matrix A,
 * such that A = P * L * U, where P is a permutation matrix, L is unit lower triangular,
 * and U is upper triangular.
 * 
 * The matrix A is specified by its diagonals:
 * - `dl`: sub-diagonal (n-1 entries)
 * - `d`: main diagonal (n entries)
 * - `du`: super-diagonal (n-1 entries)
 * 
 * On exit, these vectors and `du2`, `ipiv` store the factorization data used for solving.
 * 
 * This function corresponds to LAPACK routine `lapack_dgttrf` or `lapack_zgttrf`.
 * 
 * @param[in,out] dl       Sub-diagonal vector ('d' or 'z' type), overwritten with L components.
 * @param[in,out] d        Main diagonal vector ('d' or 'z' type), overwritten with U diagonal.
 * @param[in,out] du       Super-diagonal vector ('d' or 'z' type), overwritten with U super-diagonal.
 * @param[out]    du2      Second super-diagonal of U (length at least n-2), used in block LU factorization.
 * @param[out]    ipiv     Pivot indices ('i' type), length at least n.
 * @param[in]     n        Size of the system (default: len(d) - offsetd, use -1 for default).
 * @param[in]     offsetdl Offset into dl (nonnegative, default = 0).
 * @param[in]     offsetd  Offset into d (nonnegative, default = 0).
 * @param[in]     offsetdu Offset into du (nonnegative, default = 0).
 * 
 */
void lapack_gttrf(matrix *dl, matrix *d, matrix *du, matrix *du2, matrix *ipiv, 
                 int n, int offsetdl, int offsetd, int offsetdu)
{
    // int n=-1, odl=0, od=0, odu=0, info;
    int odl = offsetdl, od = offsetd, odu = offsetdu, info;

    if (!Matrix_Check(dl)) err_mtrx("dl");
    if (!Matrix_Check(d)) err_mtrx("d");
    if (!Matrix_Check(du)) err_mtrx("du");
    if (!Matrix_Check(du2)) err_mtrx("du");
    if ((MAT_ID(dl) != MAT_ID(d)) || (MAT_ID(dl) != MAT_ID(du)) ||
        (MAT_ID(dl) != MAT_ID(du2))) err_conflicting_ids;
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (n == 0) return;
    if (odl < 0) err_nn_int("offsetdl");
    if (odl + n - 1  > len(dl)) err_buf_len("dl");
    if (od + n > len(d)) err_buf_len("d");
    if (odu < 0) err_nn_int("offsetdu");
    if (odu + n - 1  > len(du)) err_buf_len("du");
    if (n - 2  > len(du2)) err_buf_len("du2");
    if (len(ipiv) < n) err_buf_len("ipiv");
    if (n > len(ipiv)) err_buf_len("ipiv");

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int *ipiv_ptr = malloc(n*sizeof(int));
    if (!ipiv_ptr) err_no_memory;
#else
    int *ipiv_ptr = MAT_BUFI(ipiv);
#endif

    switch (MAT_ID(dl)){
        case DOUBLE:
            dgttrf_(&n, MAT_BUFD(dl)+odl, MAT_BUFD(d)+od, MAT_BUFD(du)+odu,
                MAT_BUFD(du2), ipiv_ptr, &info);
            break;

        case COMPLEX:
            zgttrf_(&n, MAT_BUFZ(dl)+odl, MAT_BUFZ(d)+od, MAT_BUFZ(du)+odu,
                MAT_BUFZ(du2), ipiv_ptr, &info);
            break;

        default:
#if (SIZEOF_INT < SIZEOF_SIZE_T)
            free(ipiv_ptr);
#endif
            err_invalid_id;
    }

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int i;  for (i=0; i<n; i++) MAT_BUFI(ipiv)[i] = ipiv_ptr[i];
    free(ipiv_ptr);
#endif
    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Solves a tridiagonal system of equations using LU factorization from gttrf().
 * 
 * lapack_gttrs(dl, d, du, du2, ipiv, B, trans='N', n=len(d)-offsetd,
 *              nrhs=B.size[1], ldB=max(1,B.size[0]), offsetdl=0, offsetd=0,
 *              offsetdu=0, offsetB=0)
 * 
 * @details
 * Solves the system A * X = B, A^T * X = B, or A^H * X = B, where A is an n-by-n
 * real or complex tridiagonal matrix that has been previously factorized using `gttrf()`.
 * 
 * On entry:
 * - `dl`, `d`, `du`, `du2`, and `ipiv` must contain the LU factorization of A.
 * On exit:
 * - Matrix `B` is overwritten with the solution matrix `X`.
 * 
 * This function corresponds to LAPACK routine `lapack_dgttrs` or `lapack_zgttrs`.
 * 
 * @param[in]     dl       Sub-diagonal of A ('d' or 'z' type).
 * @param[in]     d        Main diagonal of A ('d' or 'z' type).
 * @param[in]     du       Super-diagonal of A ('d' or 'z' type).
 * @param[in]     du2      Second super-diagonal ('d' or 'z' type), from gttrf().
 * @param[in]     ipiv     Pivot indices ('i' type), as computed by gttrf().
 * @param[in,out] B        Right-hand side matrix, replaced by solution X. Same type as `dl`.
 * @param[in]     trans    Specifies the system to solve:
 *                         - 'N': A * X = B (no transpose)
 *                         - 'T': A^T * X = B (transpose)
 *                         - 'C': A^H * X = B (conjugate transpose)
 *                         (default = 'N')
 * @param[in]     n        Order of matrix A (default = len(d) - offsetd, use -1 for default).
 * @param[in]     nrhs     Number of right-hand sides (default = B.ncols, use -1 for default).
 * @param[in]     ldB      Leading dimension of B (≥ max(1,n), use 0 for default).
 * @param[in]     offsetdl Offset into `dl` (nonnegative, default = 0).
 * @param[in]     offsetd  Offset into `d` (nonnegative, default = 0).
 * @param[in]     offsetdu Offset into `du` (nonnegative, default = 0).
 * @param[in]     offsetB  Offset into `B` (nonnegative, default = 0).
 * 
 */
void lapack_gttrs(matrix *dl, matrix *d, matrix *du, matrix *du2, matrix *ipiv, matrix *B, char trans, 
                  int n, int nrhs, int ldB, int offsetdl, int offsetd, int offsetdu, int offsetB)
{
    // int n=-1, nrhs=-1, ldB=0, odl=0, od=0, odu=0, oB=0, info;
    int odl = offsetdl, od = offsetd, odu = offsetdu, oB = offsetB, info;

    if (trans == 0) trans = 'N';

    if (!Matrix_Check(dl)) err_mtrx("dl");
    if (!Matrix_Check(d)) err_mtrx("d");
    if (!Matrix_Check(du)) err_mtrx("du");
    if (!Matrix_Check(du2)) err_mtrx("du");
    if (!Matrix_Check(B)) err_mtrx("B");
    if ((MAT_ID(dl) != MAT_ID(d)) || (MAT_ID(dl) != MAT_ID(du)) ||
        (MAT_ID(dl) != MAT_ID(du2)) || (MAT_ID(dl) != MAT_ID(B)))
        err_conflicting_ids;
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (odl < 0) err_nn_int("offsetdl");
    if (odl + n - 1  > len(dl)) err_buf_len("dl");
    if (od + n > len(d)) err_buf_len("d");
    if (odu < 0) err_nn_int("offsetdu");
    if (odu + n - 1  > len(du)) err_buf_len("du");
    if (n - 2  > len(du2)) err_buf_len("du2");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (n > len(ipiv)) err_buf_len("ipiv");

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    int *ipiv_ptr = malloc(n*sizeof(int));
    if (!ipiv_ptr) err_no_memory;
    int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
#else
    int *ipiv_ptr = MAT_BUFI(ipiv);
#endif

    switch (MAT_ID(dl)){
        case DOUBLE:
            dgttrs_(&trans, &n, &nrhs, MAT_BUFD(dl)+odl, MAT_BUFD(d)+od,
                MAT_BUFD(du)+odu, MAT_BUFD(du2), ipiv_ptr,
                MAT_BUFD(B)+oB, &ldB, &info);
            break;

        case COMPLEX:
            zgttrs_(&trans, &n, &nrhs, MAT_BUFZ(dl)+odl, MAT_BUFZ(d)+od,
                MAT_BUFZ(du)+odu, MAT_BUFZ(du2), ipiv_ptr,
                MAT_BUFZ(B)+oB, &ldB, &info);
            break;

        default:
#if (SIZEOF_INT < SIZEOF_SIZE_T)
            free(ipiv_ptr);
#endif
            err_invalid_id;
    }

#if (SIZEOF_INT < SIZEOF_SIZE_T)
    free(ipiv_ptr);
#endif

    if (info) err_lapack(info);
    else return;
}

/**
 * @brief Solves a system of linear equations with a real or complex tridiagonal matrix.
 * 
 * lapack_gtsv(dl, d, du, B, n=len(d)-offsetd, nrhs=B.size[1], 
 *             ldB=max(1,B.size[0]), offsetdl=0, offsetd=0, offsetdu=0, 
 *             offsetB=0)
 * 
 * @details
 * Solves the linear system A * X = B, where A is an n-by-n real or complex 
 * tridiagonal matrix. The matrix A is specified by its three diagonals:
 * - `dl`: sub-diagonal (n-1 elements)
 * - `d` : main diagonal (n elements)
 * - `du`: super-diagonal (n-1 elements)
 * 
 * On exit:
 * - B is overwritten with the solution matrix X.
 * - `dl`, `d`, and `du` are overwritten with the LU factorization of A.
 * 
 * This function corresponds to LAPACK routines `lapack_dgtsv` or `lapack_zgtsv`.
 * 
 * @param[in,out] dl       Sub-diagonal of A ('d' or 'z' type), overwritten with L factors.
 * @param[in,out] d        Main diagonal of A ('d' or 'z' type), overwritten with U diagonal.
 * @param[in,out] du       Super-diagonal of A ('d' or 'z' type), overwritten with U super-diagonal.
 * @param[in,out] B        Right-hand side matrix, replaced by solution X. Must have same type as `dl`.
 * @param[in]     n        Order of the system (default = len(d) - offsetd, use -1 for default).
 * @param[in]     nrhs     Number of right-hand sides (default = B.ncols, use -1 for default).
 * @param[in]     ldB      Leading dimension of B (≥ max(1,n), use 0 for default).
 * @param[in]     offsetdl Offset into `dl` (nonnegative, default = 0).
 * @param[in]     offsetd  Offset into `d` (nonnegative, default = 0).
 * @param[in]     offsetdu Offset into `du` (nonnegative, default = 0).
 * @param[in]     offsetB  Offset into `B` (nonnegative, default = 0).
 * 
 */
void lapack_gtsv(matrix *dl, matrix *d, matrix *du, matrix *B, int n, int nrhs, 
                int ldB, int offsetdl, int offsetd, int offsetdu, int offsetB)
{
    // int n=-1, nrhs=-1, ldB=0, odl=0, od=0, odu=0, oB=0, info;
    int odl = offsetdl, od = offsetd, odu = offsetdu, oB = offsetB, info;

    if (!Matrix_Check(dl)) err_mtrx("dl");
    if (!Matrix_Check(d)) err_mtrx("d");
    if (!Matrix_Check(du)) err_mtrx("du");
    if (!Matrix_Check(B)) err_mtrx("B");
    if ((MAT_ID(dl) != MAT_ID(B)) || (MAT_ID(dl) != MAT_ID(d)) ||
        (MAT_ID(dl) != MAT_ID(du)) || (MAT_ID(dl) != MAT_ID(B)))
        err_conflicting_ids;
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (odl < 0) err_nn_int("offsetdl");
    if (odl + n - 1  > len(dl)) err_buf_len("dl");
    if (od + n > len(d)) err_buf_len("d");
    if (odu < 0) err_nn_int("offsetdu");
    if (odu + n - 1  > len(du)) err_buf_len("du");
    if (oB < 0) err_nn_int("offsetB");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(dl)){
        case DOUBLE:
            dgtsv_(&n, &nrhs, MAT_BUFD(dl)+odl, MAT_BUFD(d)+od,
                MAT_BUFD(du)+odu, MAT_BUFD(B)+oB, &ldB, &info);
            break;

        case COMPLEX:
            zgtsv_(&n, &nrhs, MAT_BUFZ(dl)+odl, MAT_BUFZ(d)+od,
                MAT_BUFZ(du)+odu, MAT_BUFZ(B)+oB, &ldB, &info);
            break;

        default:
            err_invalid_id;
    }
    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Cholesky factorization of a symmetric/Hermitian positive definite matrix
 * 
 * lapack_potrf(A, uplo='L', n=A.size[0], ldA = max(1,A.size[0]), offsetA=0)
 * 
 * @details
 * Computes the Cholesky decomposition of a positive definite matrix:
 * - A = L·Lᵀ  (real symmetric case)
 * - A = L·Lᴴ  (complex Hermitian case)
 * 
 * On exit:
 * - If uplo='L', the lower triangular part contains L
 * - If uplo='U', the upper triangular part contains Lᵀ/Lᴴ
 *
 * @param[in,out] A   Input matrix to factor ('d' or 'z' type)
 * @param[in] uplo    Triangle selection ('L' for lower, 'U' for upper) (default = 'L')
 * @param[in] n       Order of matrix A (default: A.size[0]) (default = -1)
 * @param[in] ldA     Leading dimension of A (≥ max(1,n)) (default = 0)
 * @param[in] offsetA Matrix offset (nonnegative) (default = 0)
 *
 * @note
 * - Implements the LAPACK POTRF operation
 * - Handles both real symmetric and complex Hermitian matrices
 * - For complex matrices, uses conjugate transpose (Lᴴ)
 * - Only the specified triangle (upper/lower) needs to be initialized
 * - The other triangle is not referenced
 *
 * @warning
 * - Matrix must be positive definite (undefined behavior if not)
 * - Only the specified triangle (upper/lower) is used
 * - ldA must satisfy ldA ≥ max(1,n)
 * - offsetA must be nonnegative
 * - Undefined behavior if n > A.size[0]
 * - For complex matrices, input must be Hermitian
 *
 * @see LAPACK POTRF documentation
 */
void lapack_potrf(matrix* A, char uplo, int n, int ldA, int offsetA)
{
    // int n=-1, ldA=0, oA=0, info;
    int oA = offsetA, info;

    // Default values
    if (uplo == 0) uplo = 'L';
    if (ldA == 0) ldA = 0;
    if (oA < 0) oA = 0;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols) ERR("A is not square");
    }
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (n == 0) return;
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

    switch (MAT_ID(A)){
        case DOUBLE:
            dpotrf_(&uplo, &n, MAT_BUFD(A)+oA, &ldA, &info);
	    break;

        case COMPLEX:
            zpotrf_(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, &info);
	    break;

	default:
	    err_invalid_id;
    }
    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Solves symmetric positive definite linear systems using Cholesky factorization
 *
 * Solves the matrix equation A*X = B where:
 * - A is an n-by-n symmetric (real) or Hermitian (complex) positive definite matrix
 * - B is an n-by-nrhs right-hand side matrix
 * - A contains the Cholesky factor from previous potrf() or posv() call
 *
 * On exit, B is overwritten with the solution X.
 *
 * @param[in] A       Matrix containing Cholesky factor ('d' or 'z' type)
 * @param[in,out] B   Right-hand side matrix (input), solution matrix (output)
 * @param[in] uplo    Triangle selection: (default = 'L')
 *                   - 'L': Use lower triangular factor
 *                   - 'U': Use upper triangular factor
 * @param[in] n       Order of matrix A (default = -1)
 * @param[in] nrhs    Number of right-hand sides (default = -1)
 * @param[in] ldA     Leading dimension of A (default = 0)
 * @param[in] ldB     Leading dimension of B (default = 0)
 * @param[in] offsetA Starting offset in matrix A (default = 0)
 * @param[in] offsetB Starting offset in matrix B (default = 0)
 *
 */
void lapack_potrs(matrix *A, matrix *B, char uplo, int n, int nrhs, int ldA, int ldB, 
                  int offsetA, int offsetB)
{
    // int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
    int oA = offsetA, oB = offsetB, info;

    // Default values
    if (uplo == 0) uplo = 'L';
    if (n < 0) n = -1;
    if (nrhs < 0) nrhs = -1;
    if (ldA < 0) ldA = 0;
    if (ldB < 0) ldB = 0;
    if (oA < 0) oA = 0;
    if (oB < 0) oB = 0;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0) n = A->nrows;
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            dpotrs_(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB,
                &ldB, &info);
            break;

        case COMPLEX:
            zpotrs_(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB,
                &ldB, &info);
	    break;

        default:
	    err_invalid_id;
    }
    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Inverse of a real symmetric or complex Hermitian positive definite matrix
 * 
 * potri(A, uplo='L', n=A.size[0], ldA=max(1,A.size[0]), offsetA=0)
 * 
 * @details
 * Computes the inverse of a real symmetric or complex Hermitian
 * positive definite matrix of order n. On entry, A contains the
 * Cholesky factor, as returned by posv() or potrf(). On exit it is
 * replaced by the inverse.
 *
 * @param[in,out] A       Input/output matrix ('d' or 'z' type)
 * @param[in] uplo        Triangle selection ('L' for lower, 'U' for upper) (default = 'L')
 * @param[in] n           Matrix order (default: A.nrows, -1 uses default)
 * @param[in] ldA         Leading dimension (≥ max(1,n), 0 uses default)  
 * @param[in] offsetA     Matrix offset (nonnegative) (default = 0)
 *
 * @note
 * - Requires Cholesky factorization as input (from potrf or posv)
 * - Overwrites input matrix with its inverse
 * - Handles both real symmetric and complex Hermitian matrices
 * - Only specified triangle (upper/lower) is used and updated
 *
 * @warning
 * - Matrix must contain valid Cholesky factorization
 * - ldA must satisfy ldA ≥ max(1,n)
 * - offsetA must be nonnegative
 * - Undefined behavior if input is not a valid Cholesky factor
 */
void lapack_potri(matrix* A, char uplo, int n, int ldA, int offsetA)
{
    // int n=-1, ldA=0, oA=0, info;
    int oA = offsetA, info;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0) n = A->nrows;
    if (n == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

    switch (MAT_ID(A)){
        case DOUBLE:
            dpotri_(&uplo, &n, MAT_BUFD(A)+oA, &ldA, &info);
            break;

        case COMPLEX:
            zpotri_(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, &info);
            break;

        default:
            err_invalid_id;
    }
    if (info) err_lapack(info);
    else return;
}


// static char doc_posv[] =
//     "Solves a real symmetric or complex Hermitian positive definite set\n"
//     "of linear equations.\n\n"
//     "posv(A, B, uplo='L', n=A.size[0], nrhs=B.size[1], \n"
//     "     ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, \n"
//     "     offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X = B with A n by n, real symmetric or complex Hermitian,\n"
//     "and positive definite, and B n by nrhs.\n"
//     "On exit, if uplo is 'L',  the lower triangular part of A is\n"
//     "replaced by L.  If uplo is 'U', the upper triangular part is\n"
//     "replaced by L^H.  B is replaced by the solution.\n\n"
//     "ARGUMENTS.\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* posv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B;
//     int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "B", "uplo", "n", "nrhs", "ldA", "ldB",
//         "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciiiiii", kwlist,
//         &A, &B, &uplo_, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciiiiii", kwlist,
//         &A, &B, &uplo, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0) n = A->nrows;
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1, n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dposv_(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB,
//                 &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zposv_(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB,
//                 &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_pbtrf[] =
//     "Cholesky factorization of a real symmetric or complex Hermitian\n"
//     "positive definite band matrix.\n\n"
//     "pbtrf(A, uplo='L', n=A.size[1], kd=A.size[0]-1, ldA=max(1,A.size[0]),"
//     "\n"
//     "      offsetA=0)\n\n"
//     "PURPOSE\n"
//     "Factors A as A=L*L^T or A = L*L^H, where A is an n by n real\n"
//     "symmetric or complex Hermitian positive definite band matrix with\n"
//     "kd subdiagonals and kd superdiagonals.  A is stored in the BLAS \n"
//     "format for symmetric band matrices.  On exit, A contains the\n"
//     "Cholesky factor in the BLAS format for triangular band matrices.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix.\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "kd        nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* pbtrf(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A;
//     int n=-1, kd=-1, ldA=0, oA=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "uplo", "n", "kd", "ldA", "offsetA", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|Ciiii", kwlist, &A,
//         &uplo_, &n, &kd, &ldA, &oA))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|ciiii", kwlist, &A,
//         &uplo, &n, &kd, &ldA, &oA))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (n < 0) n = A->ncols;
//     if (n == 0) return Py_BuildValue("");
//     if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
//     if (kd < 0) kd = A->nrows - 1;
//     if (kd < 0) err_nn_int("kd");
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA < kd+1) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dpbtrf_(&uplo, &n, &kd, MAT_BUFD(A)+oA, &ldA, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zpbtrf_(&uplo, &n, &kd, MAT_BUFZ(A)+oA, &ldA, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_pbtrs[] =
//     "Solves a real symmetric or complex Hermitian positive definite set\n"
//     "of linear equations with a banded coefficient matrix, given the\n"
//     "Cholesky factorization computed by pbtrf() or pbsv().\n\n"
//     "pbtrs(A, B, uplo='L', n=A.size[1], kd=A.size[0]-1, nrhs=B.size[1],\n"
//     "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
//     "      offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X = B where A is an n by n real symmetric or complex \n"
//     "Hermitian positive definite band matrix with kd subdiagonals and kd\n"
//     "superdiagonals, and B is n by nrhs.  A contains the Cholesky factor\n"
//     "of A, as returned by pbtrf() or pbtrs().  On exit, B is replaced by\n"
//     "the solution X.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix.\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "kd        nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* pbtrs(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B;
//     int n=-1, kd=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "B", "uplo", "n", "kd", "nrhs", "ldA", "ldB",
//         "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciiiiiii", kwlist,
//         &A, &B, &uplo_, &n, &kd, &nrhs, &ldA, &ldB, &oA, oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciiiiiii", kwlist,
//         &A, &B, &uplo, &n, &kd, &nrhs, &ldA, &ldB, &oA, oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
//     if (n < 0) n = A->ncols;
//     if (kd < 0) kd = A->nrows - 1;
//     if (kd < 0) err_nn_int("kd");
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < kd+1) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dpbtrs_(&uplo, &n, &kd, &nrhs, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zpbtrs_(&uplo, &n, &kd, &nrhs, MAT_BUFZ(A)+oA, &ldA,
//                 MAT_BUFZ(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_pbsv[] =
//     "Solves a real symmetric or complex Hermitian positive definite set\n"
//     "of linear equations with a banded coefficient matrix.\n\n"
//     "pbsv(A, B, uplo='L', n=A.size[1], kd=A.size[0]-1, nrhs=B.size[1],\n"
//     "     ldA=MAX(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
//     "     offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X = B where A is an n by n real symmetric or complex\n"
//     "Hermitian positive definite band matrix with kd subdiagonals and kd\n"
//     "superdiagonals, and B is n by nrhs.\n"
//     "On entry, A contains A in the BLAS format for symmetric band\n"
//     "matrices.  On exit, A is replaced with the Cholesky factors, stored\n"
//     "in the BLAS format for triangular band matrices.  B is replaced\n"
//     "by the solution X.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix.\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "kd        nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer. If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* pbsv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B;
//     int n=-1, kd=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "B", "uplo", "n", "kd", "nrhs", "ldA", "ldB",
//         "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciiiiiii", kwlist,
//         &A, &B, &uplo_, &n, &kd, &nrhs, &ldA, &ldB, &oA, oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciiiiiii", kwlist,
//         &A, &B, &uplo, &n, &kd, &nrhs, &ldA, &ldB, &oA, oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
//     if (n < 0) n = A->ncols;
//     if (kd < 0) kd = A->nrows - 1;
//     if (kd < 0) err_nn_int("kd");
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < kd+1) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dpbsv_(&uplo, &n, &kd, &nrhs, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zpbsv_(&uplo, &n, &kd, &nrhs, MAT_BUFZ(A)+oA, &ldA,
//                 MAT_BUFZ(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_pttrf[] =
//     "Cholesky factorization of a real symmetric or complex Hermitian\n"
//     "positive definite tridiagonal matrix.\n\n"
//     "pttrf(d, e, n=len(d)-offsetd, offsetd=0, offsete=0)\n\n"
//     "PURPOSE\n"
//     "Factors A  as A = L*D*L^T or A = L*D*L^H where A is n by n, real\n"
//     "symmetric or complex Hermitian, positive definite, and tridiagonal.\n"
//     "On entry, d is the subdiagonal of A and e is the diagonal.  On \n"
//     "exit, d contains the diagonal of D and e contains the subdiagonal\n"
//     "of the unit bidiagonal matrix L.\n\n"
//     "ARGUMENTS.\n"
//     "d         'd' matrix\n\n"
//     "e         'd' or 'z' matrix.\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "offsetd   nonnegative integer\n\n"
//     "offsete   nonnegative integer";

// static PyObject* pttrf(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *d, *e;
//     int n=-1, od=0, oe=0, info;
//     static char *kwlist[] = {"d", "e", "n", "offsetd", "offsete", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iii", kwlist, &d,
//         &e, &n, &od, &oe)) return NULL;

//     if (!Matrix_Check(d)) err_mtrx("d");
//     if (MAT_ID(d) != DOUBLE) err_type("d");
//     if (!Matrix_Check(e)) err_mtrx("e");
//     if (od < 0) err_nn_int("offsetd");
//     if (n < 0) n = len(d) - od;
//     if (n < 0) err_buf_len("d");
//     if (od + n > len(d)) err_buf_len("d");
//     if (n == 0) return Py_BuildValue("");
//     if (oe < 0) err_nn_int("offsete");
//     if (oe + n - 1  > len(e)) err_buf_len("e");

//     switch (MAT_ID(e)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dpttrf_(&n, MAT_BUFD(d)+od, MAT_BUFD(e)+oe, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zpttrf_(&n, MAT_BUFD(d)+od, MAT_BUFZ(e)+oe, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_pttrs[] =
//     "Solves a real symmetric or complex Hermitian positive definite set\n"
//     "of linear equations with a tridiagonal coefficient matrix, given \n"
//     "the factorization computed by pttrf().\n\n"
//     "pttrs(d, e, B, uplo='L', n=len(d)-offsetd, nrhs=B.size[1],\n"
//     "      ldB=max(1,B.size[0], offsetd=0, offsete=0, offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X=B with A n by n real or complex Hermitian positive\n"
//     "definite and tridiagonal, and B n by nrhs.  On entry, d and e\n"
//     "contain the Cholesky factorization L*D*L^T or L*D*L^H, for example,\n"
//     "as returned by pttrf().  The argument d is the diagonal of the \n"
//     "diagonal matrix D.  The argument uplo only matters in the complex\n"
//     "case.  If uplo = 'L', then e is the subdiagonal of L.  If uplo='U',\n"
//     "e is the superdiagonal of L^H.  On exit B is overwritten with the\n"
//     "solution X. \n\n"
//     "ARGUMENTS.\n"
//     "d         'd' matrix\n\n"
//     "e         'd' or 'z' matrix.\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as e.\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetd   nonnegative integer\n\n"
//     "offsete   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* pttrs(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *d, *e, *B;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     int n=-1, nrhs=-1, ldB=0, od=0, oe=0, oB=0, info;
//     static char *kwlist[] = {"d", "e", "B", "uplo", "n", "nrhs", "ldB",
//         "offsetd", "offsete", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Ciiiiii", kwlist,
//         &d, &e, &B, &uplo_, &n, &nrhs, &ldB, &od, &oe, &oB)) 
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ciiiiii", kwlist,
//         &d, &e, &B, &uplo, &n, &nrhs, &ldB, &od, &oe, &oB)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(d)) err_mtrx("d");
//     if (MAT_ID(d) != DOUBLE) err_type("d");
//     if (!Matrix_Check(e)) err_mtrx("e");
//     if (MAT_ID(e) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (od < 0) err_nn_int("offsetd");
//     if (n < 0) n = len(d) - od;
//     if (n < 0) err_buf_len("d");
//     if (od + n > len(d)) err_buf_len("d");
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (oe < 0) err_nn_int("offsete");
//     if (oe + n - 1  > len(e)) err_buf_len("e");
//     if (oB < 0) err_nn_int("offsetB");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1, n)) err_ld("ldB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

//     switch (MAT_ID(e)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dpttrs_(&n, &nrhs, MAT_BUFD(d)+od, MAT_BUFD(e)+oe,
//                 MAT_BUFD(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zpttrs_(&uplo, &n, &nrhs, MAT_BUFD(d)+od, MAT_BUFZ(e)+oe,
//                 MAT_BUFZ(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_ptsv[] =
//     "Solves a real symmetric or complex Hermitian positive definite set\n"
//     "of linear equations with a tridiagonal coefficient matrix.\n\n"
//     "ptsv(d, e, B, n=len(d)-offsetd, nrhs=B.size[1], ldB=max(1,B.size[0],"
//     "\n"
//     "     offsetd=0, offsete=0, offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X=B with A n by n real or complex Hermitian positive\n"
//     "definite and tridiagonal.  A is specified by its diagonal d and\n"
//     "subdiagonal e.  On exit B is overwritten with the solution, and d\n"
//     "and e are overwritten with the elements of Cholesky factorization\n"
//     "of A.\n\n"
//     "ARGUMENTS.\n"
//     "d         'd' matrix\n\n"
//     "e         'd' or 'z' matrix.\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as e.\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetd   nonnegative integer\n\n"
//     "offsete   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* ptsv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *d, *e, *B;
//     int n=-1, nrhs=-1, ldB=0, od=0, oe=0, oB=0, info;
//     static char *kwlist[] = {"d", "e", "B", "n", "nrhs", "ldB", "offsetd",
//         "offsete", "offsetB", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iiiiii", kwlist,
//         &d, &e, &B, &n, &nrhs, &ldB, &od, &oe, &oB)) return NULL;

//     if (!Matrix_Check(d)) err_mtrx("d");
//     if (MAT_ID(d) != DOUBLE) err_type("d");
//     if (!Matrix_Check(e)) err_mtrx("e");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(e) != MAT_ID(B)) err_conflicting_ids;
//     if (od < 0) err_nn_int("offsetd");
//     if (n < 0) n = len(d) - od;
//     if (n < 0) err_buf_len("d");
//     if (od + n > len(d)) err_buf_len("d");
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (oe < 0) err_nn_int("offsete");
//     if (oe + n - 1  > len(e)) err_buf_len("e");
//     if (oB < 0) err_nn_int("offsetB");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1, n)) err_ld("ldB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

//     switch (MAT_ID(e)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dptsv_(&n, &nrhs, MAT_BUFD(d)+od, MAT_BUFD(e)+oe,
//                 MAT_BUFD(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zptsv_(&n, &nrhs, MAT_BUFD(d)+od, MAT_BUFZ(e)+oe,
//                 MAT_BUFZ(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_sytrf[] =
//     "LDL^T factorization of a real or complex symmetric matrix.\n\n"
//     "sytrf(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]))\n\n"
//     "PURPOSE\n"
//     "Computes the LDL^T factorization of a real or complex symmetric\n"
//     "n by n matrix  A.  On exit, A and ipiv contain the details of the\n"
//     "factorization.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "ipiv      'i' matrix of length at least n\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* sytrf(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *ipiv;
//     void *work;
//     number wl;
//     int n=-1, ldA=0, oA=0, info, lwork;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciii", kwlist,
//         &A, &ipiv, &uplo_, &n, &ldA, &oA)) 
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciii", kwlist,
//         &A, &ipiv, &uplo, &n, &ldA, &oA)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (len(ipiv) < n) err_buf_len("ipiv");

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int *ipiv_ptr = malloc(n*sizeof(int));
//     if (!ipiv_ptr) return PyErr_NoMemory();
// #else
//     int *ipiv_ptr = MAT_BUFI(ipiv);
// #endif

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsytrf_(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double)))){
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dsytrf_(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zsytrf_(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t)))){
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zsytrf_(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
//                 (complex_t *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         default:
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//             free(ipiv_ptr);
// #endif
//             err_invalid_id;
//     }

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int i;  for (i=0; i<n; i++)  MAT_BUFI(ipiv)[i] = ipiv_ptr[i];
//     free(ipiv_ptr);
// #endif
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_hetrf[] =
//     "LDL^H factorization of a real symmetric or complex Hermitian matrix."
//     "\n\n"
//     "hetrf(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]))\n\n"
//     "PURPOSE\n"
//     "Computes the LDL^H factorization of a real symmetric or complex\n"
//     "Hermitian n by n matrix  A.  On exit, A and ipiv contain the\n"
//     "details of the factorization.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "ipiv      'i' matrix of length at least n\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* hetrf(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *ipiv;
//     void *work;
//     number wl;
//     int n=-1, ldA=0, oA=0, info, lwork;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciii", kwlist,
//         &A, &ipiv, &uplo_, &n, &ldA, &oA)) 
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciii", kwlist,
//         &A, &ipiv, &uplo, &n, &ldA, &oA)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (len(ipiv) < n) err_buf_len("ipiv");

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int *ipiv_ptr = malloc(n*sizeof(int));
//     if (!ipiv_ptr) return PyErr_NoMemory();
// #else
//     int *ipiv_ptr = MAT_BUFI(ipiv);
// #endif

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsytrf_(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double)))){
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dsytrf_(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zhetrf_(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t)))){
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zhetrf_(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
//                 (complex_t *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         default:
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//             free(ipiv_ptr);
// #endif
//             err_invalid_id;
//     }

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int i;  for (i=0; i<n; i++)  MAT_BUFI(ipiv)[i] = ipiv_ptr[i];
//     free(ipiv_ptr);
// #endif
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_sytrs[] =
//     "Solves a real or complex symmetric set of linear equations,\n"
//     "given the LDL^T factorization computed by sytrf() or sysv().\n\n"
//     "sytrs(A, ipiv, B, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
//     "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
//     "      offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X = B where A is real or complex symmetric and n by n,\n"
//     "and B is n by nrhs.  On entry, A and ipiv contain the\n"
//     "factorization of A as returned by sytrf() or sysv().  On exit, B is\n"
//     "replaced by the solution.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "ipiv      'i' matrix \n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* sytrs(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B, *ipiv;
//     int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "ipiv", "B", "uplo", "n", "nrhs", "ldA", "ldB",
//         "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Ciiiiii", kwlist,
//         &A, &ipiv, &B, &uplo_, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ciiiiii", kwlist,
//         &A, &ipiv, &B, &uplo, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
//     if (len(ipiv) < n) err_buf_len("ipiv");

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int *ipiv_ptr = malloc(n*sizeof(int));
//     if (!ipiv_ptr) return PyErr_NoMemory();
//     int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
// #else
//     int *ipiv_ptr = MAT_BUFI(ipiv);
// #endif

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dsytrs_(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
//                 MAT_BUFD(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

// 	case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zsytrs_(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
//                 MAT_BUFZ(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

// 	default:
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//             free(ipiv_ptr);
// #endif
// 	    err_invalid_id;
//     }

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     free(ipiv_ptr);
// #endif
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_hetrs[] =
//     "Solves a real symmetric or complex Hermitian set of linear\n"
//     "equations, given the LDL^H factorization computed by hetrf() or "
//     "hesv().\n\n"
//     "hetrs(A, ipiv, B, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
//     "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
//     "      offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X = B where A is real symmetric or complex Hermitian\n"
//     "and n by n, and B is n by nrhs.  On entry, A and ipiv contain\n"
//     "the factorization of A as returned by hetrf or hesv.  On exit, B\n"
//     "is replaced by the solution.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "ipiv      'i' matrix \n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "uplo      'U' or 'L'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* hetrs(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B, *ipiv;
//     int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ ='L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "ipiv", "B", "uplo", "n", "nrhs", "ldA", "ldB",
//         "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Ciiiiii", kwlist,
//         &A, &ipiv, &B, &uplo_, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ciiiiii", kwlist,
//         &A, &ipiv, &B, &uplo, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
//     if (len(ipiv) < n) err_buf_len("ipiv");

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int *ipiv_ptr = malloc(n*sizeof(int));
//     if (!ipiv_ptr) return PyErr_NoMemory();
//     int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
// #else
//     int *ipiv_ptr = MAT_BUFI(ipiv);
// #endif

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dsytrs_(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA,
//                 ipiv_ptr, MAT_BUFD(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

// 	case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zhetrs_(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA,
//                 ipiv_ptr, MAT_BUFZ(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

// 	default:
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//             free(ipiv_ptr);
// #endif
// 	    err_invalid_id;
//     }

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     free(ipiv_ptr);
// #endif
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_sytri[] =
//     "Inverse of a real or complex symmetric matrix.\n\n"
//     "sytri(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]),\n"
//     "      offsetA=0)\n\n"
//     "PURPOSE\n"
//     "Computes the inverse of a real or complex symmetric matrix of\n"
//     "order n.  On entry, A and ipiv contain the LDL^T factorization,\n"
//     "as returned by sysv() or sytrf().  On exit A is replaced by the\n"
//     "inverse.  \n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "ipiv      'i' matrix \n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* sytri(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *ipiv;
//     int n=-1, ldA=0, oA=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     void *work;
//     char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciii", kwlist,
//         &A, &ipiv, &uplo_, &n, &ldA, &oA))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciii", kwlist,
//         &A, &ipiv, &uplo, &n, &ldA, &oA))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (len(ipiv) < n) err_buf_len("ipiv");

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int *ipiv_ptr = malloc(n*sizeof(int));
//     if (!ipiv_ptr) return PyErr_NoMemory();
//     int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
// #else
//     int *ipiv_ptr = MAT_BUFI(ipiv);
// #endif

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             if (!(work = (void *) calloc(n, sizeof(double)))) {
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dsytri_(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
//                 (double *) work, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

// 	case COMPLEX:
//             if (!(work = (void *) calloc(2*n, sizeof(complex_t)))){
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zsytri_(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
//                 (complex_t *) work, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         default:
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//             free(ipiv_ptr);
// #endif
//             err_invalid_id;
//     }

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     free(ipiv_ptr);
// #endif
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_hetri[] =
//     "Inverse of a real symmetric or complex Hermitian matrix.\n\n"
//     "hetri(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]),\n"
//     "      offsetA=0)\n\n"
//     "PURPOSE\n"
//     "Computes the inverse of a real symmetric or complex Hermitian\n"
//     "matrix of order n.  On entry, A and ipiv contain the LDL^T\n"
//     "factorization, as returned by hesv() or hetrf().  On exit A is\n"
//     "replaced by the inverse. \n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "ipiv      'i' matrix \n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* hetri(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *ipiv;
//     int n=-1, ldA=0, oA=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     void *work;
//     char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciii", kwlist,
//         &A, &ipiv, &uplo_, &n, &ldA, &oA))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciii", kwlist,
//         &A, &ipiv, &uplo, &n, &ldA, &oA))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (len(ipiv) < n) err_buf_len("ipiv");

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int *ipiv_ptr = malloc(n*sizeof(int));
//     if (!ipiv_ptr) return PyErr_NoMemory();
//     int i;  for (i=0; i<n; i++) ipiv_ptr[i] = MAT_BUFI(ipiv)[i];
// #else
//     int *ipiv_ptr = MAT_BUFI(ipiv);
// #endif

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             if (!(work = (void *) calloc(n, sizeof(double)))){
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dsytri_(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
//                 (double *) work, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         case COMPLEX:
//             if (!(work = (void *) calloc(n, sizeof(complex_t)))){
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 free(ipiv_ptr);
// #endif
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zhetri_(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
//                 (complex_t *) work, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         default:
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//             free(ipiv_ptr);
// #endif
//             err_invalid_id;
//     }

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     free(ipiv_ptr);
// #endif
//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_sysv[] =
//     "Solves a real or complex symmetric set of linear equations.\n\n"
//     "sysv(A, B, ipiv=None, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
//     "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]),\n"
//     "     offsetA=0, offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X = B where A is real or complex symmetric and n by n.\n"
//     "If ipiv is provided, then on exit A and ipiv contain the details\n"
//     "of the LDL^T factorization of A.  If ipiv is not provided, then\n"
//     "the factorization is not returned and A is not modified.  On\n"
//     "exit, B contains the solution.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "ipiv      'i' matrix of length at least n\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* sysv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B, *ipiv=NULL;
//     int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, lwork, k,
//         *ipivc=NULL;
//     void *work=NULL, *Ac=NULL;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "B", "ipiv", "uplo", "n", "nrhs", "ldA",
//         "ldB", "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OCiiiiii", kwlist,
//         &A, &B, &ipiv, &uplo_, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ociiiiii", kwlist,
//         &A, &B, &ipiv, &uplo, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
//         err_int_mtrx("ipiv");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1, n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
//     if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsytrf_(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             if (ipiv) {
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 if (!(ipivc = (int *) calloc(n, sizeof(int)))){
//                     free(work);
//                     return PyErr_NoMemory();
//                 }
//                 for (k=0; k<n; k++) ipivc[k] = MAT_BUFI(ipiv)[k];
// #else
//                 ipivc = MAT_BUFI(ipiv);
// #endif
//                 Py_BEGIN_ALLOW_THREADS
//                 dsysv_(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
//                     MAT_BUFD(B)+oB, &ldB, (double *) work, &lwork,
//                     &info);
//                 Py_END_ALLOW_THREADS
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
// 		for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
//                 free(ipivc);
// #endif
// 	    }
//             else {
//                 ipivc = (int *) calloc(n, sizeof(int));
//                 Ac = (void *) calloc(n*n, sizeof(double));
//                 if (!ipivc || !Ac){
//                     free(work);  free(ipivc);  free(Ac);
//                     return PyErr_NoMemory();
//                 }
//                 for (k=0; k<n; k++)
//                     memcpy((double *) Ac + k*n, MAT_BUFD(A) + oA + k*ldA,
//                         n*sizeof(double));
//                 Py_BEGIN_ALLOW_THREADS
//                 dsysv_(&uplo, &n, &nrhs, (double *) Ac, &n, ipivc,
//                     MAT_BUFD(B)+oB, &ldB, work, &lwork, &info);
//                 Py_END_ALLOW_THREADS
//                 free(ipivc); free(Ac);
//             }
//             free(work);
//             break;

//         case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zsytrf_(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             if (ipiv) {
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 if (!(ipivc = (int *) calloc(n, sizeof(int)))){
//                     free(work);
//                     return PyErr_NoMemory();
//                 }
//                 for (k=0; k<n; k++) ipivc[k] = MAT_BUFI(ipiv)[k];
// #else
//                 ipivc = MAT_BUFI(ipiv);
// #endif
//                 Py_BEGIN_ALLOW_THREADS
//                 zsysv_(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
//                     MAT_BUFZ(B)+oB, &ldB, (complex_t *) work, 
//                     &lwork, &info);
//                 Py_END_ALLOW_THREADS
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
//                 free(ipivc);
// #endif
//             }
//             else {
//                 ipivc = (int *) calloc(n, sizeof(int));
//                 Ac = (void *) calloc(n*n, sizeof(complex_t));
//                 if (!ipivc || !Ac){
//                     free(work);  free(ipivc);  free(Ac);
//                     return PyErr_NoMemory();
//                 }
//                 for (k=0; k<n; k++)
//                     memcpy((complex_t *) Ac + k*n, 
//                         MAT_BUFZ(A) + oA + k*ldA,
//                         n*sizeof(complex_t));
//                 Py_BEGIN_ALLOW_THREADS
//                 zsysv_(&uplo, &n, &nrhs, (complex_t *) Ac, &n, ipivc,
//                     MAT_BUFZ(B)+oB, &ldB, work, &lwork, &info);
//                 Py_END_ALLOW_THREADS
//                 free(ipivc);  free(Ac);
//             }
//             free(work);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_hesv[] =
//     "Solves a real symmetric or complex Hermitian set of linear\n"
//     "equations.\n\n"
//     "herv(A, B, ipiv=None, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
//     "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]), offsetA=0,\n"
//     "     offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Solves A*X=B where A is real symmetric or complex Hermitian and\n"
//     "n by n.  If ipiv is provided, then on exit A and ipiv contain\n"
//     "the details of the LDL^H factorization of A.  If ipiv is not\n"
//     "provided, then the factorization is not returned and A is not\n"
//     "modified.  On exit, B contains the solution.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "ipiv      'i' matrix of length at least n\n\n"
//     "uplo      'U' or 'L'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* hesv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B, *ipiv=NULL;
//     int n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, lwork, k,
//         *ipivc=NULL;
//     void *work=NULL, *Ac=NULL;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L';
// #endif
//     char uplo = 'L';
//     char *kwlist[] = {"A", "B", "ipiv", "uplo", "n", "nrhs", "ldA", "ldB",
//         "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OCiiiiii", kwlist,
//         &A, &B, &ipiv, &uplo_, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ociiiiii", kwlist,
//         &A, &B, &ipiv, &uplo, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
//         err_int_mtrx("ipiv");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1, n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
//     if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsytrf_(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             if (ipiv) {
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 if (!(ipivc = (int *) calloc(n,sizeof(int)))){
//                     free(work);
//                     return PyErr_NoMemory();
//                 }
//                 int i; for (i=0; i<n; i++) ipivc[i] = MAT_BUFI(ipiv)[i];
// #else
//                 ipivc = MAT_BUFI(ipiv);
// #endif
//                 Py_BEGIN_ALLOW_THREADS
//                 dsysv_(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
//                     MAT_BUFD(B)+oB, &ldB, (double *) work, &lwork,
//                     &info);
//                 Py_END_ALLOW_THREADS
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 for (i=0; i<n; i++) MAT_BUFI(ipiv)[i] = ipivc[i];
//                 free(ipivc);
// #endif
//             }
//             else {
//                 ipivc = (int *) calloc(n, sizeof(int));
//                 Ac = (void *) calloc(n*n, sizeof(double));
//                 if (!ipivc || !Ac){
//                     free(work);  free(ipivc);  free(Ac);
//                     return PyErr_NoMemory();
//                 }
//                 for (k=0; k<n; k++)
//                     memcpy((double *) Ac + k*n, MAT_BUFD(A) + oA + k*ldA,
//                         n*sizeof(double));
//                 Py_BEGIN_ALLOW_THREADS
//                 dsysv_(&uplo, &n, &nrhs, (double *) Ac, &n, ipivc,
//                     MAT_BUFD(B)+oB, &ldB, work, &lwork, &info);
//                 Py_END_ALLOW_THREADS
//                 free(ipivc);  free(Ac);
//             }
//             free(work);
//             break;

//         case COMPLEX:
//             lwork = -1;
//             zhetrf_(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             if (ipiv) {
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 if (!(ipivc = (int *) calloc(n,sizeof(int)))){
//                     free(work);
//                     return PyErr_NoMemory();
//                 }
//                 for (k=0; k<n; k++) ipivc[k] = MAT_BUFI(ipiv)[k];
// #else
//                 ipivc = MAT_BUFI(ipiv);
// #endif
//                 Py_BEGIN_ALLOW_THREADS
//                 zhesv_(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
//                     MAT_BUFZ(B)+oB, &ldB, (complex_t *) work, 
//                     &lwork, &info);
//                 Py_END_ALLOW_THREADS
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//                 for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
//                 free(ipivc);
// #endif
//             }
//             else {
//                 ipivc = (int *) calloc(n, sizeof(int));
//                 Ac = (void *) calloc(n*n, sizeof(complex_t));
//                 if (!ipivc || !Ac){
//                     free(work);  free(ipivc);  free(Ac);
//                     return PyErr_NoMemory();
//                 }
//                 for (k=0; k<n; k++)
//                     memcpy((complex_t *) Ac + k*n, 
//                         MAT_BUFZ(A) + oA + k*ldA,
//                         n*sizeof(complex_t));
//                 Py_BEGIN_ALLOW_THREADS
//                 zhesv_(&uplo, &n, &nrhs, (complex_t *) Ac, &n, ipivc,
//                     MAT_BUFZ(B)+oB, &ldB, work, &lwork, &info);
//                 Py_END_ALLOW_THREADS
//                 free(ipivc);  free(Ac);
//             }
//             free(work);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }



/**
 * @brief Solves triangular systems with multiple right-hand sides (BLAS TRTRS operation)
 * 
 * @details
 * Solves one of the following triangular systems:
 * \f[
 * \begin{cases}
 * A X = B       & \text{if trans='N'} \\
 * A^\top X = B  & \text{if trans='T'} \\
 * A^H X = B     & \text{if trans='C'}
 * \end{cases}
 * \f]
 * where:
 * - A is an n×n triangular matrix
 * - B is an n×nrhs matrix of right-hand sides
 * - X is the n×nrhs solution matrix (overwrites B)
 *
 * @param A       Triangular matrix ('d' or 'z' type)
 * @param B       Right-hand side matrix (input) / solution matrix (output)
 * @param uplo    Triangle selection:  (default = 'L')
 *                    - 'L': Use lower triangular part
 *                    - 'U': Use upper triangular part
 * @param trans   Transposition operation:  (default = 'N')
 *                    - 'N': No transpose
 *                    - 'T': Transpose
 *                    - 'C': Conjugate transpose
 * @param diag    Diagonal type:  (default = 'N')
 *                    - 'N': Non-unit triangular
 *                    - 'U': Unit triangular
 * @param n       Order of matrix A (default = -1, which uses A's size)
 * @param nrhs    Number of right-hand sides (default = -1, which uses B's size)
 * @param ldA     Leading dimension of A (default = 0, which uses max(1,n))
 * @param ldB     Leading dimension of B (default = 0, which uses max(1,n))
 * @param offsetA Matrix offset in A (default = 0, nonnegative)
 * @param offsetB Matrix offset in B (default = 0, nonnegative)
 */
void lapack_trtrs(matrix *A, matrix *B, char uplo, char trans, char diag, 
                int n, int nrhs, int ldA, int ldB, int offsetA, int offsetB)
{
    int oA = offsetA, oB = offsetB, info;

    // Set default values
    if (uplo == 0) uplo = 'L';
    if (trans == 0) trans = 'N';
    if (diag == 0) diag = 'N';

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (n < 0){
        n = A->nrows;
        if (A->nrows != A->ncols){
            fprintf(stderr, "Error: A must be square\n");
            return;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, MAT_BUFD(A)+oA,
                &ldA, MAT_BUFD(B)+oB, &ldB, &info);
            break;

        case COMPLEX:
            ztrtrs_(&uplo, &trans, &diag, &n, &nrhs, MAT_BUFZ(A)+oA,
                &ldA, MAT_BUFZ(B)+oB, &ldB, &info);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack(info);
    else return;
}


// static char doc_trtri[] =
//     "Inverse of a triangular matrix.\n\n"
//     "trtri(A, uplo='L', diag='N', n=A.size[0], ldA=max(1,A.size[0]),\n"
//     "      offsetA=0)\n\n"
//     "PURPOSE\n"
//     "Computes the inverse of a triangular matrix of order n.\n"
//     "On exit, A is replaced with its inverse.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "diag      'N' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* trtri(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A;
//     int n=-1, ldA=0, oA=0, info;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', diag_ = 'N';
// #endif
//     char uplo = 'L', diag = 'N';
//     char *kwlist[] = {"A", "uplo", "diag", "n", "ldA", "offsetA", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|CCiii", kwlist,
//         &A, &uplo_, &diag_, &n, &ldA, &oA)) return NULL;
//     uplo = (char) uplo_;
//     diag = (char) diag_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|cciii", kwlist,
//         &A, &uplo, &diag, &n, &ldA, &oA)) return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (A->nrows != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dtrtri_(&uplo, &diag, &n, MAT_BUFD(A)+oA, &ldA, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             ztrtri_(&uplo, &diag, &n, MAT_BUFZ(A)+oA, &ldA, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_tbtrs[] =
//     "Solution of a triangular set of equations with banded coefficient\n"
//     "matrix.\n\n"
//     "tbtrs(A, B, uplo='L', trans='N', diag='N', n=A.size[1], \n"
//     "      kd=A.size[0]-1, nrhs=B.size[1], ldA=max(1,A.size[0]),\n"
//     "      ldB=max(1,B.size[0]), offsetA=0, offsetB=0)\n\n"
//     "PURPOSE\n"
//     "If trans is 'N', solves A*X = B.\n"
//     "If trans is 'T', solves A^T*X = B.\n"
//     "If trans is 'C', solves A^H*X = B.\n"
//     "B is n by nrhs and A is a triangular band matrix of order n with kd\n"
//     "subdiagonals (uplo is 'L') or superdiagonals (uplo is 'U').\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "trans     'N', 'T' or 'C'\n\n"
//     "diag      'N' or 'U'\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "kd        nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "nrhs      nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* tbtrs(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', trans_ = 'N', diag_ = 'N';
// #endif
//     char uplo = 'L', trans = 'N', diag = 'N';
//     int n=-1, kd=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
//     char *kwlist[] = {"A", "B", "uplo", "trans", "diag", "n", "kd", "nrhs",
//         "ldA", "ldB", "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCiiiiiii", kwlist,
//         &A, &B, &uplo_, &trans_, &diag_, &n, &kd, &nrhs, &ldA, &ldB, &oA,
//         &oB))
//         return NULL;
//     uplo = (char) uplo_;
//     trans = (char) trans_;
//     diag = (char) diag_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccciiiiiii", kwlist,
//         &A, &B, &uplo, &trans, &diag, &n, &kd, &nrhs, &ldA, &ldB, &oA,
//         &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
//     if (trans != 'N' && trans != 'T' && trans != 'C')
//         err_char("trans", "'N', 'T', 'C'");
//     if (n < 0) n = A->ncols;
//     if (kd < 0) kd = A->nrows - 1;
//     if (kd < 0) err_nn_int("kd");
//     if (nrhs < 0) nrhs = B->ncols;
//     if (n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < kd+1) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             if (trans == 'C') trans = 'T';
//             Py_BEGIN_ALLOW_THREADS
//             dtbtrs_(&uplo, &trans, &diag, &n, &kd, &nrhs, MAT_BUFD(A)+oA,
//                 &ldA, MAT_BUFD(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             ztbtrs_(&uplo, &trans, &diag, &n, &kd, &nrhs, MAT_BUFZ(A)+oA,
//                 &ldA, MAT_BUFZ(B)+oB, &ldB, &info);
//             Py_END_ALLOW_THREADS
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_gels[] =
//     "Solves least-squares and least-norm problems with full rank\n"
//     "matrices.\n\n"
//     "gels(A, B, trans='N', m=A.size[0], n=A.size[1], nrhs=B.size[1],\n"
//     "     ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
//     "     offsetB=0)\n\n"
//     "PURPOSE\n"
//     "1. If trans is 'N' and A and B are real/complex:\n"
//     "- if m >= n: minimizes ||A*X - B||_F.\n"
//     "- if m < n: minimizes ||X||_F subject to A*X = B.\n\n"
//     "2. If trans is 'N' or 'C' and A and B are real:\n"
//     "- if m >= n: minimizes ||X||_F subject to A^T*X = B.\n"
//     "- if m < n: minimizes ||X||_F subject to A^T*X = B.\n\n"
//     "3. If trans is 'C' and A and B are complex:\n"
//     "- if m >= n: minimizes ||X||_F subject to A^H*X = B.\n"
//     "- if m < n: minimizes ||X||_F subject to A^H*X = B.\n\n"
//     "A is an m by n matrix.  B has nrhs columns.  On exit, B is\n"
//     "replaced with the solution, and A is replaced with the details\n"
//     "of its QR or LQ factorization.\n\n"
//     "Note that gels does not check whether A has full rank.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "trans     'N', 'T' or 'C' if A is real.  'N' or 'C' if A is\n"
//     "          complex.\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "nrhs      integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "ldB       nonnegative integer.  ldB >= max(1,m,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* gels(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B;
//     int m=-1, n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, lwork;
//     void *work;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int trans_ = 'N';
// #endif
//     char trans = 'N';
//     char *kwlist[] = {"A", "B", "trans", "m", "n", "nrhs", "ldA", "ldB",
//         "offsetA", "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciiiiiii",
//         kwlist, &A, &B, &trans_, &m, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
//     trans = (char) trans_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciiiiiii",
//         kwlist, &A, &B, &trans, &m, &n, &nrhs, &ldA, &ldB, &oA, &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (trans != 'N' && trans != 'T' && trans != 'C')
//         err_char("trans", "'N', 'T', 'C'");
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = A->ncols;
//     if (nrhs < 0) nrhs = B->ncols;
//     if (m == 0 || n == 0 || nrhs == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,m)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(MAX(1,n),m)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (nrhs-1)*ldB + ((trans == 'N') ? n : m) > len(B))
//         err_buf_len("B");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             if (trans == 'C') trans = 'T';
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dgels_(&trans, &m, &n, &nrhs, NULL, &ldA, NULL, &ldB, &wl.d,
//                 &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dgels_(&trans, &m, &n, &nrhs, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(B)+oB, &ldB, (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         case COMPLEX:
//             if (trans == 'T') err_char("trans", "'N', 'C'");
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zgels_(&trans, &m, &n, &nrhs, NULL, &ldA, NULL, &ldB, &wl.z,
//                 &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zgels_(&trans, &m, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA,
//                 MAT_BUFZ(B)+oB, &ldB, (complex_t *) work, &lwork, 
//                 &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


/**
 * @brief QR factorization of a real or complex matrix (BLAS GEQRF operation)
 * 
 * @details
 * Computes the QR factorization of an m×n matrix A:
 * \f[
 * A = Q \cdot R = \begin{cases}
 * \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \cdot \begin{bmatrix} R_1 \\ 0 \end{bmatrix} & \text{if } m \geq n \\
 * Q \cdot \begin{bmatrix} R_1 & R_2 \end{bmatrix} & \text{if } m < n
 * \end{cases}
 * \f]
 * where:
 * - \f$ Q \f$ is an m×m orthogonal/unitary matrix
 * - \f$ R \f$ is an m×n upper trapezoidal matrix with \f$ R_1 \f$ triangular
 *
 * On exit:
 * - Upper triangular part contains \f$ R \f$
 * - Lower triangular part of first k=min(m,n) columns contains reflectors
 * - Vector tau contains reflector coefficients
 *
 * @param A        Input matrix to factor ('d' or 'z' type)
 * @param tau      Reflector coefficients (same type as A, length ≥ min(m,n))
 * @param m        Rows of matrix A (default = -1, meaning A.size[0])
 * @param n        Columns of matrix A (default = -1, meaning A.size[1])
 * @param ldA      Leading dimension of A (default = 0, meaning max(1,m))
 * @param offsetA  Matrix offset (default = 0)
 */
void lapack_geqrf(matrix *A, matrix *tau, int m, int n, int ldA, int offsetA)
{
    int oA=offsetA, info, lwork;
    void *work;
    number wl;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (len(tau) < MIN(m,n)) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            dgeqrf_(&m, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            lwork = (int) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                err_no_memory;
            dgeqrf_(&m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            free(work);
	    break;

        case COMPLEX:
            lwork = -1;
            zgeqrf_(&m, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            lwork = (int) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                err_no_memory;
            zgeqrf_(&m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(tau),
                (complex_t *) work, &lwork, &info);
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack(info);
    else return;
}


/**
 * @brief Product with real orthogonal matrix Q (BLAS ORMQR operation)
 * 
 * @details
 * Multiplies matrix C by orthogonal matrix Q or its transpose:
 * \f[
 * \begin{align*}
 * C &:= Q \cdot C \quad \text{(side='L', trans='N')} \\
 * C &:= Q^\top \cdot C \quad \text{(side='L', trans='T')} \\
 * C &:= C \cdot Q \quad \text{(side='R', trans='N')} \\
 * C &:= C \cdot Q^\top \quad \text{(side='R', trans='T')}
 * \end{align*}
 * \f]
 * 
 * Matrix Q is defined by elementary reflectors from geqrf():
 * \f[
 * Q = H_1 H_2 \cdots H_k
 * \f]
 * stored in first k columns of A and vector tau.
 *
 * @param[in] A       Matrix containing elementary reflectors ('d' type)
 * @param[in] tau     Scalar factors of elementary reflectors ('d' type)
 * @param[in,out] C   Input/output matrix ('d' type)
 * @param[in] side    Multiplication side: (default = 'L')
 *                    - 'L': apply Q from Left
 *                    - 'R': apply Q from Right
 * @param[in] trans   Transposition flag: (default = 'N')
 *                    - 'N': use Q
 *                    - 'T': use Q transpose
 * @param[in] m       Rows of matrix C (default = -1, meaning C.size[0])
 * @param[in] n       Columns of matrix C (default = -1, meaning C.size[1])
 * @param[in] k       Number of elementary reflectors (default = -1, meaning len(tau))
 *                    - k must be ≤ m if side='L'
 *                    - k must be ≤ n if side='R'   
 * @param[in] ldA     Leading dimension of A: (default = 0, meaning max(1,m))
 *                    - ≥max(1,m) if side='L'
 *                    - ≥max(1,n) if side='R'
 * @param[in] ldC     Leading dimension of C (default = 0, meaning max(1,m))
 * @param[in] offsetA Offset in matrix A (default = 0)
 * @param[in] offsetC Offset in matrix C (default = 0)
 */
void lapack_ormqr(matrix *A, matrix *tau, matrix *C, char side, char trans, 
                int m, int n, int k, int ldA, int ldC, int offsetA, int offsetC)
{
    int info, lwork, oA = offsetA, oC = offsetC;
    void *work;
    number wl;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
        err_conflicting_ids;
    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (trans != 'N' && trans != 'T') err_char("trans", "'N', 'T'");
    if (m < 0) m = C->nrows;
    if (n < 0) n = C->ncols;
    if (k < 0) k = len(tau);
    if (m == 0 || n == 0 || k == 0) return;
    if (k > ((side == 'L') ? m : n)) err_ld("k");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < ((side == 'L') ? MAX(1,m) : MAX(1,n))) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + k*ldA  > len(A)) err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            dormqr_(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
                &ldC, &wl.d, &lwork, &info);
            lwork = (int) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                err_no_memory;
            dormqr_(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
                &lwork, &info);
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack(info);
    else return;
}


// static char doc_unmqr[] =
//     "Product with a real or complex orthogonal matrix.\n\n"
//     "unmqr(A, tau, C, side='L', trans='N', m=C.size[0], n=C.size[1],\n"
//     "      k=len(tau), ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
//     "      offsetA=0, offsetC=0)\n\n"
//     "PURPOSE\n"
//     "Computes\n"
//     "C := Q*C   if side = 'L' and trans = 'N'.\n"
//     "C := Q^T*C if side = 'L' and trans = 'T'.\n"
//     "C := Q^H*C if side = 'L' and trans = 'C'.\n"
//     "C := C*Q   if side = 'R' and trans = 'N'.\n"
//     "C := C*Q^T if side = 'R' and trans = 'T'.\n"
//     "C := C*Q^H if side = 'R' and trans = 'C'.\n"
//     "C is m by n and Q is a square orthogonal/unitary matrix computed\n"
//     "by geqrf.  Q is defined as a product of k elementary reflectors,\n"
//     "stored as the first k columns of A and the first k entries of tau."
//     "\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
//     "          same type as A.\n\n"
//     "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "side      'L' or 'R'\n\n"
//     "trans     'N', 'T', or 'C'n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "k         integer.  k <= m if side = 'R' and k <= n if side = 'L'.\n"
//     "          If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m) if side = 'L'\n"
//     "          and ldA >= max(1,n) if side = 'R'.  If zero, the\n"
//     "          default value is used.\n\n"
//     "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* unmqr(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau, *C;
//     int m=-1, n=-1, k=-1, ldA=0, ldC=0, oA=0, oC=0, info, lwork;
//     void *work;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int side_ = 'L', trans_ = 'N';
// #endif
//     char side = 'L', trans = 'N';
//     char *kwlist[] = {"A", "tau", "C", "side", "trans", "m", "n", "k",
//         "ldA", "ldC", "offsetA", "offsetC", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCiiiiiii",
//         kwlist, &A, &tau, &C, &side_, &trans_, &m, &n, &k, &ldA, &ldC,
//         &oA, &oC)) 
//         return NULL;
//     side = (char) side_;
//     trans = (char) trans_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cciiiiiii",
//         kwlist, &A, &tau, &C, &side, &trans, &m, &n, &k, &ldA, &ldC,
//         &oA, &oC)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (!Matrix_Check(C)) err_mtrx("C");
//     if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
//         err_conflicting_ids;
//     if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
//     if (trans != 'N' && trans != 'T' && trans != 'C')
//         err_char("trans", "'N', 'T', 'C'");
//     if (m < 0) m = C->nrows;
//     if (n < 0) n = C->ncols;
//     if (k < 0) k = len(tau);
//     if (m == 0 || n == 0 || k == 0) return Py_BuildValue("");
//     if (k > ((side == 'L') ? m : n)) err_ld("k");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < ((side == 'L') ? MAX(1,m) : MAX(1,n))) err_ld("ldA");
//     if (ldC == 0) ldC = MAX(1,C->nrows);
//     if (ldC < MAX(1,m)) err_ld("ldC");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + k*ldA > len(A)) err_buf_len("A");
//     if (oC < 0) err_nn_int("offsetC");
//     if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
//     if (len(tau) < k) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             if (trans == 'C') trans = 'T';
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dormqr_(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
//                 &ldC, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dormqr_(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
//                 &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         case COMPLEX:
//             if (trans == 'T') err_char("trans", "'N', 'C'");
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zunmqr_(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
//                 &ldC, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zunmqr_(&side, &trans, &m, &n, &k, MAT_BUFZ(A)+oA, &ldA,
//                 MAT_BUFZ(tau), MAT_BUFZ(C)+oC, &ldC, 
//                 (complex_t *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_orgqr[] =
//     "Generate the orthogonal matrix in a QR factorization.\n\n"
//     "ormqr(A, tau, m=A.size[0], n=min(A.size), k=len(tau), \n"
//     "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
//     "PURPOSE\n"
//     "On entry, A and tau contain an m by m orthogonal matrix Q.\n"
//     "Q is defined as a product of k elementary reflectors, stored in the\n"
//     "first k columns of A and in tau, as computed by geqrf().  On exit,\n"
//     "the first n columns of Q are stored in the leading columns of A.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "tau       'd' matrix of length at least k\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  n <= m.  If negative, the default value is used."
//     "\n\n"
//     "k         integer.  k <= n.  If negative, the default value is \n"
//     "          used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* orgqr(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau;
//     int m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
//     void *work;
//     number wl;
//     char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii", kwlist, &A,
//         &tau, &m, &n, &k, &ldA, &oA)) return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = MIN(A->nrows, A->ncols);
//     if (n > m) err_ld("n");
//     if (k < 0) k = len(tau);
//     if (k > n) err_ld("k");
//     if (m == 0 || n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA <  MAX(1, m)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + n*ldA  > len(A)) err_buf_len("A");
//     if (len(tau) < k) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dorgqr_(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dorgqr_(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_ungqr[] =
//     "Generate the orthogonal or unitary matrix in a QR factorization.\n\n"
//     "ungqr(A, tau, m=A.size[0], n=min(A.size), k=len(tau), \n"
//     "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
//     "PURPOSE\n"
//     "On entry, A and tau contain an m by m orthogonal/unitary matrix Q.\n"
//     "Q is defined as a product of k elementary reflectors, stored in the\n"
//     "first k columns of A and in tau, as computed by geqrf().  On exit,\n"
//     "the first n columns of Q are stored in the leading columns of A.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
//     "          same type as A.\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  n <= m.  If negative, the default value is used."
//     "\n\n"
//     "k         integer.  k <= n.  If negative, the default value is \n"
//     "          used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* ungqr(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau;
//     int m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
//     void *work;
//     number wl;
//     char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii",
//         kwlist, &A, &tau, &m, &n, &k, &ldA, &oA)) return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = MIN(A->nrows, A->ncols);
//     if (n > m) err_ld("n");
//     if (k < 0) k = len(tau);
//     if (k > n) err_ld("k");
//     if (m == 0 || n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA <  MAX(1, m)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + n*ldA  > len(A)) err_buf_len("A");
//     if (len(tau) < k) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dorgqr_(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dorgqr_(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zungqr_(&m, &n, &k, NULL, &ldA, NULL, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zungqr_(&m, &n, &k, MAT_BUFZ(A) + oA, &ldA, MAT_BUFZ(tau),
//                 (complex_t *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_gelqf[] =
//     "LQ factorization.\n\n"
//     "gelqf(A, tau, m=A.size[0], n=A.size[1], ldA=max(1,A.size[0]),\n"
//     "      offsetA=0)\n\n"
//     "PURPOSE\n"
//     "LQ factorization of an m by n real or complex matrix A:\n\n"
//     "A = L*Q = [L1; 0] * [Q1; Q2] if m <= n\n"
//     "A = L*Q = [L1; L2] * Q if m >= n,\n\n"
//     "where Q is n by n and orthogonal/unitary and L is m by n with L1\n"
//     "lower triangular.  On exit, L is stored in the lower triangular\n"
//     "part of A.  Q is stored as a product of k=min(m,n) elementary\n"
//     "reflectors.  The parameters of the reflectors are stored in the\n"
//     "first k entries of tau and in the upper  triangular part of the\n"
//     "first k rows of A.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "tau       'd' or 'z' matrix of length at least min(m,n).  Must\n"
//     "          have the same type as A.\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* gelqf(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau;
//     int m=-1, n=-1, ldA=0, oA=0, info, lwork;
//     void *work;
//     number wl;
//     char *kwlist[] = {"A", "tau", "m", "n", "ldA", "offsetA", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiii", kwlist,
//         &A, &tau, &m, &n, &ldA, &oA))
//         return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = A->ncols;
//     if (m == 0 || n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,m)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
//     if (len(tau) < MIN(m,n)) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dgelqf_(&m, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dgelqf_(&m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(tau),
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zgelqf_(&m, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zgelqf_(&m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(tau),
//                 (complex_t *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_ormlq[] =
//     "Product with a real orthogonal matrix.\n\n"
//     "ormlq(A, tau, C, side='L', trans='N', m=C.size[0], n=C.size[1],\n"
//     "      k=min(A.size), ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
//     "      offsetA=0, offsetC=0)\n\n"
//     "PURPOSE\n"
//     "Computes\n"
//     "C := Q*C   if side = 'L' and trans = 'N'.\n"
//     "C := Q^T*C if side = 'L' and trans = 'T'.\n"
//     "C := C*Q   if side = 'R' and trans = 'N'.\n"
//     "C := C*Q^T if side = 'R' and trans = 'T'.\n"
//     "C is m by n and Q is a square orthogonal matrix computed by gelqf."
//     "\n"
//     "Q is defined as a product of k elementary reflectors, stored as\n"
//     "the first k rows of A and the first k entries of tau.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "tau       'd' matrix of length at least k\n\n"
//     "C         'd' matrix\n\n"
//     "side      'L' or 'R'\n\n"
//     "trans     'N' or 'T'\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "k         integer.  k <= m if side = 'L' and k <= n if side = 'R'.\n"
//     "          If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,k).  If zero, the\n"
//     "          default value is used.\n\n"
//     "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* ormlq(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau, *C;
//     int m=-1, n=-1, k=-1, ldA=0, ldC=0, oA=0, oC=0, info, lwork;
//     void *work;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int side_ = 'L', trans_ = 'N';
// #endif
//     char side = 'L', trans = 'N';
//     char *kwlist[] = {"A", "tau", "C", "side", "trans", "m", "n", "k",
//         "ldA", "ldC", "offsetA", "offsetC", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCiiiiiii",
//         kwlist, &A, &tau, &C, &side_, &trans_, &m, &n, &k, &ldA, &ldC,
//         &oA, &oC)) 
//         return NULL;
//     side = (char) side_;
//     trans = (char) trans_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cciiiiiii",
//         kwlist, &A, &tau, &C, &side, &trans, &m, &n, &k, &ldA, &ldC,
//         &oA, &oC)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (!Matrix_Check(C)) err_mtrx("C");
//     if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
//         err_conflicting_ids;
//     if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
//     if (trans != 'N' && trans != 'T') err_char("trans", "'N', 'T'");
//     if (m < 0) m = C->nrows;
//     if (n < 0) n = C->ncols;
//     if (k < 0) k = MIN(A->nrows, A->ncols);
//     if (m == 0 || n == 0 || k == 0) return Py_BuildValue("");
//     if (k > ((side == 'L') ? m : n)) err_ld("k");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,k)) err_ld("ldA");
//     if (ldC == 0) ldC = MAX(1,C->nrows);
//     if (ldC < MAX(1,m)) err_ld("ldC");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + ldA * ((side == 'L') ? m : n) > len(A)) err_buf_len("A");
//     if (oC < 0) err_nn_int("offsetC");
//     if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
//     if (len(tau) < k) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dormlq_(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
//                 &ldC, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dormlq_(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
//                 &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_unmlq[] =
//     "Product with a real or complex orthogonal matrix.\n\n"
//     "unmlq(A, tau, C, side='L', trans='N', m=C.size[0], n=C.size[1],\n"
//     "      k=min(A.size), ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
//     "      offsetA=0, offsetC=0)\n\n"
//     "PURPOSE\n"
//     "Computes\n"
//     "C := Q*C   if side = 'L' and trans = 'N'.\n"
//     "C := Q^T*C if side = 'L' and trans = 'T'.\n"
//     "C := Q^H*C if side = 'L' and trans = 'C'.\n"
//     "C := C*Q   if side = 'R' and trans = 'N'.\n"
//     "C := C*Q^T if side = 'R' and trans = 'T'.\n"
//     "C := C*Q^H if side = 'R' and trans = 'C'.\n"
//     "C is m by n and Q is a square orthogonal/unitary matrix computed\n"
//     "by gelqf.  Q is defined as a product of k elementary reflectors,\n"
//     "stored as the first k rows of A and the first k entries of tau."
//     "\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
//     "          same type as A.\n\n"
//     "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "side      'L' or 'R'\n\n"
//     "trans     'N', 'T', or 'C'n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "k         integer.  k <= m if side = 'R' and k <= n if side = 'L'.\n"
//     "          If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,k).  If zero, the\n"
//     "          default value is used.\n\n"
//     "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* unmlq(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau, *C;
//     int m=-1, n=-1, k=-1, ldA=0, ldC=0, oA=0, oC=0, info, lwork;
//     void *work;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int side_ = 'L', trans_ = 'N';
// #endif
//     char side = 'L', trans = 'N';
//     char *kwlist[] = {"A", "tau", "C", "side", "trans", "m", "n", "k",
//         "ldA", "ldC", "offsetA", "offsetC", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCiiiiiii",
//         kwlist, &A, &tau, &C, &side_, &trans_, &m, &n, &k, &ldA, &ldC,
//         &oA, &oC)) 
//         return NULL;
//     side = (char) side_;
//     trans = (char) trans_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cciiiiiii",
//         kwlist, &A, &tau, &C, &side, &trans, &m, &n, &k, &ldA, &ldC,
//         &oA, &oC)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (!Matrix_Check(C)) err_mtrx("C");
//     if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
//         err_conflicting_ids;
//     if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
//     if (trans != 'N' && trans != 'T' && trans != 'C')
//         err_char("trans", "'N', 'T', 'C'");
//     if (m < 0) m = C->nrows;
//     if (n < 0) n = C->ncols;
//     if (k < 0) k = MIN(A->nrows, A->ncols);
//     if (m == 0 || n == 0 || k == 0) return Py_BuildValue("");
//     if (k > ((side == 'L') ? m : n)) err_ld("k");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,k)) err_ld("ldA");
//     if (ldC == 0) ldC = MAX(1,C->nrows);
//     if (ldC < MAX(1,m)) err_ld("ldC");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + ldA * ((side == 'L') ? m : n) > len(A)) err_buf_len("A");
//     if (oC < 0) err_nn_int("offsetC");
//     if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
//     if (len(tau) < k) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             if (trans == 'C') trans = 'T';
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dormlq_(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
//                 &ldC, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dormlq_(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
//                 &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         case COMPLEX:
//             if (trans == 'T') err_char("trans", "'N', 'C'");
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zunmlq_(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
//                 &ldC, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zunmlq_(&side, &trans, &m, &n, &k, MAT_BUFZ(A)+oA, &ldA,
//                 MAT_BUFZ(tau), MAT_BUFZ(C)+oC, &ldC, 
//                 (complex_t *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_orglq[] =
//     "Generate the orthogonal matrix in an LQ factorization.\n\n"
//     "orglq(A, tau, m=min(A.size), n=A.size[1], k=len(tau), \n"
//     "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
//     "PURPOSE\n"
//     "On entry, A and tau contain an n by n orthogonal matrix Q.\n"
//     "Q is defined as a product of k elementary reflectors, stored in the\n"
//     "first k rows of A and in tau, as computed by gelqf().  On exit,\n"
//     "the first m rows of Q are stored in the leading rows of A.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "tau       'd' matrix of length at least k\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  n >= m.  If negative, the default value is used."
//     "\n\n"
//     "k         integer.  k <= m.  If negative, the default value is \n"
//     "          used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* orglq(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau;
//     int m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
//     void *work;
//     number wl;
//     char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii", kwlist, &A,
//         &tau, &m, &n, &k, &ldA, &oA)) return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
//     if (m < 0) m = MIN(A->nrows, A->ncols);
//     if (n < 0) n = A->ncols;
//     if (m > n) err_ld("n");
//     if (k < 0) k = len(tau);
//     if (k > m) err_ld("k");
//     if (m == 0 || n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA <  MAX(1, m)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + n*ldA  > len(A)) err_buf_len("A");
//     if (len(tau) < k) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dorglq_(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dorglq_(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_unglq[] =
//     "Generate the orthogonal or unitary matrix in an LQ factorization.\n\n"
//     "unglq(A, tau, m=min(A.size), n=A.size[1], k=len(tau), \n"
//     "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
//     "PURPOSE\n"
//     "On entry, A and tau contain an n by n orthogonal/unitary matrix Q.\n"
//     "Q is defined as a product of k elementary reflectors, stored in the\n"
//     "first k rows of A and in tau, as computed by gelqf().  On exit,\n"
//     "the first m rows of Q are stored in the leading rows of A.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
//     "          same type as A.\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  n >= m.  If negative, the default value is used."
//     "\n\n"
//     "k         integer.  k <= m.  If negative, the default value is \n"
//     "          used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* unglq(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau;
//     int m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
//     void *work;
//     number wl;
//     char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii",
//         kwlist, &A, &tau, &m, &n, &k, &ldA, &oA)) return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
//     if (m < 0) m = MIN(A->nrows, A->ncols);
//     if (n < 0) n = A->ncols;
//     if (m > n) err_ld("n");
//     if (k < 0) k = len(tau);
//     if (k > m) err_ld("k");
//     if (m == 0 || n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA <  MAX(1, m)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + n*ldA  > len(A)) err_buf_len("A");
//     if (len(tau) < k) err_buf_len("tau");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dorglq_(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dorglq_(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zunglq_(&m, &n, &k, NULL, &ldA, NULL, &wl.z, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zunglq_(&m, &n, &k, MAT_BUFZ(A) + oA, &ldA, MAT_BUFZ(tau),
//                 (complex_t *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_geqp3[] =
//     "QR factorization with column pivoting.\n\n"
//     "geqp3(A, jpvt, tau, m=A.size[0], n=A.size[1], ldA=max(1,A.size[0]),\n"
//     "      offsetA=0)\n\n"
//     "PURPOSE\n"
//     "QR factorization with column pivoting of an m by n real or complex\n"
//     "matrix A:\n\n"
//     "A*P = Q*R = [Q1 Q2] * [R1; 0] if m >= n\n"
//     "A*P = Q*R = Q * [R1 R2] if m <= n,\n\n"
//     "where P is a permutation matrix, Q is m by m and orthogonal/unitary\n"
//     "and R is m by n with R1 upper triangular.  On exit, R is stored in\n"
//     "the upper triangular part of A.  Q is stored as a product of\n"
//     "k=min(m,n) elementary reflectors.  The parameters of the\n"
//     "reflectors are stored in the first k entries of tau and in the\n"
//     "lower triangular part of the first k columns of A.  On entry, if\n"
//     "jpvt[j] is nonzero, the jth column of A is permuted to the front of\n"
//     "A*P.  If jpvt[j] is zero, the jth column is a free column.  On exit\n"
//     "A*P = A[:, jpvt - 1].\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "jpvt      'i' matrix of length n\n\n"
//     "tau       'd' or 'z' matrix of length min(m,n).  Must have the same\n"
//     "          type as A.\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer";

// static PyObject* geqp3(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *tau, *jpvt;
//     int m=-1, n=-1, ldA=0, oA=0, info, lwork;
//     double *rwork = NULL;
//     void *work = NULL;
//     number wl;
//     char *kwlist[] = {"A", "jpvt", "tau", "m", "n", "ldA", "offsetA",
//         NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iiii", kwlist,
//         &A, &jpvt, &tau, &m, &n, &ldA, &oA))
//         return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(jpvt) || jpvt ->id != INT) err_int_mtrx("jpvt");
//     if (!Matrix_Check(tau)) err_mtrx("tau");
//     if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = A->ncols;
//     if (m == 0 || n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA < MAX(1,m)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
//     if (len(jpvt) < n) err_buf_len("jpvt");
//     if (len(tau) < MIN(m,n)) err_buf_len("tau");

//     int i;
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     int *jpvt_ptr = malloc(n*sizeof(int));
//     if (!jpvt_ptr) return PyErr_NoMemory();
//     for (i=0; i<n; i++) jpvt_ptr[i] = MAT_BUFI(jpvt)[i];
// #else
//     int *jpvt_ptr = MAT_BUFI(jpvt);
// #endif

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dgeqp3_(&m, &n, NULL, &ldA, NULL, NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = (void *) calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dgeqp3_(&m, &n, MAT_BUFD(A)+oA, &ldA, jpvt_ptr, MAT_BUFD(tau),
//                 (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
// 	    break;

//         case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zgeqp3_(&m, &n, NULL, &ldA, NULL, NULL, &wl.z, &lwork, NULL,
//                 &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             if (!(work = (void *) calloc(lwork, sizeof(complex_t))) ||
//                 !(rwork = (double *) calloc(2*n, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zgeqp3_(&m, &n, MAT_BUFZ(A)+oA, &ldA, jpvt_ptr, MAT_BUFZ(tau),
//                 (complex_t *) work, &lwork, rwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             free(rwork);
// 	    break;

//         default:
// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//             free(jpvt_ptr);
// #endif
//             err_invalid_id;
//     }

// #if (SIZEOF_INT < SIZEOF_SIZE_T)
//     for (i=0; i<n; i++) MAT_BUFI(jpvt)[i] = jpvt_ptr[i];
//     free(jpvt_ptr);
// #endif

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }



// static char doc_syev[] =
//     "Eigenvalue decomposition of a real symmetric matrix.\n\n"
//     "syev(A, W, jobz='N', uplo='L', n=A.size[0], "
//     "ldA = max(1,A.size[0]),\n"
//     "     offsetA=0, offsetW=0)\n\n"
//     "PURPOSE\n"
//     "Returns eigenvalues/vectors of a real symmetric nxn matrix A.\n"
//     "On exit, W contains the eigenvalues in ascending order.  If jobz\n"
//     "is 'V', the (normalized) eigenvectors are also computed and\n"
//     "returned in A.  If jobz is 'N', only the eigenvalues are\n"
//     "computed, and the content of A is destroyed.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "W         'd' matrix of length at least n\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* syev(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W;
//     int n=-1, ldA=0, oA=0, oW=0, info, lwork;
//     double *work;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N';
// #endif
//     char uplo = 'L', jobz = 'N';
//     char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
//         "offsetW", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCiiii", kwlist,
//         &A, &W, &jobz_, &uplo_, &n, &ldA, &oA, &oW)) 
//         return NULL;
//     jobz = (char) jobz_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cciiii", kwlist,
//         &A, &W, &jobz, &uplo, &n, &ldA, &oA, &oW)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("i",0);
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");

//     switch (MAT_ID(A)){
// 	case DOUBLE:
// 	    lwork=-1;
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyev_(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork,
//                 &info);
//             Py_END_ALLOW_THREADS
// 	    lwork = (int) wl.d;
// 	    if (!(work = calloc(lwork, sizeof(double))))
// 		return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyev_(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(W)+oW, work, &lwork, &info);
//             Py_END_ALLOW_THREADS
// 	    free(work);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_heev[] =
//     "Eigenvalue decomposition of a real symmetric or complex Hermitian"
//     "\nmatrix.\n\n"
//     "heev(A, W, jobz='N', uplo='L', n=A.size[0], "
//     "ldA = max(1,A.size[0]),\n"
//     "     offsetA=0, offsetW=0)\n\n"
//     "PURPOSE\n"
//     "Returns eigenvalues/vectors of a real symmetric or complex\n"
//     "Hermitian nxn matrix A.  On exit, W contains the eigenvalues in\n"
//     "ascending order.  If jobz is 'V', the (normalized) eigenvectors\n"
//     "are also computed and returned in A.  If jobz is 'N', only the\n"
//     "eigenvalues are computed, and the content of A is destroyed.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "W         'd' matrix of length at least n\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* heev(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W;
//     int n=-1, ldA=0, oA=0, oW=0, info, lwork;
//     double *rwork=NULL;
//     void *work=NULL;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N';
// #endif
//     char uplo = 'L', jobz = 'N';
//     char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
//         "offsetW", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCiiii", kwlist,
//         &A, &W, &jobz_, &uplo_, &n, &ldA, &oA, &oW)) 
//         return NULL;
//     jobz = (char) jobz_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cciiii", kwlist,
//         &A, &W, &jobz, &uplo, &n, &ldA, &oA, &oW)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");
//     switch (MAT_ID(A)){
// 	case DOUBLE:
// 	    lwork=-1;
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyev_(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork,
//                 &info);
//             Py_END_ALLOW_THREADS
// 	    lwork = (int) wl.d;
// 	    if (!(work = (void *) calloc(lwork, sizeof(double))))
// 		return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyev_(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(W)+oW, (double *) work, &lwork, &info);
//             Py_END_ALLOW_THREADS
// 	    free(work);
// 	    break;

//         case COMPLEX:
// 	    lwork=-1;
//             Py_BEGIN_ALLOW_THREADS
// 	    zheev_(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork,
//                 NULL, &info);
//             Py_END_ALLOW_THREADS
// 	    lwork = (int) creal(wl.z);
// 	    work = (void *) calloc(lwork, sizeof(complex_t));
// 	    rwork = (double *) calloc(3*n-2, sizeof(double));
// 	    if (!work || !rwork){
// 		free(work);  free(rwork);
// 		return PyErr_NoMemory();
// 	    }
//             Py_BEGIN_ALLOW_THREADS
// 	    zheev_(&jobz, &uplo, &n, MAT_BUFZ(A)+oA, &ldA,
// 		MAT_BUFD(W)+oW,  (complex_t *) work, &lwork, rwork,
// 		&info);
//             Py_END_ALLOW_THREADS
// 	    free(work);  free(rwork);
// 	    break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_syevx[] =
//     "Computes selected eigenvalues and eigenvectors of a real symmetric"
//     "\nmatrix (expert driver).\n\n"
//     "m = syevx(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
//     "          il=1, iu=1, Z=None, n=A.size[0], ldA=max(1,A.size[0]),\n"
//     "          ldZ=None, abstol=0.0, offsetA=0, offsetW=0,\n"
//     "          offsetZ=0)\n\n"
//     "PURPOSE\n"
//     "Computes selected eigenvalues/vectors of a real symmetric n by n\n"
//     "matrix A.\n"
//     "If range is 'A', all eigenvalues are computed.\n"
//     "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
//     "computed.\n"
//     "If range is 'I', all eigenvalues il through iu are computed\n"
//     "(sorted in ascending order with 1 <= il <= iu <= n).\n"
//     "If jobz is 'N', only the eigenvalues are returned in W.\n"
//     "If jobz is 'V', the eigenvectors are also returned in Z.\n"
//     "On exit, the content of A is destroyed.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "W         'd' matrix of length at least n.  On exit, contains\n"
//     "          the computed eigenvalues in ascending order.\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "range     'A', 'V' or 'I'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "vl,vu     doubles.  Only required when range is 'V'.\n\n"
//     "il,iu     integers.  Only required when range is 'I'.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "Z         'd' matrix.  Only required when jobz is 'V'.  If range\n"
//     "          is 'A' or 'V', Z must have at least n columns.  If\n"
//     "          range is 'I', Z must have at least iu-il+1 columns.\n"
//     "          On exit the first m columns of Z contain the computed\n"
//     "          (normalized) eigenvectors.\n\n"
//     "abstol    double.  Absolute error tolerance for eigenvalues.\n"
//     "          If nonpositive, the LAPACK default value is used.\n\n"
//     "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
//     "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
//     "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
//     "          If zero, the default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetW   nonnegative integer\n\n"
//     "offsetZ   nonnegative integer\n\n"
//     "m         the number of eigenvalues computed";

// static PyObject* syevx(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W, *Z=NULL;
//     int n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info, lwork,
//         *iwork, m, *ifail=NULL;
//     double *work, vl=0.0, vu=0.0, abstol=0.0;
//     double wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
// #endif
//     char uplo = 'L', jobz = 'N', range = 'A';
//     char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
// 	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
//         "offsetW", "offsetZ", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddiiOiiidiii",
//         kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
//     jobz = (char) jobz_;
//     range = (char) range_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddiiOiiidiii",
//         kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (range != 'A' && range != 'V' && range != 'I')
// 	err_char("range", "'A', 'V', 'I'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("i",0);
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (range == 'V' && vl >= vu){
//         PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
//         return NULL;
//     }
//     if (range == 'I' && (il < 1 || il > iu || iu > n)){
//         PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
//             "1 <= il <= iu <= n");
//         return NULL;
//     }
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");
//     if (jobz == 'V'){
//         if (!Z || !Matrix_Check(Z) || MAT_ID(Z) != DOUBLE)
//             err_dbl_mtrx("Z");
//         if (ldZ == 0) ldZ = MAX(1,Z->nrows);
//         if (ldZ < MAX(1,n)) err_ld("ldZ");
//         if (oZ < 0) err_nn_int("offsetZ");
//         if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
// 	    err_buf_len("Z");
//     } else {
//         if (ldZ == 0) ldZ = 1;
//         if (ldZ < 1) err_ld("ldZ");
//     }

//     switch (MAT_ID(A)){
//         case DOUBLE:
// 	    lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyevx_(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
//                 &iu, &abstol, &m, NULL, NULL, &ldZ, &wl, &lwork, NULL,
// 	       	NULL, &info);
//             Py_END_ALLOW_THREADS
// 	    lwork = (int) wl;
// 	    work = (double *) calloc(lwork, sizeof(double));
// 	    iwork = (int *) calloc(5*n, sizeof(int));
// 	    if (jobz == 'V') ifail = (int *) calloc(n, sizeof(int));
// 	    if (!work || !iwork || (jobz == 'V' && !ifail)){
// 		free(work); free(iwork); free(ifail);
// 	        return PyErr_NoMemory();
// 	    }
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyevx_(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
//                 &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
// 		(jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ, work,
// 		&lwork, iwork, ifail, &info);
//             Py_END_ALLOW_THREADS
// 	    free(work);   free(iwork);   free(ifail);
//             break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("i", m);
// }


// static char doc_heevx[] =
//     "Computes selected eigenvalues and eigenvectors of a real symmetric"
//     "\nor complex Hermitian matrix (expert driver).\n\n"
//     "m = syevx(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
//     "          il=1, iu=1, Z=None, n=A.size[0], \n"
//     "          ldA = max(1,A.size[0]), ldZ=None, abstol=0.0, \n"
//     "          offsetA=0, offsetW=0, offsetZ=0)\n\n"
//     "PURPOSE\n"
//     "Computes selected eigenvalues/vectors of a real symmetric or\n"
//     "complex Hermitian n by n matrix A.\n"
//     "If range is 'A', all eigenvalues are computed.\n"
//     "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
//     "computed.\n"
//     "If range is 'I', all eigenvalues il through iu are computed\n"
//     "(sorted in ascending order with 1 <= il <= iu <= n).\n"
//     "If jobz is 'N', only the eigenvalues are returned in W.\n"
//     "If jobz is 'V', the eigenvectors are also returned in Z.\n"
//     "On exit, the content of A is destroyed.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "W         'd' matrix of length at least n.  On exit, contains\n"
//     "          the computed eigenvalues in ascending order.\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "range     'A', 'V' or 'I'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "vl,vu     doubles.  Only required when range is 'V'.\n\n"
//     "il,iu     integers.  Only required when range is 'I'.\n\n"
//     "Z         'd' or 'z' matrix.  Must have the same type as A.\n"
//     "          Z is only required when jobz is 'V'.  If range is 'A'\n"
//     "          or 'V', Z must have at least n columns.  If range is\n"
//     "          'I', Z must have at least iu-il+1 columns.  On exit\n"
//     "          the first m columns of Z contain the computed\n"
//     "          (normalized) eigenvectors.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
//     "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
//     "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
//     "          If zero, the default value is used.\n\n"
//     "abstol    double.  Absolute error tolerance for eigenvalues.\n"
//     "          If nonpositive, the LAPACK default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetW   nonnegative integer\n\n"
//     "offsetZ   nonnegative integer\n\n"
//     "m         the number of eigenvalues computed";

// static PyObject* heevx(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W, *Z=NULL;
//     int n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info, lwork,
//         *iwork, m, *ifail=NULL;
//     double vl=0.0, vu=0.0, abstol=0.0, *rwork;
//     number wl;
//     void *work;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
// #endif
//     char uplo = 'L', jobz = 'N', range = 'A';
//     char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
// 	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
//         "offsetW", "offsetZ", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddiiOiiidiii",
//         kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
//     jobz = (char) jobz_;
//     range = (char) range_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddiiOiiidiii",
//         kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (range != 'A' && range != 'V' && range != 'I')
// 	err_char("range", "'A', 'V', 'I'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("i",0);
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (range == 'V' && vl >= vu){
//         PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
//         return NULL;
//     }
//     if (range == 'I' && (il < 1 || il > iu || iu > n)){
//         PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
//             "1 <= il <= iu <= n");
//         return NULL;
//     }
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");
//     if (jobz == 'V'){
//         if (!Z || !Matrix_Check(Z)) err_mtrx("Z");
// 	if (MAT_ID(Z) != MAT_ID(A)) err_conflicting_ids;
//         if (ldZ == 0) ldZ = MAX(1,Z->nrows);
//         if (ldZ < MAX(1,n)) err_ld("ldZ");
//         if (oZ < 0) err_nn_int("offsetZ");
//         if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
// 	    err_buf_len("Z");
//     } else {
//         if (ldZ == 0) ldZ = 1;
//         if (ldZ < 1) err_ld("ldZ");
//     }

//     switch (MAT_ID(A)){
//         case DOUBLE:
// 	    lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyevx_(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
//                 &iu, &abstol, &m, NULL, NULL, &ldZ, &wl.d, &lwork, NULL,
// 	       	NULL, &info);
//             Py_END_ALLOW_THREADS
// 	    lwork = (int) wl.d;
// 	    work = (void *) calloc(lwork, sizeof(double));
// 	    iwork = (int *) calloc(5*n, sizeof(int));
// 	    if (jobz == 'V') ifail = (int *) calloc(n, sizeof(int));
// 	    if (!work || !iwork || (jobz == 'V' && !ifail)){
// 		free(work); free(iwork); free(ifail);
// 	        return PyErr_NoMemory();
// 	    }
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyevx_(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
//                 &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
// 		(jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ,
// 		(double *) work, &lwork, iwork, ifail, &info);
//             Py_END_ALLOW_THREADS
// 	    free(work);  free(iwork);  free(ifail);
//             break;

// 	case COMPLEX:
// 	    lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
// 	    zheevx_(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
//                 &iu, &abstol, &m, NULL, NULL, &ldZ, &wl.z, &lwork, NULL,
// 	       	NULL, NULL, &info);
//             Py_END_ALLOW_THREADS
// 	    lwork = (int) creal(wl.z);
// 	    work = (void *) calloc(lwork, sizeof(complex_t));
// 	    rwork = (double *) calloc(7*n, sizeof(double));
// 	    iwork = (int *) calloc(5*n, sizeof(int));
// 	    if (jobz == 'V') ifail = (int *) calloc(n, sizeof(int));
// 	    if (!work || !rwork || !iwork || (jobz == 'V' && !ifail)){
// 		free(work); free(rwork); free(iwork); free(ifail);
// 	        return PyErr_NoMemory();
// 	    }
//             Py_BEGIN_ALLOW_THREADS
// 	    zheevx_(&jobz, &range, &uplo, &n, MAT_BUFZ(A)+oA, &ldA, &vl,
//                 &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
// 		(jobz == 'V') ? MAT_BUFZ(Z)+oZ : NULL,  &ldZ,
// 		(complex_t *) work, &lwork, rwork, iwork, ifail, 
//                 &info);
//             Py_END_ALLOW_THREADS
// 	    free(work);  free(rwork);  free(iwork);  free(ifail);
//             break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("i", m);
// }


// static char doc_syevd[] =
//     "Eigenvalue decomposition of a real symmetric matrix\n"
//     "(divide-and-conquer driver).\n\n"
//     "syevd(A, W, jobz='N', uplo='L', n=A.size[0], "
//     "ldA = max(1,A.size[0]),\n"
//     "      offsetA=0, offsetW=0)\n\n"
//     "PURPOSE\n"
//     "Returns  eigenvalues/vectors of a real symmetric nxn matrix A.\n"
//     "On exit, W contains the eigenvalues in ascending order.\n"
//     "If jobz is 'V', the (normalized) eigenvectors are also computed\n"
//     "and returned in A.  If jobz is 'N', only the eigenvalues are\n"
//     "computed, and the content of A is destroyed.\n"
//     "\n\nARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "W         'd' matrix of length at least n.  On exit, contains\n"
//     "          the computed eigenvalues in ascending order.\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* syevd(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W;
//     int n=-1, ldA=0, oA=0, oW=0, info, lwork, liwork, *iwork, iwl;
//     double *work=NULL, wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N';
// #endif
//     char uplo = 'L', jobz = 'N';
//     char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
//         "offsetW", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCiiii", kwlist,
//         &A, &W, &jobz_, &uplo_, &n, &ldA, &oA, &oW)) 
//         return NULL;
//     uplo = (char) uplo_;
//     jobz = (char) jobz_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cciiii", kwlist,
//         &A, &W, &jobz, &uplo, &n, &ldA, &oA, &oW)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || W->id != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             liwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsyevd_(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl, &lwork,
//                 &iwl, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl;
//             liwork = iwl;
//             work = (double *) calloc(lwork, sizeof(double));
//             iwork = (int *) calloc(liwork, sizeof(int));
//             if (!work || !iwork){
//                 free(work);  free(iwork);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
// 	    dsyevd_(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(W)+oW, work, &lwork, iwork, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);  free(iwork);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_heevd[] =
//     "Eigenvalue decomposition of a real symmetric or complex Hermitian"
//     "\nmatrix (divide-and-conquer driver).\n\n"
//     "heevd(A, W, jobz='N', uplo='L', n=A.size[0], "
//     "ldA = max(1,A.size[0]),\n"
//     "      offsetA=0, offsetW=0)\n\n"
//     "PURPOSE\n"
//     "Returns  eigenvalues/vectors of a real symmetric or complex\n"
//     "Hermitian n by n matrix A.  On exit, W contains the eigenvalues\n"
//     "in ascending order.  If jobz is 'V', the (normalized) eigenvectors"
//     "\nare also computed and returned in A.  If jobz is 'N', only the\n"
//     "eigenvalues are computed, and the content of A is destroyed.\n"
//     "\n\nARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "W         'd' matrix of length at least n.  On exit, contains\n"
//     "          the computed eigenvalues in ascending order.\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* heevd(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W;
//     int n=-1, ldA=0, oA=0, oW=0, info, lwork, liwork, *iwork, iwl,
// 	lrwork;
//     double *rwork, rwl;
//     number wl;
//     void *work;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N';
// #endif
//     char uplo = 'L', jobz = 'N';
//     char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
//         "offsetW", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCiiii", kwlist,
//         &A, &W, &jobz_, &uplo_, &n, &ldA, &oA, &oW)) 
//         return NULL;
//     jobz = (char) jobz_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cciiii", kwlist,
//         &A, &W, &jobz, &uplo, &n, &ldA, &oA, &oW)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || W->id != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             liwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsyevd_(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork,
//                 &iwl, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             liwork = iwl;
//             work = (void *) calloc(lwork, sizeof(double));
//             iwork = (int *) calloc(liwork, sizeof(int));
//             if (!work || !iwork){
//                 free(work);  free(iwork);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dsyevd_(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(W)+oW, (double *) work, &lwork, iwork, &liwork,
//                 &info);
//             Py_END_ALLOW_THREADS
//             free(work);   free(iwork);
//             break;

//         case COMPLEX:
//             lwork = -1;
//             liwork = -1;
//             lrwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zheevd_(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork,
//                 &rwl, &lrwork, &iwl, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             lrwork = (int) rwl;
//             liwork = iwl;
//             work = (void *) calloc(lwork, sizeof(complex_t));
//             rwork = (double *) calloc(lrwork, sizeof(double));
//             iwork = (int *) calloc(liwork, sizeof(int));
//             if (!work || !rwork || !iwork){
//                 free(work);  free(rwork);  free(iwork);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zheevd_(&jobz, &uplo, &n, MAT_BUFZ(A)+oA, &ldA,
//                 MAT_BUFD(W)+oW, (complex_t *) work, &lwork, rwork,
//                 &lrwork, iwork, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);  free(rwork);  free(iwork);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_syevr[] =
//     "Computes selected eigenvalues and eigenvectors of a real symmetric"
//     "\n"
//     "matrix (RRR driver).\n\n"
//     "m = syevr(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
//     "          il=1, iu=1, Z=None, n=A.size[0], ldA=max(1,A.size[0]),\n"
//     "          ldZ=None, abstol=0.0, offsetA=0, offsetW=0, offsetZ=0)\n"
//     "\n"
//     "PURPOSE\n"
//     "Computes selected eigenvalues/vectors of a real symmetric n by n\n"
//     "matrix A.\n"
//     "If range is 'A', all eigenvalues are computed.\n"
//     "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
//     "computed.\n"
//     "If range is 'I', all eigenvalues il through iu are computed\n"
//     "(sorted in ascending order with 1 <= il <= iu <= n).\n"
//     "If jobz is 'N', only the eigenvalues are returned in W.\n"
//     "If jobz is 'V', the eigenvectors are also returned in Z.\n"
//     "On exit, the content of A is destroyed.\n"
//     "syevr is usually the fastest of the four eigenvalue routines.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "W         'd' matrix of length at least n.  On exit, contains\n"
//     "          the computed eigenvalues in ascending order.\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "range     'A', 'V' or 'I'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "vl,vu     doubles.  Only required when range is 'V'.\n\n"
//     "il,iu     integers.  Only required when range is 'I'.\n\n"
//     "Z         'd' matrix.  Only required when jobz = 'V'.\n"
//     "          If range is 'A' or 'V', Z must have at least n columns."
//     "\n"
//     "          If range is 'I', Z must have at least iu-il+1 columns.\n"
//     "          On exit the first m columns of Z contain the computed\n"
//     "          (normalized) eigenvectors.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).\n"
//     "          If zero, the default value is used.\n\n"
//     "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
//     "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
//     "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
//     "          If zero, the default value is used.\n\n"
//     "abstol    double.  Absolute error tolerance for eigenvalues.\n"
//     "          If nonpositive, the LAPACK default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetW   nonnegative integer\n\n"
//     "offsetZ   nonnegative integer\n\n"
//     "m         the number of eigenvalues computed";

// static PyObject* syevr(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W, *Z=NULL;
//     int n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info, lwork,
//         *iwork=NULL, liwork, m, *isuppz=NULL, iwl;
//     double *work=NULL, vl=0.0, vu=0.0, abstol=0.0, wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
// #endif
//     char uplo = 'L', jobz = 'N', range = 'A';
//     char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
// 	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
//         "offsetW", "offsetZ", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddiiOiiidiii",
//         kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
//     jobz = (char) jobz_;
//     range = (char) range_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddiiOiiidiii",
//         kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (range != 'A' && range != 'V' && range != 'I')
// 	err_char("range", "'A', 'V', 'I'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("i",0);
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (range == 'V' && vl >= vu){
//         PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
//         return NULL;
//     }
//     if (range == 'I' && (il < 1 || il > iu || iu > n)){
//         PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
//             "1 <= il <= iu <= n");
//         return NULL;
//     }
//     if (jobz == 'V'){
//         if (!Z || !Matrix_Check(Z) || MAT_ID(Z) != DOUBLE)
//             err_dbl_mtrx("Z");
//         if (ldZ == 0) ldZ = MAX(1,Z->nrows);
//         if (ldZ < MAX(1,n)) err_ld("ldZ");
//     } else {
//         if (ldZ == 0) ldZ = 1;
//         if (ldZ < 1) err_ld("ldZ");
//     }
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");
//     if (jobz == 'V'){
//         if (oZ < 0) err_nn_int("offsetZ");
//         if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
// 	    err_buf_len("Z");
//     }

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             liwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsyevr_(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
//                 &iu, &abstol, &m, NULL, NULL, &ldZ, NULL, &wl, &lwork,
//                 &iwl, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl;
//             liwork = iwl;
//             work = (void *) calloc(lwork, sizeof(double));
//             iwork = (int *) calloc(liwork, sizeof(int));
//             if (jobz == 'V')
//                 isuppz = (int *) calloc(2*MAX(1, (range == 'I') ?
//                    iu-il+1 : n), sizeof(int));
//             if (!work  || !iwork || (jobz == 'V' && !isuppz)){
//                 free(work);  free(iwork);  free(isuppz);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dsyevr_(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
//                 &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
//                 (jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ,
//                 (jobz == 'V') ? isuppz : NULL, work, &lwork, iwork,
//                 &liwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);   free(iwork);   free(isuppz);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("i",m);
// }


// static char doc_heevr[] =
//     "Computes selected eigenvalues and eigenvectors of a real symmetric"
//     "\nor complex Hermitian matrix (RRR driver).\n\n"
//     "m = syevr(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
//     "          il=1, iu=1, Z=None, n=A.size[0], ldA=max(1,A.size[0]),\n"
//     "          ldZ=None, abstol=0.0, offsetA=0, offsetW=0, offsetZ=0)\n"
//     "\n"
//     "PURPOSE\n"
//     "Computes selected eigenvalues/vectors of a real symmetric or\n"
//     "complex Hermitian n by n matrix A.\n"
//     "If range is 'A', all eigenvalues are computed.\n"
//     "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
//     "computed.\n"
//     "If range is 'I', all eigenvalues il through iu are computed\n"
//     "(sorted in ascending order with 1 <= il <= iu <= n).\n"
//     "If jobz is 'N', only the eigenvalues are returned in W.\n"
//     "If jobz is 'V', the eigenvectors are also returned in Z.\n"
//     "On exit, the content of A is destroyed.\n"
//     "syevr is usually the fastest of the four eigenvalue routines.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "W         'd' matrix of length at least n.  On exit, contains\n"
//     "          the computed eigenvalues in ascending order.\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "range     'A', 'V' or 'I'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "vl,vu     doubles.  Only required when range is 'V'.\n\n"
//     "il,iu     integers.  Only required when range is 'I'.\n\n"
//     "Z         'd' or 'z' matrix.  Must have the same type as A.\n"
//     "          Only required when jobz = 'V'.  If range is 'A' or\n"
//     "          'V', Z must have at least n columns.  If range is 'I',\n"
//     "          Z must have at least iu-il+1 columns.  On exit the\n"
//     "          first m columns of Z contain the computed (normalized)\n"
//     "          eigenvectors.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).\n"
//     "          If zero, the default value is used.\n\n"
//     "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
//     "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
//     "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
//     "          If zero, the default value is used.\n\n"
//     "abstol    double.  Absolute error tolerance for eigenvalues.\n"
//     "          If nonpositive, the LAPACK default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetW   nonnegative integer\n\n"
//     "offsetZ   nonnegative integer\n\n"
//     "m         the number of eigenvalues computed";

// static PyObject* heevr(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *W, *Z=NULL;
//     int n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info,
//         lwork, *iwork, liwork, lrwork, m, *isuppz=NULL, iwl;
//     double vl=0.0, vu=0.0, abstol=0.0, *rwork, rwl;
//     void *work;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
// #endif
//     char uplo = 'L', jobz = 'N', range = 'A';
//     char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
// 	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
//         "offsetW", "offsetZ", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddiiOiiidiii",
//         kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
//     jobz = (char) jobz_;
//     range = (char) range_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddiiOiiidiii",
//         kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &il, &iu, &Z,
// 	&n, &ldA, &ldZ, &abstol, &oA, &oW, &oZ)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (range != 'A' && range != 'V' && range != 'I')
// 	err_char("range", "'A', 'V', 'I'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (n == 0) return Py_BuildValue("i",0);
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (range == 'V' && vl >= vu){
//         PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
//         return NULL;
//     }
//     if (range == 'I' && (il < 1 || il > iu || iu > n)){
//         PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
//             "1 <= il <= iu <= n");
//         return NULL;
//     }
//     if (jobz == 'V'){
//         if (!Z || !Matrix_Check(Z)) err_mtrx("Z");
// 	if (MAT_ID(Z) != MAT_ID(A)) err_conflicting_ids;
//         if (ldZ == 0) ldZ = MAX(1,Z->nrows);
//         if (ldZ < MAX(1,n)) err_ld("ldZ");
//     } else {
//         if (ldZ == 0) ldZ = 1;
//         if (ldZ < 1) err_ld("ldZ");
//     }
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");
//     if (jobz == 'V'){
//         if (oZ < 0) err_nn_int("offsetZ");
//         if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
// 	    err_buf_len("Z");
//     }

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             liwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dsyevr_(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
//                 &iu, &abstol, &m, NULL, NULL, &ldZ, NULL, &wl.d, &lwork,
//                 &iwl, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             liwork = iwl;
//             work = (void *) calloc(lwork, sizeof(double));
//             iwork = (int *) calloc(liwork, sizeof(int));
//             if (jobz == 'V')
//                 isuppz = (int *) calloc(2*MAX(1, (range == 'I') ?
//                    iu-il+1 : n), sizeof(int));
//             if (!work  || !iwork || (jobz == 'V' && !isuppz)){
//                 free(work);  free(iwork);  free(isuppz);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dsyevr_(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
//                 &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
//                 (jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ,
//                 (jobz == 'V') ? isuppz : NULL, (double *) work, &lwork,
// 	       	iwork, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);   free(iwork);   free(isuppz);
//             break;

// 	case COMPLEX:
//             lwork = -1;
//             liwork = -1;
// 	    lrwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zheevr_(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
//                 &iu, &abstol, &m, NULL, NULL, &ldZ, NULL, &wl.z, &lwork,
//                 &rwl, &lrwork, &iwl, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
// 	    lrwork = (int) rwl;
//             liwork = iwl;
//             work = (void *) calloc(lwork, sizeof(complex_t));
//             rwork = (double *) calloc(lrwork, sizeof(double));
//             iwork = (int *) calloc(liwork, sizeof(int));
//             if (jobz == 'V')
//                 isuppz = (int *) calloc(2*MAX(1, (range == 'I') ?
//                    iu-il+1 : n), sizeof(int));
//             if (!work  || !rwork || !iwork || (jobz == 'V' && !isuppz)){
//                 free(work);  free(rwork);  free(iwork);  free(isuppz);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zheevr_(&jobz, &range, &uplo, &n, MAT_BUFZ(A)+oA, &ldA, &vl,
//                 &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
//                 (jobz == 'V') ? MAT_BUFZ(Z)+oZ : NULL,  &ldZ,
//                 (jobz == 'V') ? isuppz : NULL, (complex_t *) work, 
//                 &lwork, rwork, &lrwork, iwork, &liwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);   free(rwork); free(iwork);   free(isuppz);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("i",m);
// }


// static char doc_sygv[] =
//     "Generalized symmetric-definite eigenvalue decomposition with real"
//     "\n"
//     "matrices.\n\n"
//     "sygv(A, B, W, itype=1, jobz='N', uplo='L', n=A.size[0], \n"
//     "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]), offsetA=0, \n"
//     "     offsetB=0, offsetW=0)\n\n"
//     "PURPOSE\n"
//     "Returns eigenvalues/vectors of a real generalized \n"
//     "symmetric-definite eigenproblem of order n, with B positive \n"
//     "definite. \n"
//     "1. If itype is 1: A*x = lambda*B*x.\n"
//     "2. If itype is 2: A*Bx = lambda*x.\n"
//     "3. If itype is 3: B*Ax = lambda*x.\n\n"
//     "On exit, W contains the eigenvalues in ascending order.  If jobz\n"
//     "is 'V', the matrix of eigenvectors Z is also computed and\n"
//     "returned in A, normalized as follows: \n"
//     "1. If itype is 1: Z^T*A*Z = diag(W), Z^T*B*Z = I\n"
//     "2. If itype is 2: Z^T*A^{-1}*Z = diag(W)^{-1}, Z^T*B*Z = I\n"
//     "3. If itype is 3: Z^T*A*Z = diag(W), Z^T*B^{-1}*Z = I.\n\n"
//     "If jobz is 'N', only the eigenvalues are computed, and the\n"
//     "contents of A is destroyed.   On exit, the matrix B is replaced\n"
//     "by its Cholesky factor.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' matrix\n\n"
//     "B         'd' matrix\n\n"
//     "W         'd' matrix of length at least n\n\n"
//     "itype     integer 1, 2, or 3\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* sygv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B, *W;
//     int n=-1, itype=1, ldA=0, ldB=0, oA=0, oB=0, oW=0, info, lwork;
//     double *work;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N';
// #endif
//     char uplo = 'L', jobz = 'N';
//     char *kwlist[] = {"A", "B", "W", "itype", "jobz", "uplo", "n",
//         "ldA", "ldB", "offsetA", "offsetB", "offsetW", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iCCiiiiii",
//         kwlist, &A, &B, &W, &itype, &jobz_, &uplo_, &n, &ldA, &ldB, &oA,
//         &oB, &oW)) 
//         return NULL;
//     jobz = (char) jobz_;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|icciiiiii",
//         kwlist, &A, &B, &W, &itype, &jobz, &uplo, &n, &ldA, &ldB, &oA,
//         &oB, &oW)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B) || MAT_ID(B) != MAT_ID(A)) err_conflicting_ids;
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (itype != 1 && itype != 2 && itype != 3)
//         err_char("itype", "1, 2, 3");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//         if (A->nrows != n || A->ncols != n){
//             PyErr_SetString(PyExc_TypeError, "B must have the same "
//                 "dimension as A");
//             return NULL;
// 	}
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (n-1)*ldB + n > len(B)) err_buf_len("B");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");

//     switch (MAT_ID(A)){
// 	case DOUBLE:
//             lwork=-1;
//             Py_BEGIN_ALLOW_THREADS
//             dsygv_(&itype, &jobz, &uplo, &n, NULL, &ldA, NULL, &ldB,
//                 NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dsygv_(&itype, &jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(B)+oB, &ldB, MAT_BUFD(W)+oW, work, &lwork,
//                 &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_hegv[] =
//     "Generalized symmetric-definite eigenvalue decomposition with\n"
//     "real or complex matrices.\n\n"
//     "hegv(A, B, W, itype=1, jobz='N', uplo='L', n=A.size[0], \n"
//     "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]), offsetA=0, \n"
//     "     offsetB=0, offsetW=0)\n\n"
//     "PURPOSE\n"
//     "Returns eigenvalues/vectors of a real or complex generalized\n"
//     "symmetric-definite eigenproblem of order n, with B positive\n"
//     "definite. \n"
//     "1. If itype is 1: A*x = lambda*B*x.\n"
//     "2. If itype is 2: A*Bx = lambda*x.\n"
//     "3. If itype is 3: B*Ax = lambda*x.n\n\n"
//     "On exit, W contains the eigenvalues in ascending order.  If jobz\n"
//     "is 'V', the matrix of eigenvectors Z is also computed and \n"
//     "returned in A, normalized as follows: \n"
//     "1. If itype is 1: Z^H*A*Z = diag(W), Z^H*B*Z = I\n"
//     "2. If itype is 2: Z^H*A^{-1}*Z = diag(W), Z^H*B*Z = I\n"
//     "3. If itype is 3: Z^H*A*Z = diag(W), Z^H*B^{-1}*Z = I.\n\n"
//     "If jobz is 'N', only the eigenvalues are computed, and the \n"
//     "contents of A is destroyed.   On exit, the matrix B is replaced\n"
//     "by its Cholesky factor.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "W         'd' matrix of length at least n\n\n"
//     "itype     integer 1, 2, or 3\n\n"
//     "jobz      'N' or 'V'\n\n"
//     "uplo      'L' or 'U'\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* hegv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B, *W;
//     int n=-1, itype=1, ldA=0, ldB=0, oA=0, oB=0, oW=0, info, lwork;
//     double *rwork=NULL;
//     void *work=NULL;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'L', jobz_ = 'N';
// #endif
//     char uplo = 'L', jobz = 'N';
//     char *kwlist[] = {"A", "B", "W", "itype", "jobz", "uplo", "n",
//         "ldA", "offsetA", "offsetB", "offsetW", NULL};
// #if 0
//     int ispec=1, n2=-1, n3=-1, n4=-1;
//     char *name = "zhetrd", *uplol = "L", *uplou = "U";
// #endif

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iCCiiiii",
//         kwlist, &A, &B, &W, &itype, &jobz_, &uplo_, &n, &ldA, &ldB, &oA,
//         &oB, &oW)) 
//         return NULL;
//     uplo = (char) uplo_;
//     jobz = (char) jobz_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|icciiiii",
//         kwlist, &A, &B, &W, &itype, &jobz, &uplo, &n, &ldA, &ldB, &oA,
//         &oB, &oW)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B) || MAT_ID(B) != MAT_ID(A)) err_conflicting_ids;
//     if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
//     if (itype != 1 && itype != 2 && itype != 3)
//         err_char("itype", "1, 2, 3");
//     if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
//     if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//         if (A->nrows != n || A->ncols != n){
//             PyErr_SetString(PyExc_TypeError, "B must have the same "
//                 "dimension as A");
//             return NULL;
// 	}
//     }
//     if (n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1,B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (n-1)*ldB + n > len(B)) err_buf_len("B");
//     if (oW < 0) err_nn_int("offsetW");
//     if (oW + n > len(W)) err_buf_len("W");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork=-1;
//             Py_BEGIN_ALLOW_THREADS
//             dsygv_(&itype, &jobz, &uplo, &n, NULL, &ldA, NULL, &ldB,
//                 NULL, &wl.d, &lwork, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             if (!(work = calloc(lwork, sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dsygv_(&itype, &jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
//                 MAT_BUFD(B)+oB, &ldB, MAT_BUFD(W)+oW, work, &lwork,
//                 &info);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         case COMPLEX:
// #if 1
//             lwork=-1;
//             Py_BEGIN_ALLOW_THREADS
//             zhegv_(&itype, &jobz, &uplo, &n, NULL, &n, NULL, &n, NULL,
//                 &wl.z, &lwork, NULL, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
// #endif
// #if 0
//             /* zhegv used to handle lwork=-1 incorrectly.
//                The following replaces the call to zhegv with lwork=-1 */
//             lwork = n * (1 + ilaenv_(&ispec, &name, (uplo == 'L') ?
//                 &uplol : &uplou, &n, &n2, &n3, &n4));
// #endif

//             work = (void *) calloc(lwork, sizeof(complex_t));
//             rwork = (double *) calloc(3*n-2, sizeof(double));
//             if (!work || !rwork){
//                 free(work);  free(rwork);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zhegv_(&itype, &jobz, &uplo, &n, MAT_BUFZ(A)+oA, &ldA,
//                 MAT_BUFZ(B)+oB, &ldB, MAT_BUFD(W)+oW, 
//                 (complex_t *) work, &lwork, rwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);  free(rwork);
//             break;

//         default:
// 	    err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }

/**
 * @brief Singular Value Decomposition of a real or complex matrix
 * 
 * lapack_gesvd(A, S, jobu='N', jobvt='N', U=None, Vt=None, m=A.size[0],
                n=A.size[1], ldA=max(1,A.size[0]), ldU=None, ldVt=None,
                offsetA=0, offsetS=0, offsetU=0, offsetVt=0)
 * 
 * @details
 * Computes the SVD of an m×n matrix A:
 * - A = U·Σ·Vᴴ (complex)
 * - A = U·Σ·Vᵀ (real)
 * 
 * Where:
 * - U contains left singular vectors (columns)
 * - Σ contains singular values (diagonal matrix, returned as vector)
 * - Vᴴ/Vᵀ contains right singular vectors (rows of Vt)
 *
 * @param[in,out] A    Input matrix to decompose ('d' or 'z' type)
 * @param[out] S       Output vector of singular values ('d' type)
 * @param[in] jobu     Controls left singular vector computation: (default = 'N')
 *                     - 'N': none
 *                     - 'A': all (m vectors)
 *                     - 'S': first min(m,n) vectors
 *                     - 'O': first min(m,n) vectors (overwrite A)
 * @param[in] jobvt    Controls right singular vector computation: (default = 'N')
 *                     - 'N': none
 *                     - 'A': all (n vectors)
 *                     - 'S': first min(m,n) vectors
 *                     - 'O': first min(m,n) vectors (overwrite A)
 * @param[out] U       Output matrix for left singular vectors  (default = 'N')
 *                     (same type as A, not referenced if jobu='N'/'O')
 * @param[out] Vt      Output matrix for right singular vectors
 *                     (same type as A, not referenced if jobvt='N'/'O')
 * @param[in] m        Rows of A (default: A.size[0])
 * @param[in] n        Columns of A (default: A.size[1])
 * @param[in] ldA      Leading dimension of A (≥ max(1,m))
 * @param[in] ldU      Leading dimension of U:
 *                     - ≥ max(1,m) if jobu='A'/'S'
 *                     - ≥ 1 otherwise
 * @param[in] ldVt     Leading dimension of Vt:
 *                     - ≥ max(1,n) if jobvt='A'
 *                     - ≥ max(1,min(m,n)) if jobvt='S'
 *                     - ≥ 1 otherwise
 * @param[in] offsetA  Matrix A offset (nonnegative)
 * @param[in] offsetS  Vector S offset (nonnegative)
 * @param[in] offsetU  Matrix U offset (nonnegative)
 * @param[in] offsetVt Matrix Vt offset (nonnegative)
 *
 * @see LAPACK GESVD documentation
 */
void lapack_gesvd(matrix *A, matrix *S, char jobu, char jobvt, matrix *U, matrix *Vt, int m, int n, 
                int ldA, int ldU, int ldVt, int offsetA, int offsetS, int offsetU, int offsetVt)
{
    // int m=-1, n=-1, ldA=0, ldU=0, ldVt=0, oA=0, oS=0, oU=0, oVt=0, info, lwork;
    double *rwork=NULL;
    void *work=NULL;
    number wl;

    int oA = offsetA, oS = offsetS, oU = offsetU, oVt = offsetVt, info, lwork;

    // Default values
    if (jobu == 0) jobu = 'N';
    if (jobvt == 0) jobvt = 'N';
    if (ldA == 0) ldA = 0;
    if (ldU == 0) ldU = 0;
    if (ldVt == 0) ldVt = 0;
    if (oA < 0) oA = 0;
    if (oS < 0) oS = 0;
    if (oU < 0) oU = 0;
    if (oVt < 0) oVt = 0;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(S) || MAT_ID(S) != DOUBLE) err_dbl_mtrx("S");
    if (jobu != 'N' && jobu != 'A' && jobu != 'O' && jobu != 'S')
        err_char("jobu", "'N', 'A', 'S', 'O'");
    if (jobvt != 'N' && jobvt != 'A' && jobvt != 'O' && jobvt != 'S')
        err_char("jobvt", "'N', 'A', 'S', 'O'");
    if (jobu == 'O' && jobvt == 'O') {
        ERR("'jobu' and 'jobvt' cannot both be 'O'");
        return;
    }
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return;
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (jobu == 'A' || jobu == 'S'){
        if (!U || !Matrix_Check(U)) err_mtrx("U");
        if (MAT_ID(U) != MAT_ID(A)) err_conflicting_ids;
        if (ldU == 0) ldU = MAX(1,U->nrows);
        if (ldU < MAX(1,m)) err_ld("ldU");
    } else {
        if (ldU == 0) ldU = 1;
        if (ldU < 1) err_ld("ldU");
    }
    if (jobvt == 'A' || jobvt == 'S'){
        if (!Vt || !Matrix_Check(Vt)) err_mtrx("Vt");
	if (MAT_ID(Vt) != MAT_ID(A)) err_conflicting_ids;
        if (ldVt == 0) ldVt = MAX(1,Vt->nrows);
        if (ldVt < ((jobvt == 'A') ?  MAX(1,n) : MAX(1,MIN(m,n))))
            err_ld("ldVt");
    } else {
        if (ldVt == 0) ldVt = 1;
        if (ldVt < 1) err_ld("ldVt");
    }
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (oS < 0) err_nn_int("offsetS");
    if (oS + MIN(m,n) > len(S)) err_buf_len("S");
    if (jobu == 'A' || jobu == 'S'){
        if (oU < 0) err_nn_int("offsetU");
        if (oU + ((jobu == 'A') ? m-1 : MIN(m,n)-1)*ldU + m > len(U))
            err_buf_len("U");
    }
    if (jobvt == 'A' || jobvt == 'S'){
        if (oVt < 0) err_nn_int("offsetVt");
        if (oVt + (n-1)*ldVt + ((jobvt == 'A') ? n : MIN(m,n)) >
            len(Vt)) err_buf_len("Vt");
    }

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            dgesvd_(&jobu, &jobvt, &m, &n, NULL, &ldA, NULL, NULL,
                &ldU, NULL, &ldVt, &wl.d, &lwork, &info);
            lwork = (int) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double)))){
                free(work);
                err_no_memory;
            }
            dgesvd_(&jobu, &jobvt, &m, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(S)+oS,  (jobu == 'A' || jobu == 'S') ?
		        MAT_BUFD(U)+oU : NULL, &ldU, (jobvt == 'A' ||
                jobvt == 'S') ?  MAT_BUFD(Vt)+oVt : NULL, &ldVt,
		        (double *) work, &lwork, &info);
            free(work);
            break;

	    case COMPLEX:
            lwork = -1;
            zgesvd_(&jobu, &jobvt, &m, &n, NULL, &ldA, NULL, NULL,
                &ldU, NULL, &ldVt, &wl.z, &lwork, NULL, &info);
            lwork = (int) creal(wl.z);
            work = (void *) calloc(lwork, sizeof(complex_t));
            rwork = (double *) calloc(5*MIN(m,n), sizeof(double));
	    if (!work || !rwork){
                free(work);  free(rwork);
                err_no_memory;
            }
            zgesvd_(&jobu, &jobvt, &m, &n, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFD(S)+oS, (jobu == 'A' || jobu == 'S') ?
                MAT_BUFZ(U)+oU : NULL,  &ldU, (jobvt == 'A' ||
                jobvt == 'S') ? MAT_BUFZ(Vt)+oVt : NULL, &ldVt,
                (complex_t *) work, &lwork, rwork, &info);
            free(work);   free(rwork);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack(info);
    else return;
}


// static char doc_gesdd[] =
//     "Singular value decomposition of a real or complex matrix\n"
//     "(divide-and-conquer driver).\n\n"
//     "gesdd(A, S, jobz='N', U=None, V=None, m=A.size[0], n=A.size[1], \n"
//     "      ldA=max(1,A.size[0]), ldU=None, ldVt=None, offsetA=0, \n"
//     "      offsetS=0, offsetU=0, offsetVt=0)\n\n"
//     "PURPOSE\n"
//     "Computes singular values and, optionally, singular vectors of a \n"
//     "real or complex m by n matrix A.  The argument jobz controls how\n"
//     "many singular vectors are computed:\n\n"
//     "'N': no singular vectors are computed.\n"
//     "'A': all m left singular vectors are computed and returned as\n"
//     "     columns of U;  all n right singular vectors are computed \n"
//     "     and returned as rows of Vt.\n"
//     "'S': the first min(m,n) left and right singular vectors are\n"
//     "     computed and returned as columns of U and rows of Vt.\n"
//     "'O': if m>=n, the first n left singular vectors are returned as\n"
//     "     columns of A and the n right singular vectors are returned\n"
//     "     as rows of Vt.  If m<n, the m left singular vectors are\n"
//     "     returned as columns of U and the first m right singular\n"
//     "     vectors are returned as rows of A.\n\n"
//     "Note that the (conjugate) transposes of the right singular \n"
//     "vectors are returned in Vt or A.\n\n"
//     "On exit (in all cases), the contents of A are destroyed.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "S         'd' matrix of length at least min(m,n).  On exit, \n"
//     "          contains the computed singular values in descending\n"
//     "          order.\n\n"
//     "jobz      'N', 'A', 'S' or 'O'\n\n"
//     "U         'd' or 'z' matrix.  Must have the same type as A.\n"
//     "          Not referenced if jobz is 'N' or jobz is 'O' and m>=n.\n"
//     "          If jobz is 'A' or jobz is 'O' and m<n, a matrix with\n"
//     "          at least m columns.   If jobz is 'S', a matrix with at\n"
//     "          least min(m,n) columns.  On exit (except when jobz is\n"
//     "          'N' or jobz is 'O' and m>=n), contains the computed\n"
//     "          left singular vectors stored columnwise.\n\n"
//     "Vt        'd' or 'z' matrix.  Must have the same type as A.\n"
//     "          Not referenced if jobz is 'N' or jobz is 'O' and m<n.\n"
//     "          If jobz is 'A' or 'S' or jobz is 'O' and m>=n, a\n"
//     "          matrix with at least n columns.   On exit (except when\n"
//     "          jobz is 'N' or jobz is 'O' and m<n), the rows of Vt\n"
//     "          contain the computed right singular vectors, or, in\n"
//     "          the complex case, their complex conjugates.\n\n"
//     "m         integer.  If negative, the default value is used.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,m).\n"
//     "          If zero, the default value is used.\n\n"
//     "ldU       nonnegative integer.\n"
//     "          ldU >= 1 if jobz is 'N' or 'O'.\n"
//     "          ldU >= max(1,m) if jobz is 'S' or 'A' or jobz is 'O'\n"
//     "          and m<n.  The default value is max(1,U.size[0]) if\n"
//     "          jobz is 'S' or 'A' or jobz is'O' and m<n, and 1\n"
//     "          otherwise.\n\n"
//     "ldVt      nonnegative integer.\n"
//     "          ldVt >= 1 if jobz is 'N'.\n"
//     "          ldVt >= max(1,n) if jobz is 'A' or jobz is 'O' and \n"
//     "          m>=n.  \n"
//     "          ldVt >= max(1,min(m,n)) if ldVt is 'S'.\n"
//     "          The default value is max(1,Vt.size[0]) if jobvt is 'A'\n"
//     "          or 'S' or jobvt is 'O' and m>=n, and 1 otherwise.\n"
//     "          If zero, the default value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetS   nonnegative integer\n\n"
//     "offsetU   nonnegative integer\n\n"
//     "offsetVt  nonnegative integer";

// static PyObject* gesdd(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *S, *U=NULL, *Vt=NULL;
//     int m=-1, n=-1, ldA=0, ldU=0, ldVt=0, oA=0, oS=0, oU=0, oVt=0, info,
//        	*iwork=NULL, lwork;
//     double *rwork=NULL;
//     void *work=NULL;
//     number wl;
// #if PY_MAJOR_VERSION >= 3
//     int jobz_ = 'N';
// #endif
//     char jobz = 'N';
//     char *kwlist[] = {"A", "S", "jobz", "U", "Vt", "m", "n", "ldA",
// 	"ldU", "ldVt", "offsetA", "offsetS", "offsetU", "offsetVt",
//         NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|COOiiiiiiiii",
//         kwlist, &A, &S, &jobz_, &U, &Vt, &m, &n, &ldA, &ldU, &ldVt, &oA,
//        	&oS, &oU, &oVt)) 
//         return NULL;
//     jobz = (char) jobz_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cOOiiiiiiiii",
//         kwlist, &A, &S, &jobz, &U, &Vt, &m, &n, &ldA, &ldU, &ldVt, &oA,
//        	&oS, &oU, &oVt)) 
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(S) || MAT_ID(S) != DOUBLE) err_dbl_mtrx("S");
//     if (jobz != 'A' && jobz != 'S' && jobz != 'O' && jobz != 'N')
//         err_char("jobz", "'A', 'S', 'O', 'N'");
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = A->ncols;
//     if (m == 0 || n == 0) return Py_BuildValue("");
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,m)) err_ld("ldA");
//     if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)){
//         if (!U || !Matrix_Check(U)) err_mtrx("U");
//         if (MAT_ID(U) != MAT_ID(A)) err_conflicting_ids;
//         if (ldU == 0) ldU = MAX(1,U->nrows);
//         if (ldU < MAX(1,m)) err_ld("ldU");
//     } else {
//         if (ldU == 0) ldU = 1;
//         if (ldU < 1) err_ld("ldU");
//     }
//     if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m>=n)){
//         if (!Vt || !Matrix_Check(Vt)) err_mtrx("Vt");
// 	if (MAT_ID(Vt) != MAT_ID(A)) err_conflicting_ids;
//         if (ldVt == 0) ldVt = MAX(1,Vt->nrows);
//         if (ldVt < ((jobz == 'A' || jobz == 'O') ?  MAX(1,n) :
//             MAX(1,MIN(m,n)))) err_ld("ldVt");
//     } else {
//         if (ldVt == 0) ldVt = 1;
//         if (ldVt < 1) err_ld("ldVt");
//     }
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
//     if (oS < 0) err_nn_int("offsetS");
//     if (oS + MIN(m,n) > len(S)) err_buf_len("S");
//     if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)){
//         if (oU < 0) err_nn_int("offsetU");
//         if (oU + ((jobz == 'A' || jobz == 'O') ? m-1 : MIN(m,n)-1)*ldU
//             + m > len(U))
// 	    err_buf_len("U");
//     }
//     if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m>=n)){
//         if (oVt < 0) err_nn_int("offsetVt");
//         if (oVt + (n-1)*ldVt + ((jobz == 'A' || jobz == 'O') ? n :
//             MIN(m,n)) > len(Vt)) err_buf_len("Vt");
//     }

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             dgesdd_(&jobz, &m, &n, NULL, &ldA, NULL, NULL, &ldU, NULL,
//                 &ldVt, &wl.d, &lwork, NULL, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) wl.d;
//             work = (void *) calloc(lwork, sizeof(double));
//             iwork = (int *) calloc(8*MIN(m,n), sizeof(int));
// 	    if (!work || !iwork){
//                 free(work);  free(iwork);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             dgesdd_(&jobz, &m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(S)+oS,
//                 (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)) ?
//                 MAT_BUFD(U)+oU : NULL, &ldU, (jobz == 'A' ||
//                 jobz == 'S' || (jobz == 'O'  && m>=n)) ?
//                 MAT_BUFD(Vt)+oVt : NULL, &ldVt, (double *) work, &lwork,
//                 iwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);   free(iwork);
//             break;

// 	case COMPLEX:
//             lwork = -1;
//             Py_BEGIN_ALLOW_THREADS
//             zgesdd_(&jobz, &m, &n, NULL, &ldA, NULL, NULL, &ldU, NULL,
//                 &ldVt, &wl.z, &lwork, NULL, NULL, &info);
//             Py_END_ALLOW_THREADS
//             lwork = (int) creal(wl.z);
//             work = (void *) calloc(lwork, sizeof(complex_t));
//             iwork = (int *) calloc(8*MIN(m,n), sizeof(int));
//             rwork = (double *) calloc( (jobz == 'N') ? 7*MIN(m,n) :
//                 5*MIN(m,n)*(MIN(m,n)+1), sizeof(double));
// 	    if (!work || !iwork || !rwork){
//                 free(work);  free(iwork); free(rwork);
//                 return PyErr_NoMemory();
//             }
//             Py_BEGIN_ALLOW_THREADS
//             zgesdd_(&jobz, &m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFD(S)+oS,
//                 (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)) ?
// 		MAT_BUFZ(U)+oU : NULL,  &ldU, (jobz == 'A' ||
//                 jobz == 'S' || (jobz == 'O' && m>=n)) ?
//                 MAT_BUFZ(Vt)+oVt : NULL, &ldVt, (complex_t *) work,
//                 &lwork, rwork, iwork, &info);
//             Py_END_ALLOW_THREADS
//             free(work);  free(iwork); free(rwork);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (info) err_lapack
//     else return Py_BuildValue("");
// }


// static char doc_gees[] =
//     "Schur factorization of a real of complex matrix.\n\n"
//     "sdim = gees(A, w=None, V=None, select=None, n=A.size[0],\n"
//     "            ldA=max(1,A.size[0]), ldV=max(1,Vs.size[0]), offsetA=0,\n"
//     "            offsetw=0, offsetV=0)\n\n"
//     "PURPOSE\n"
//     "Computes the real Schur form A = V * S * V^T or the complex Schur\n"
//     "form A = V * S * V^H, the eigenvalues, and, optionally, the matrix\n"
//     "of Schur vectors of an n by n matrix A.  The real Schur form is \n"
//     "computed if A is real, and the conmplex Schur form is computed if \n"
//     "A is complex.  On exit, A is replaced with S.  If the argument w is\n"
//     "provided, the eigenvalues are returned in w.  If V is provided, the\n"
//     "Schur vectors are computed and returned in V.  The argument select\n"
//     "can be used to obtain an ordered Schur factorization.  It must be a\n"
//     "Python function that can be called as f(s) with s complex, and \n"
//     "returns 0 or 1.  The eigenvalues s for which f(s) is 1 will be \n"
//     "placed first in the Schur factorization.   For the real case, \n"
//     "eigenvalues s for which f(s) or f(conj(s)) is 1, are placed first.\n"
//     "If select is provided, gees() returns the number of eigenvalues \n"
//     "that satisfy the selection criterion.   Otherwise, it returns 0.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "w         'z' matrix of length at least n\n\n"
//     "V         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "select    Python function that takes a complex number as argument\n"
//     "          and returns True or False.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).\n"
//     "          If zero, the default value is used.\n\n"
//     "ldV       nonnegative integer.  ldV >= 1 and ldV >= n if V is\n"
//     "          present.  If zero, the default value is used (with \n"
//     "          V.size[0] replaced by 0 if V is None).\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetW   nonnegative integer\n\n"
//     "offsetV   nonnegative integer\n\n"
//     "sdim      number of eigenvalues that satisfy the selection\n"
//     "          criterion specified by select.";

// static PyObject *py_select_r;
// static PyObject *py_select_c;

// extern int fselect_c(complex_t *w)
// {
//     PyObject *wpy, *result;
//     int a = 0;

//     wpy = PyComplex_FromDoubles(creal(*w), cimag(*w));
//     if (!(result = PyObject_CallFunctionObjArgs(py_select_c, wpy, NULL))) {
//         Py_XDECREF(wpy);
//         return -1;
//     }
// #if PY_MAJOR_VERSION >= 3
//     if (PyLong_Check(result)) a = (int) PyLong_AsLong(result);
// #else
//     if PyInt_Check(result) a = (int) PyInt_AsLong(result);
// #endif
//     else
//         PyErr_SetString(PyExc_TypeError, "select() must return an integer "
//             "argument");
//     Py_XDECREF(wpy);  Py_XDECREF(result);
//     return a;
// }

// extern int fselect_r(double *wr, double *wi)
// {
//     PyObject *wpy, *result;
//     int a = 0;

//     wpy = PyComplex_FromDoubles(*wr, *wi);
//     if (!(result = PyObject_CallFunctionObjArgs(py_select_r, wpy, NULL))) {
//         Py_XDECREF(wpy);
//         return -1;
//     }
// #if PY_MAJOR_VERSION >= 3
//     if (PyLong_Check(result)) a = (int) PyLong_AsLong(result);
// #else
//     if PyInt_Check(result) a = (int) PyInt_AsLong(result);
// #endif
//     else
//         PyErr_SetString(PyExc_TypeError, "select() must return an integer "
//            "argument");
//     Py_XDECREF(wpy);  Py_XDECREF(result);
//     return a;
// }

// static PyObject* gees(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     PyObject *F=NULL;
//     matrix *A, *W=NULL, *Vs=NULL;
//     int n=-1, ldA=0, ldVs=0, oA=0, oVs=0, oW=0, info, lwork, sdim, k,
//         *bwork=NULL;
//     double *wr=NULL, *wi=NULL, *rwork=NULL;
//     complex_t *w=NULL;
//     void *work=NULL;
//     number wl;
//     char *kwlist[] = {"A", "w", "V", "select", "n", "ldA", "ldV",
//         "offsetA", "offsetw", "offsetV", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OOOiiiiii",
//         kwlist, &A, &W, &Vs, &F, &n, &ldA, &ldVs, &oA, &oW, &oVs))
//         return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//     }
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

//     if (W){
//         if (!Matrix_Check(W) || MAT_ID(W) != COMPLEX)
//             PY_ERR_TYPE("W must be a matrix with typecode 'z'")
//         if (oW < 0) err_nn_int("offsetW");
//         if (oW + n > len(W)) err_buf_len("W");
//     }

//     if (Vs){
//         if (!Matrix_Check(Vs)) err_mtrx("Vs");
//         if (MAT_ID(Vs) != MAT_ID(A)) err_conflicting_ids;
//         if (ldVs == 0) ldVs = MAX(1, Vs->nrows);
//         if (ldVs < MAX(1,n)) err_ld("ldVs");
//         if (oVs < 0) err_nn_int("offsetVs");
//         if (oVs + (n-1)*ldVs + n > len(Vs)) err_buf_len("Vs");
//     } else {
//         if (ldVs == 0) ldVs = 1;
//         if (ldVs < 1) err_ld("ldVs");
//     }

//     if (F && !PyFunction_Check(F))
//         PY_ERR_TYPE("select must be a Python function")


//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             dgees_(Vs ? "V" : "N", F ? "S" : "N", NULL, &n, NULL, &ldA,
//                 &sdim, NULL, NULL, NULL, &ldVs, &wl.d, &lwork, NULL,
//                 &info);
//             lwork = (int) wl.d;
//             work = (void *) calloc(lwork, sizeof(double));
//             wr = (double *) calloc(n, sizeof(double));
//             wi = (double *) calloc(n, sizeof(double));
//             if (F) bwork = (int *) calloc(n, sizeof(int));
//             if (!work || !wr || !wi || (F && !bwork)){
//                 free(work);  free(wr);  free(wi);  free(bwork);
//                 return PyErr_NoMemory();
//             }
//             py_select_r = F;
//             dgees_(Vs ? "V": "N", F ? "S" : "N", F ? &fselect_r : NULL,
//                 &n, MAT_BUFD(A) + oA, &ldA, &sdim, wr, wi,
//                 Vs ? MAT_BUFD(Vs) + oVs : NULL, &ldVs, (double *) work,
//                 &lwork, bwork, &info);
//             if (W) for (k=0; k<n; k++)
// #ifndef _MSC_VER
//                 MAT_BUFZ(W)[oW + k] = wr[k] + I * wi[k];
// #else
//   	        MAT_BUFZ(W)[oW + k] = _Cbuild(wr[k],wi[k]);
// #endif
//             free(work);  free(wr);  free(wi);  free(bwork);
//             break;

// 	case COMPLEX:
//             lwork = -1;
//             zgees_(Vs ? "V" : "N", F ? "S": "N", NULL, &n, NULL, &ldA,
//                 &sdim, NULL, NULL, &ldVs, &wl.z, &lwork, NULL, NULL,
//                 &info);
//             lwork = (int) creal(wl.z);
//             work = (void *) calloc(lwork, sizeof(complex_t));
//             rwork = (double *) calloc(n, sizeof(complex_t));
//             if (F) bwork = (int *) calloc(n, sizeof(int));
//             if (!W) 
//                 w = (complex_t *) calloc(n, sizeof(complex_t));
// 	    if (!work || !rwork || (F && !bwork) || (!W && !w) ){
//                 free(work);  free(rwork); free(bwork);  free(w);
//                 return PyErr_NoMemory();
//             }
//             py_select_c = F;
//             zgees_(Vs ? "V": "N", F ? "S" : "N", F ? &fselect_c : NULL,
//                 &n, MAT_BUFZ(A) + oA, &ldA, &sdim,
//                 W ? MAT_BUFZ(W) + oW : w, Vs ? MAT_BUFZ(Vs) + oVs : NULL,
//                 &ldVs, (complex_t *) work, &lwork, 
//                 (complex_t *) rwork,  bwork, &info);
//             free(work);  free(rwork); free(bwork);  free(w);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (PyErr_Occurred()) return NULL;

//     if (info) err_lapack
//     else return Py_BuildValue("i", F ? sdim : 0);
// }


// static char doc_gges[] =
//     "Generalized Schur factorization of real or complex matrices.\n\n"
//     "sdim = gges(A, B, a=None, b=None, Vl=None, Vr=None, select=None,\n"
//     "            n=A.size[0], ldA=max(1,A.size[0]),\n"
//     "            ldB=max(1,B.size[0]), ldVl=max(1,Vl.size[0]),\n"
//     "            ldVr=max(1,Vr.size[0]), offsetA=0, offestB=0, \n"
//     "            offseta=0, offsetb=0, offsetVl=0, offsetVr=0)\n\n"
//     "PURPOSE\n"
//     "Computes the real generalized Schur form A = Vl * S * Vr^T, \n"
//     "B = Vl * T * Vr^T, or the complex generalized Schur form \n"
//     "A = Vl * S * Vr^H, B = Vl * T * Vr^H of the square matrices A, B,\n"
//     "the generalized eigenvalues, and, optionally, the matrices of left \n"
//     "and right Schur vectors.  The real form is computed if A and B are \n"
//     "real, and the complex Schur form is computed if A and B are\n"
//     "complex.  On exit, A is replaced with S and B is replaced with T.\n"
//     "If the arguments a and b are provided, then on return a[i] / b[i] \n"
//     "is the ith generalized eigenvalue.  If Vl is provided, then the \n"
//     "left Schur vectors are computed and returned in Vl.  If Vr is \n"
//     "provided then the right Schur vectors are computed and returned \n"
//     "in Vr.  The argument select can be used to obtain an ordered Schur\n"
//     "factorization.  It must be a Python function that can be called as\n"
//     "f(u,v) with u complex, and v real, and returns 0 or 1.  The \n"
//     "eigenvalues u/v for which f(u, v) is 1 will be placed first on the\n"
//     "diagonal of S and T.  For the real case, eigenvalues u/v for which\n"
//     "f(u, v) or f(conj(u), v) is 1, are placed first.  If select is \n"
//     "provided, gges() returns the number of eigenvalues that satisfy the\n"
//     "selection criterion.  Otherwise, it returns 0.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "a         'z' matrix of length at least n\n\n"
//     "b         'd' matrix of length at least n\n\n"
//     "Vl        'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "Vr        'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "select    Python function that takes a complex and a real number \n"
//     "          as argument and returns True or False.\n\n"
//     "n         integer.  If negative, the default value is used.\n\n"
//     "ldA       nonnegative integer.  ldA >= max(1,n).\n"
//     "          If zero, the default value is used.\n\n"
//     "ldB       nonnegative integer.  ldB >= max(1,n).\n"
//     "          If zero, the default value is used.\n\n"
//     "ldVl      nonnegative integer.  ldVl >= 1 and ldVl >= n if Vl \n"
//     "          is present.  If zero, the default value is used (with \n"
//     "          Vl.size[0] replaced by 0 if Vl is None).\n\n"
//     "ldVr      nonnegative integer.  ldVr >= 1 and ldVr >= n if Vr \n"
//     "          is present.  If zero, the default value is used (with \n"
//     "          Vr.size[0] replaced by 0 if Vr is None).\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer\n\n"
//     "offseta   nonnegative integer\n\n"
//     "offsetb   nonnegative integer\n\n"
//     "offsetVl  nonnegative integer\n\n"
//     "offsetVr  nonnegative integer\n\n"
//     "sdim      number of eigenvalues that satisfy the selection\n"
//     "          criterion specified by select.";

// static PyObject *py_select_gr;
// static PyObject *py_select_gc;

// extern int fselect_gc(complex_t *w, double *v)
// {
//    PyObject *wpy, *vpy, *result;
//    int a = 0;

//    wpy = PyComplex_FromDoubles(creal(*w), cimag(*w));
//    vpy = PyFloat_FromDouble(*v);
//    if (!(result = PyObject_CallFunctionObjArgs(py_select_gc, wpy, vpy,
//        NULL))) {
//        Py_XDECREF(wpy); Py_XDECREF(vpy);
//        return -1;
//    }
// #if PY_MAJOR_VERSION >= 3
//    if (PyLong_Check(result)) a = (int) PyLong_AsLong(result);
// #else
//    if PyInt_Check(result) a = (int) PyInt_AsLong(result);
// #endif
//    else
//        PyErr_SetString(PyExc_TypeError, "select() must return an integer "
//            "argument");
//    Py_XDECREF(wpy);  Py_XDECREF(vpy);  Py_XDECREF(result);
//    return a;
// }

// extern int fselect_gr(double *wr, double *wi, double *v)
// {
//    PyObject *wpy, *vpy, *result;
//    int a = 0;

//    wpy = PyComplex_FromDoubles(*wr, *wi);
//    vpy = PyFloat_FromDouble(*v);
//    if (!(result = PyObject_CallFunctionObjArgs(py_select_gr, wpy, vpy,
//        NULL))) {
//        Py_XDECREF(wpy); Py_XDECREF(vpy);
//        return -1;
//    }
// #if PY_MAJOR_VERSION >= 3
//    if (PyLong_Check(result)) a = (int) PyLong_AsLong(result);
// #else
//    if PyInt_Check(result) a = (int) PyInt_AsLong(result);
// #endif
//    else
//        PyErr_SetString(PyExc_TypeError, "select() must return an integer "
//            "argument");
//    Py_XDECREF(wpy);  Py_XDECREF(vpy);  Py_XDECREF(result);
//    return a;
// }

// static PyObject* gges(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     PyObject *F=NULL;
//     matrix *A, *B, *a=NULL, *b=NULL, *Vsl=NULL, *Vsr=NULL;
//     int n=-1, ldA=0, ldB=0, ldVsl=0, ldVsr=0, oA=0, oB=0, oa=0, ob=0,
//         oVsl=0, oVsr=0, info, lwork, sdim, k, *bwork=NULL;
//     double *ar=NULL, *ai=NULL, *rwork=NULL;
//     complex_t *ac=NULL;
//     void *work=NULL, *bc=NULL;
//     number wl;

//     char *kwlist[] = {"A", "B", "a", "b", "Vl", "Vr", "select", "n",
//         "ldA", "ldB", "ldVl", "ldVr", "offsetA", "offsetB", "offseta",
//         "offsetb", "offsetVl", "offsetVr", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OOOOOiiiiiiiiiii",
//         kwlist, &A, &B, &a, &b, &Vsl, &Vsr, &F, &n, &ldA, &ldB, &ldVsl,
//         &ldVsr, &oA, &oB, &oa, &ob, &oVsl, &oVsr)) return NULL;

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(B) != MAT_ID(A)) err_conflicting_ids;
//     if (n < 0){
//         n = A->nrows;
//         if (n != A->ncols){
//             PyErr_SetString(PyExc_TypeError, "A must be square");
//             return NULL;
//         }
//         if (n != B->nrows || n != B->ncols){
//             PyErr_SetString(PyExc_TypeError, "B must be square and of the "
//                 "same order as A");
//             return NULL;
//         }
//     }
//     if (ldA == 0) ldA = MAX(1,A->nrows);
//     if (ldA < MAX(1,n)) err_ld("ldA");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
//     if (ldB == 0) ldB = MAX(1, B->nrows);
//     if (ldB < MAX(1,n)) err_ld("ldB");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (n-1)*ldB + n > len(B)) err_buf_len("B");

//     if (a){
//         if (!Matrix_Check(a) || MAT_ID(a) != COMPLEX)
//             PY_ERR_TYPE("a must be a matrix with typecode 'z'")
//         if (oa < 0) err_nn_int("offseta");
//         if (oa + n > len(a)) err_buf_len("a");
//         if (!b){
//             PyErr_SetString(PyExc_ValueError, "'b' must be provided if "
//                 "'a' is provided");
//             return NULL;
//         }
//     }
//     if (b){
//         if (!Matrix_Check(b) || MAT_ID(b) != DOUBLE)
//             PY_ERR_TYPE("b must be a matrix with typecode 'd'")
//         if (ob < 0) err_nn_int("offsetb");
//         if (ob + n > len(b)) err_buf_len("b");
//         if (!a){
//             PyErr_SetString(PyExc_ValueError, "'a' must be provided if "
//                 "'b' is provided");
//             return NULL;
//         }
//     }

//     if (Vsl){
//         if (!Matrix_Check(Vsl)) err_mtrx("Vsl");
//         if (MAT_ID(Vsl) != MAT_ID(A)) err_conflicting_ids;
//         if (ldVsl == 0) ldVsl = MAX(1, Vsl->nrows);
//         if (ldVsl < MAX(1,n)) err_ld("ldVsl");
//         if (oVsl < 0) err_nn_int("offsetVsl");
//         if (oVsl + (n-1)*ldVsl + n > len(Vsl)) err_buf_len("Vsl");
//     } else {
//         if (ldVsl == 0) ldVsl = 1;
//         if (ldVsl < 1) err_ld("ldVsl");
//     }

//     if (Vsr){
//         if (!Matrix_Check(Vsr)) err_mtrx("Vsr");
//         if (MAT_ID(Vsr) != MAT_ID(A)) err_conflicting_ids;
//         if (ldVsr == 0) ldVsr = MAX(1, Vsr->nrows);
//         if (ldVsr < MAX(1,n)) err_ld("ldVsr");
//         if (oVsr < 0) err_nn_int("offsetVsr");
//         if (oVsr + (n-1)*ldVsr + n > len(Vsr)) err_buf_len("Vsr");
//     } else {
//         if (ldVsr == 0) ldVsr = 1;
//         if (ldVsr < 1) err_ld("ldVsr");
//     }

//     if (F && !PyFunction_Check(F))
//         PY_ERR_TYPE("select must be a Python function")

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             lwork = -1;
//             dgges_(Vsl ? "V" : "N", Vsr ? "V" : "N", F ? "S" : "N", NULL,
//                 &n, NULL, &ldA, NULL, &ldB, &sdim, NULL, NULL, NULL, NULL,
//                 &ldVsl, NULL, &ldVsr, &wl.d, &lwork, NULL, &info);
//             lwork = (int) wl.d;
//             work = (void *) calloc(lwork, sizeof(double));
//             ar = (double *) calloc(n, sizeof(double));
//             ai = (double *) calloc(n, sizeof(double));
//             if (!b) bc = (double *) calloc(n, sizeof(double));
//             if (F) bwork = (int *) calloc(n, sizeof(int));
//             if (!work || !ar || !ai || (!b && !bc) || (F && !bwork)){
//                 free(work);  free(ar);  free(ai);  free(b);  free(bwork);
//                 return PyErr_NoMemory();
//             }
//             py_select_gr = F;
//             dgges_(Vsl ? "V" : "N", Vsr ? "V" : "N", F ? "S" : "N",
//                 F ? &fselect_gr : NULL, &n, MAT_BUFD(A) + oA, &ldA,
//                 MAT_BUFD(B) + oB, &ldB, &sdim, ar, ai,
//                 b ? MAT_BUFD(b) + ob : (double *) bc,
//                 Vsl ? MAT_BUFD(Vsl) + oVsl : NULL, &ldVsl,
//                 Vsr ? MAT_BUFD(Vsr) + oVsr : NULL, &ldVsr,
//                 (double *) work, &lwork, bwork, &info);
//             if (a) for (k=0; k<n; k++)
// #ifndef _MSC_VER
//                 MAT_BUFZ(a)[oa + k] = ar[k] + I * ai[k];
// #else
// 	        MAT_BUFZ(a)[oa + k] = _Cbuild(ar[k],ai[k]);
// #endif
//             free(work);  free(ar);  free(ai);  free(bc); free(bwork);
//             break;

// 	case COMPLEX:
//             lwork = -1;
//             zgges_(Vsl ? "V" : "N", Vsr ? "V" : "N", F ? "S" : "N", NULL,
//                 &n, NULL, &ldA, NULL, &ldB, &sdim, NULL, NULL, NULL,
//                 &ldVsl, NULL, &ldVsr, &wl.z, &lwork, NULL, NULL, &info);
//             lwork = (int) creal(wl.z);
//             work = (void *) calloc(lwork, sizeof(complex_t));
//             rwork = (double *) calloc(8*n, sizeof(double));
//             if (F) bwork = (int *) calloc(n, sizeof(int));
//             if (!a) 
//                 ac = (complex_t *) calloc(n, sizeof(complex_t));
//             bc = (complex_t *) calloc(n, sizeof(complex_t));
// 	    if (!work || !rwork || (F && !bwork) || (!a && !ac) || !bc){
//                 free(work);  free(rwork); free(bwork); free(ac); free(bc);
//                 return PyErr_NoMemory();
//             }
//             py_select_gc = F;
//             zgges_(Vsl ? "V": "N", Vsr ? "V" : "N", F ? "S" : "N",
//                 F ? &fselect_gc : NULL, &n, MAT_BUFZ(A) + oA, &ldA,
//                 MAT_BUFZ(B) + oB, &ldB, &sdim, a ? MAT_BUFZ(a) + oa : ac,
//                 (complex_t *) bc, 
//                 Vsl ? MAT_BUFZ(Vsl) + oVsl : NULL, &ldVsl,
//                 Vsr ? MAT_BUFZ(Vsr) + oVsr : NULL, &ldVsr,
//                 (complex_t *) work, &lwork, rwork,  bwork, &info);
//             if (b) for (k=0; k<n; k++)
//                 MAT_BUFD(b)[ob + k] = 
//                     (double) creal(((complex_t *) bc)[k]);

//             free(work);  free(rwork); free(bwork); free(ac);  free(bc);
//             break;

//         default:
//             err_invalid_id;
//     }

//     if (PyErr_Occurred()) return NULL;

//     if (info) err_lapack
//     else return Py_BuildValue("i", F ? sdim : 0);
// }


// static char doc_lacpy[] =
//     "Copy all or part of a matrix.\n\n"
//     "lacpy(A, B, uplo='N', m=A.size[0], n=A.size[1], \n"
//     "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, \n"
//     "      offsetB=0)\n\n"
//     "PURPOSE\n"
//     "Copy the m x n matrix A to B.  If uplo is 'U', the upper\n"
//     "trapezoidal part of A is copied.  If uplo is 'L', the lower \n"
//     "trapezoidal part is copied.  if uplo is 'N', the entire matrix is\n"
//     "copied.\n\n"
//     "ARGUMENTS\n"
//     "A         'd' or 'z' matrix\n\n"
//     "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
//     "uplo      'N', 'L' or 'U'\n\n"
//     "m         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "n         nonnegative integer.  If negative, the default value is\n"
//     "          used.\n\n"
//     "ldA       positive integer.  ldA >= max(1,m).  If zero, the default\n"
//     "          value is used.\n\n"
//     "ldB       positive integer.  ldB >= max(1,m).  If zero, the default\n"
//     "          value is used.\n\n"
//     "offsetA   nonnegative integer\n\n"
//     "offsetB   nonnegative integer";

// static PyObject* lacpy(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *A, *B;
//     int m = -1, n = -1, ldA = 0, ldB = 0, oA = 0, oB = 0;
// #if PY_MAJOR_VERSION >= 3
//     int uplo_ = 'N';
// #endif
//     char uplo = 'N';
//     char *kwlist[] = {"A", "B", "uplo", "m", "n", "ldA", "ldB", "offsetA",
//         "offsetB", NULL};

// #if PY_MAJOR_VERSION >= 3
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciiiiii", kwlist,
//         &A, &B, &uplo_, &m, &n, &ldA, &ldB, &oA, &oB))
//         return NULL;
//     uplo = (char) uplo_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciiiiii", kwlist,
//         &A, &B, &uplo, &m, &n, &ldA, &ldB, &oA, &oB))
//         return NULL;
// #endif

//     if (!Matrix_Check(A)) err_mtrx("A");
//     if (!Matrix_Check(B)) err_mtrx("B");
//     if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
//     if (uplo != 'N' && uplo != 'L' && uplo != 'U')
//         err_char("trans", "'N', 'L', 'U'");
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = A->ncols;
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA < MAX(1, m)) err_ld("ldA");
//     if (ldB == 0) ldB = MAX(1, B->nrows);
//     if (ldB < MAX(1, m)) err_ld("ldB");
//     if (oA < 0) err_nn_int("offsetA");
//     if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
//     if (oB < 0) err_nn_int("offsetB");
//     if (oB + (n-1)*ldB + m > len(B)) err_buf_len("B");

//     switch (MAT_ID(A)){
//         case DOUBLE:
//             dlacpy_(&uplo, &m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB,
//                 &ldB);
//             break;

//         case COMPLEX:
//             zlacpy_(&uplo, &m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB,
//                 &ldB);
//             break;

// 	default:
//             err_invalid_id;
//     }

//     return Py_BuildValue("");
// }


// static char doc_larfg[] =
//     "Generate an elementary Householder reflector.\n\n"
//     "tau = larfg(alpha, x, n=None, offseta=0, offsetx=0)\n\n"
//     "PURPOSE\n"
//     "Generates a Householder reflector\n\n"
//     "    H = I - tau * [1; v] * [1; v]^H\n\n"
//     "such that\n\n"
//     "    H^H * [alpha; x] = [beta; 0].\n\n"
//     "In other words,\n\n"
//     "    (I - tau.conjugate() * [1; v] * [1; v]^H) * [alpha; x] = "
//     "[beta; 0].\n\n"
//     "The matrix H is unitary, so\n\n"
//     "    2 * tau.real = abs(tau)**2 * ( 1.0 + ||v||**2).\n\n"
//     "On exit x contains the vector v and alpha is overwritten with beta.\n"
//     "The parameter tau is returned as the output value of the function."
//     "\n\n"
//     "ARGUMENTS\n"
//     "alpha     'd' or 'z' matrix.  On exit, contains beta.\n\n"
//     "x         'd' or 'z' matrix.  Must have the same type as alpha.\n"
//     "          On exit, contains v. \n\n"
//     "n         positive integer.  The dimension of the vector [alpha; x].\n"
//     "          If n <= 0, the default value is used, which is equal to\n"
//     "          1 + ( (len(x) - offsetx >= 1) ? len(x) - ox : 0 ).\n\n"
//     "offseta   nonnegative integer \n\n"
//     "offsetx   nonnegative integer \n\n"
//     "tau       scalar of the same type as alpha and x";

// static PyObject* larfg(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *a, *x;
//     number tau;
//     int n = 0, oa = 0, ox = 0, ix = 1; 
//     char *kwlist[] = {"alpha", "x", "n", "offseta", "offsetx", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iii", kwlist,
//         &a, &x, &n, &oa, &ox)) return NULL;

//     if (!Matrix_Check(a)) err_mtrx("alpha");
//     if (!Matrix_Check(x)) err_mtrx("x");
//     if (MAT_ID(a) != MAT_ID(x)) err_conflicting_ids;
//     if (oa < 0) err_nn_int("offseta");
//     if (ox < 0) err_nn_int("offsetx");
//     if (n <= 0) n = 1 + ((len(x) >= ox + 1) ? len(x) - ox : 0);
//     if (len(x) < ox + n - 1) err_buf_len("x");
//     if (len(a) < oa + 1) err_buf_len("alpha");

//     switch (MAT_ID(a)){
//         case DOUBLE:
//             Py_BEGIN_ALLOW_THREADS
//             dlarfg_(&n, MAT_BUFD(a)+oa, MAT_BUFD(x)+ox, &ix, &tau.d);
//             Py_END_ALLOW_THREADS
//             return Py_BuildValue("d", tau.d);
//             break;

//         case COMPLEX:
//             Py_BEGIN_ALLOW_THREADS
//             zlarfg_(&n, MAT_BUFZ(a)+oa, MAT_BUFZ(x)+ox, &ix, &tau.z);
//             Py_END_ALLOW_THREADS
//             return PyComplex_FromDoubles(creal(tau.z), cimag(tau.z));
//             break;

// 	default:
//             err_invalid_id;
//     }

//     return Py_BuildValue("");
// }


// static char doc_larfx[] =
//     "Apply an elementary Householder reflector to a matrix.\n\n"
//     "larfx(v, tau, C, side='L', m=C.size[0], n=C.size[1],\n" 
//     "      ldC=max(1,C.size[0]), offsetv=0, offsetC=0)\n\n"
//     "PURPOSE\n"
//     "Computes H*C (side is 'L') or C*H (side is 'R') where\n\n"
//     "    H = I - tau * v * v^H.\n\n"
//     "On exit C is overwritten with the result.\n\n"
//     "ARGUMENTS\n"
//     "v         'd' or 'z' matrix\n\n"
//     "tau       number.  Can only be complex if v is complex.\n\n"
//     "C         'd' or 'z' matrix of the same type as v\n\n"
//     "side      'L' or 'R'\n\n"
//     "m         nonnegative integer.  If negative, the default value is \n"
//     "          used.\n\n" 
//     "n         nonnegative integer.  If negative, the default value is \n"
//     "          used.\n\n" 
//     "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
//     "          default value is used.\n\n"
//     "offsetv   nonnegative integer \n\n"
//     "offsetC   nonnegative integer";

// static PyObject* larfx(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *v, *C;
//     PyObject *tauo=NULL;
//     number tau;
//     int m = -1, n = -1, ov = 0, oC = 0, ldC = 0; 
//     void *work = NULL;
// #if PY_MAJOR_VERSION >= 3
//     int side_ = 'L';
// #endif
//     char side = 'L';
//     char *kwlist[] = {"v", "tau", "C", "side", "m", "n", "ldC", "offsetv",
//         "offsetC", NULL};

// #if PY_MAJOR_VERSION >= 3 
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Ciiiii", kwlist, 
//         &v, &tauo, &C, &side_, &m, &n, &ldC, &ov, &oC))
//         return NULL;
//     side = (char) side_;
// #else
//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ciiiii", kwlist, 
//         &v, &tauo, &C, &side, &m, &n, &ldC, &ov, &oC))
//         return NULL;
// #endif
 
//     if (!Matrix_Check(v)) err_mtrx("v");
//     if (!Matrix_Check(C)) err_mtrx("C");
//     if (MAT_ID(v) != MAT_ID(C)) err_conflicting_ids;
//     if (tauo && number_from_pyobject(tauo, &tau, MAT_ID(v)))
//         err_type("tau")

//     if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");

//     if (m < 0) m = C->nrows;
//     if (n < 0) n = C->ncols;
    
//     if (ov < 0) err_nn_int("offsetv");
//     if ((side == 'L' && len(v) - ov < m) ||
//         (side == 'R' && len(v) - ov < n)) err_buf_len("v")

//     if (ldC == 0) ldC = MAX(1, C->nrows);
//     if (ldC < MAX(1,m)) err_ld("ldC");
//     if (oC < 0) err_nn_int("offsetC");
//     if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");


//     switch (MAT_ID(v)){
//         case DOUBLE:
//             if (!(work = (void *) calloc((side == 'L') ? n : m, 
//                 sizeof(double))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             dlarfx_(&side, &m, &n, MAT_BUFD(v)+ov, &tau.d, 
//                 MAT_BUFD(C) + oC, &ldC, (double *) work);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

//         case COMPLEX:
//             if (!(work = (void *) calloc((side == 'L') ? n : m,
//                 sizeof(complex_t))))
//                 return PyErr_NoMemory();
//             Py_BEGIN_ALLOW_THREADS
//             zlarfx_(&side, &m, &n, MAT_BUFZ(v)+ov, &tau.z,
//                 MAT_BUFZ(C) + oC, &ldC, (complex_t *) work);
//             Py_END_ALLOW_THREADS
//             free(work);
//             break;

// 	default:
//             err_invalid_id;
//     }

//     return Py_BuildValue("");
// }

