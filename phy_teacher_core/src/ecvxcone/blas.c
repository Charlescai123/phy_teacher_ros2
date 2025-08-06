#define BLAS_MODULE

#include "blas.h"

/**
 * @brief Converts void pointer to number struct
 * @param src Input pointer (double or double complex)
 * @param a Output number struct to store result
 * @param id Type ID (DOUBLE or COMPLEX)
 * @return 0 on success, -1 on error
 * 
 * @note src must match the specified type (DOUBLE/COMPLEX)
 * @warning a must point to valid allocated memory
 */
int number_from_raw(const void *src, number *a, int id) {
    if (!src || !a) return -1;

    switch (id) {
        case DOUBLE:
            a->d = *(const double *)src;
            return 0;

        case COMPLEX:
            a->z = *(const double complex *)src;
            return 0;

        default:
            return -1;
    }
}

/**
 * @brief Copies a vector x to a vector y (y := x).
 * 
 * @details
 * Performs the vector copy operation y = x.
 * This implements the BLAS COPY operation.
 *
 * @param[in] x       Source vector ('d' for real, 'z' for complex)
 * @param[out] y      Destination vector (same type as x)
 * @param[in] n       Number of elements to copy. If negative, default is:
 *                    (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0
 * @param[in] incx    Stride for x vector (nonzero integer)
 * @param[in] incy    Stride for y vector (nonzero integer)
 * @param[in] offsetx Starting offset in x vector (nonnegative integer)
 * @param[in] offsety Starting offset in y vector (nonnegative integer)
 *
 * @note
 * - Implements the standard BLAS COPY operation
 * - For complex vectors, performs complex copy
 * - When n < 0, automatically determines length based on x's dimensions
 * - The operation overwrites the contents of y
 *
 * @warning
 * - Vectors x and y must be of same type ('d' or 'z')
 * - Strides incx and incy must be nonzero
 * - Offsets must be nonnegative
 * - Destination vector y must have sufficient capacity
 *
 * @see BLAS Level 1 Specification for COPY
 */
void blas_copy(matrix *x, matrix *y, int n, int ix, int iy, int ox, int oy)
{
    // Set default values
    if (n == -1) n = -1;  // Use -1 as indicator for default
    if (ix == 0) ix = 1;  // Default increment
    if (iy == 0) iy = 1;  // Default increment
    if (ox == -1) ox = 0; // Default offset
    if (oy == -1) oy = 0; // Default offset

    // Input validation
    if (!x) err_mtrx("x");
    if (!y) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    // Calculate default n if needed
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
    if (n == 0) return; // Nothing to copy

    // Buffer length validation
    if (len(x) < ox+1+(n-1)*abs(ix)) err_buf_len("x");
    if (len(y) < oy+1+(n-1)*abs(iy)) err_buf_len("y");

    // Perform the copy operation based on matrix type
    switch (MAT_ID(x)){
        case DOUBLE:
            dcopy_(&n, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy);
            break;

        case COMPLEX:
            zcopy_(&n, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy);
            break;

        default:
            err_invalid_id;
    }
}

/**
 * @brief Constant times a vector plus a vector (y := alpha*x + y).
 * 
 * @details
 * Performs the scaled vector addition operation y = alpha*x + y.
 * This is the BLAS AXPY operation.
 *
 * @param[in] x       Input vector ('d' for real, 'z' for complex)
 * @param[in,out] y   Input/output vector of same type as x
 * @param[in] alpha   Scaling factor (number - int, float or complex)
 *                    Complex alpha only allowed when x is complex
 * @param[in] n       Vector length. If negative, default is calculated as:
 *                    (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0
 * @param[in] incx    x vector stride (nonzero integer)
 * @param[in] incy    y vector stride (nonzero integer)
 * @param[in] offsetx Starting offset in x vector (nonnegative integer)
 * @param[in] offsety Starting offset in y vector (nonnegative integer)
 *
 * @note
 * - Implements the standard BLAS AXPY operation
 * - For complex vectors, performs complex arithmetic
 * - When n < 0, automatically calculates compatible vector length
 * - The operation is performed in-place on y
 *
 * @warning
 * - Vectors x and y must be of same type ('d' or 'z')
 * - Strides incx and incy must be nonzero
 * - Offsets must be nonnegative
 * - Complex alpha requires complex x and y
 * - Default length calculation must produce compatible length for x and y
 *
 * @see BLAS Level 1 Specification for AXPY
 */
void blas_axpy(matrix *x, matrix *y, number *alpha, int n, int incx, int incy, int offsetx, int offsety)
{
    number a;
    int ix = incx, iy = incy, ox = offsetx, oy = offsety;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
    if (n == 0) return;

    if (len(x) < ox + 1+(n-1)*abs(ix)) err_buf_len("x");
    if (len(y) < oy + 1+(n-1)*abs(iy)) err_buf_len("y");

    if (alpha) a = *alpha;

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!alpha) a.d=1.0;
            daxpy_(&n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy);
            break;

        case COMPLEX:
            if (!alpha) a.z=1.0+0.0*I;
            zaxpy_(&n, &a.z, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy);
            break;

        default:
            err_invalid_id;
    }
}


/**
 * @brief Scales a vector by a constant (x := alpha*x).
 * 
 * blas_scal(alpha, x, n, inc, offset)
 * 
 * @param alpha     number (double or double complex). Complex alpha is only
 *                  allowed if x is complex.
 * 
 * @param x         'd' or 'z' matrix
 * 
 * @param n         integer. If n<0, the default value of n is used.
 *                  The default value is equal to
 *                  (len(x)>=offset+1) ? 1+(len-offset-1)/inc : 0. (default = -1)
 * 
 * @param inc       positive integer (default = 1)
 * 
 * @param offset    nonnegative integer (default = 0)
 */

void blas_scal(void* alpha, matrix* x, int n, int inc, int offset)
{              
    number a;
    int ix = inc, ox = offset;

    // Default parameter handling
    if (n < 0) n = -1;
    if (ix == 0) ix = 1;
    if (ox < 0) ox = 0;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("inc");
    if (ox < 0) err_nn_int("offset");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return;
    if (len(x) < ox+1+(n-1)*ix) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (number_from_raw(alpha, &a, MAT_ID(x)))
                err_type("alpha");
            dscal_(&n, &a.d, MAT_BUFD(x)+ox, &ix);
	    break;

        case COMPLEX:
            if (!number_from_raw(alpha, &a, DOUBLE))
                zdscal_(&n, &a.d, MAT_BUFZ(x)+ox, &ix);
            else if (!number_from_raw(alpha, &a, COMPLEX))
                zscal_(&n, &a.z, MAT_BUFZ(x)+ox, &ix);
            else
                err_type("alpha");
	    break;

        default:
            err_invalid_id;
    }

}


/**
 * @brief Returns the Euclidean norm of a vector (returns ||x||_2).
 * 
 * This function computes the Euclidean norm (L2 norm) of a vector using
 * BLAS-style parameters for flexible vector access with stride and offset.
 *
 * @param x Pointer to 'd' (double precision) or 'z' (complex double) matrix/vector data
 * @param n Integer representing the number of elements to process.
 *          If n < 0, the default value of n is used.
 *          The default value is equal to:
 *          (len(x) >= offsetx + 1) ? 1 + (len(x) - offsetx - 1) / incx : 0
 *          (default = -1)
 * @param inc Positive integer representing the increment/stride between elements
 *            (default = 1)
 * @param offset Nonnegative integer representing the starting offset in the array
 *               (default = 0)
 * 
 * @return Returns 0 if n = 0, otherwise returns the Euclidean norm ||x||_2
 *
 * @note This function follows BLAS naming conventions and parameter ordering.
 *       The function handles both real (double) and complex (double complex) 
 *       data types.
 */
double blas_nrm2(matrix *x, int n, int inc, int offset)
{
    int ix = inc, ox = offset;

    // Default parameter handling
    if (n < 0) n = -1;
    if (ix <= 0) ix = 1;
    if (ox < 0) ox = 0;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("incx");
    if (ox < 0) err_nn_int("offsetx");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return 0.0;
    if (len(x) < ox + 1+(n-1)*ix) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            return dnrm2_(&n, MAT_BUFD(x)+ox, &ix);

        case COMPLEX:
            return dznrm2_(&n, MAT_BUFZ(x)+ox, &ix);

        default:
            err_invalid_id;
    }
    
    return 0.0; // Should never reach here
}


/**
 * @brief Solution of a triangular and banded set of equations.
 *
 * @details
 * This function computes the solution of a triangular and banded set of equations.
 *
 * If trans is 'N', computes x := A^{-1}*x.
 * If trans is 'T', computes x := A^{-T}*x.
 * If trans is 'C', computes x := A^{-H}*x.
 * A is banded triangular of order n and with bandwidth k.
 *
 * @param A         Pointer to the 'd' (double) or 'z' (complex double) matrix.
 * @param x         Pointer to the 'd' (double) or 'z' (complex double) vector/matrix. 
 *                  Must have the same type as A.
 * @param uplo      Character. Specifies whether the upper or lower triangular part 
 *                  of the matrix A is to be referenced.
 *                  - 'L': Lower triangular part.
 *                  - 'U': Upper triangular part.
 * @param trans     Character. Specifies the form of the system of equations:
 *                  - 'N': A * x = b (no transpose)
 *                  - 'T': A^T * x = b (transpose)
 *                  - 'C': A^H * x = b (conjugate transpose)
 * @param diag      Character. Specifies whether A is unit triangular.
 *                  - 'N': A is not unit triangular.
 *                  - 'U': A is unit triangular.
 * @param n         Nonnegative integer. The order of the matrix A. If negative, 
 *                  the default value is used.
 * @param k         Nonnegative integer. The number of super-diagonals or sub-diagonals 
 *                  of the matrix A. If negative, the default value is used.
 * @param ldA       Nonnegative integer. The leading dimension of the array A. ldA >= 1+k. 
 *                  If zero, the default value is used.
 * @param incx      Nonzero integer. The increment for the elements of x.
 * @param offsetA   Nonnegative integer. The offset from the start of the array A.
 * @param offsetx   Nonnegative integer. The offset from the start of the array x.
 */
void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
          int n, int k, int ldA, int incx, int offsetA, int offsetx)
{
    // Set default values
    if (n < 0) n = -1;
    if (k < 0) k = -1;
    if (ldA == 0) ldA = 0;
    if (incx == 0) incx = 1;
    if (offsetA < 0) offsetA = 0;
    if (offsetx < 0) offsetx = 0;
    
    // Set default character values
    if (uplo == 0) uplo = 'L';
    if (trans == 0) trans = 'N';
    if (diag == 0) diag = 'N';

    // Validate matrix inputs
    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    // Validate character parameters
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");

    // Validate incx parameter
    if (incx == 0) err_nz_int("incx");

    // Set default values based on matrix dimensions
    if (n < 0) n = A->ncols;
    if (n == 0) return;
    if (k < 0) k = MAX(0, A->nrows-1);

    // Validate and set leading dimension
    if (ldA == 0) ldA = A->nrows;
    if (ldA < k+1) err_ld("ldA");

    // Validate offset parameters
    if (offsetA < 0) err_nn_int("offsetA");
    if (offsetA + (n-1)*ldA + k + 1 > len(A)) err_buf_len("A");
    if (offsetx < 0) err_nn_int("offsetx");
    if (offsetx + (n-1)*abs(incx) + 1 > len(x)) err_buf_len("x");

    // Call appropriate BLAS routine based on matrix type
    switch (MAT_ID(x)){
        case DOUBLE:
            dtbsv_(&uplo, &trans, &diag, &n, &k, MAT_BUFD(A)+offsetA, &ldA,
                MAT_BUFD(x)+offsetx, &incx);
            break;

        case COMPLEX:
            ztbsv_(&uplo, &trans, &diag, &n, &k, MAT_BUFZ(A)+offsetA, &ldA,
                MAT_BUFZ(x)+offsetx, &incx);
            break;

        default:
            err_invalid_id;
    }
}

/**
 * @brief Computes the dot product xᴴy for real or complex vectors x and y.
 * 
 * @details 
 * Returns the inner product (conjugate transpose) of vectors x and y.
 * Returns 0 if n=0.
 *
 * @param[in] x       Input vector ('d' for real, 'z' for complex)
 * @param[in] y       Input vector of same type as x
 * @param[in] n       Length of vectors. If negative, default is calculated as:
 *                    - For x: (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0
 *                    - For y: must match calculated length of x
 * @param[in] incx    Stride for x vector (nonzero integer)
 * @param[in] incy    Stride for y vector (nonzero integer)
 * @param[in] offsetx Starting offset in x vector (nonnegative integer)
 * @param[in] offsety Starting offset in y vector (nonnegative integer)
 * 
 * @return The dot product xᴴy
 *
 * @note 
 * - The default length calculation ensures compatibility between x and y
 * - When using default length, the calculated length for x and y must match
 * - For complex vectors, this computes the conjugate dot product
 *
 * @warning 
 * - Both incx and incy must be nonzero
 * - offsetx and offsety must be nonnegative
 * - Vectors x and y must be of same type ('d' or 'z')
 */

number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, int offsetx, int offsety)
{
    number val;

    int ix = incx;
    int iy = incy;
    int ox = offsetx;
    int oy = offsety;

    // Set default values
    if (n < 0) n = -1;
    if (ix == 0) ix = 1;
    if (iy == 0) iy = 1;
    if (ox < 0) ox = 0;
    if (oy < 0) oy = 0;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n < 0){
        n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
        if (n != ((len(y) >= oy+1) ? 1+(len(y)-oy-1)/abs(iy) : 0)){
            ERR("arrays have unequal default lengths");
        }
    }

    if (n && len(x) < ox + 1 + (n-1)*abs(ix)) err_buf_len("x");
    if (n && len(y) < oy + 1 + (n-1)*abs(iy)) err_buf_len("y");

    switch (MAT_ID(x)){
        case DOUBLE:
            val.d = (n==0) ? 0.0 : ddot_(&n, MAT_BUFD(x)+ox, &ix,
                MAT_BUFD(y)+oy, &iy);
            return val;

        case COMPLEX:
#ifndef _MSC_VER
	        if (n==0) val.z = 0.0;
#else
	        if (n==0) val.z = _Cbuild(0.0,0.0);
#endif
	        else
#if USE_CBLAS_ZDOT
                cblas_zdotc_sub(n, MAT_BUFZ(x)+ox, ix, MAT_BUFZ(y)+oy,
                    iy, &val.z);
#else
                ix *= 2;
                iy *= 2;
#ifndef _MSC_VER
                val.z = (ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy, &iy) +
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy)) +
                    I*(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy) -
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy, &iy));
#else
                val.z = _Cbuild(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy, &iy) +
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy),
				ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy) -
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy, &iy));
#endif

#endif
	    return val;

        default:
            err_invalid_id;
    }
}


/**
 * @brief Matrix-vector product with a triangular band matrix.
 * 
 * @details 
 * If trans is 'N', computes x := A*x.\n
 * If trans is 'T', computes x := A^T*x.\n
 * If trans is 'C', computes x := A^H*x.\n
 * A is banded triangular of order n and with bandwidth k.
 * 
 * @param[in] A       'd' or 'z' matrix
 * @param[in,out] x   'd' or 'z' matrix. Must have the same type as A.
 * @param[in] uplo    'L' or 'U'    (default = 'L')
 * @param[in] trans   'N', 'T' or 'C'   (default = 'N')
 * @param[in] diag    'N' or 'U'    (default = 'N')
 * @param[in] n       Nonnegative integer. If negative, default value -1 is used.
 * @param[in] k       Nonnegative integer. If negative, default value -1 is used.
 * @param[in] ldA     Nonnegative integer. lda >= 1+k. If zero default value 0 is used.
 * @param[in] incx    Nonzero integer (default = 1)
 * @param[in] offsetA Nonnegative integer (default = 0)
 * @param[in] offsetx Nonnegative integer (default = 0)
 * 
 * @note The function performs triangular band matrix-vector multiplication.
 * @warning ldA must satisfy ldA >= 1+k for proper operation.
 */

void blas_tbmv(matrix *A, matrix *x, char uplo, char trans, char diag, 
          int n, int k, int ldA, int incx, int offsetA, int offsetx)
{
    int ix=incx, oA=offsetA, ox=offsetx;

    // Set default values
    if (n < 0) n = -1;
    if (k < 0) k = -1;
    if (ldA == 0) ldA = 0;
    if (ix == 0) ix = 1;
    if (oA < 0) oA = 0;
    if (ox < 0) ox = 0;
    
    // Set default character values
    if (uplo == 0) uplo = 'L';
    if (trans == 0) trans = 'N';
    if (diag == 0) diag = 'N';

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'U', 'N'");

    if (ix == 0) err_nz_int("incx");

    if (n < 0) n = A->ncols;
    if (n == 0) return;
    if (k < 0) k = MAX(0,A->nrows-1);

    if (ldA == 0) ldA = A->nrows;
    if (ldA < k+1)  err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + k + 1 > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            dtbmv_(&uplo, &trans, &diag, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix);
            break;

        case COMPLEX:
            ztbmv_(&uplo, &trans, &diag, &n, &k, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix);
            break;

        default:
            err_invalid_id;
    }
}


/**
 * @brief Triangular matrix-matrix multiplication (BLAS TRMM operation)
 * 
 * trmm(A, B, side='L', uplo='L', transA='N', diag='N', alpha=1.0,
         m=None, n=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0])
*
 * @details
 * Computes one of six possible triangular matrix products:
 * - B := α·A·B    (transA='N', side='L')
 * - B := α·B·A    (transA='N', side='R')
 * - B := α·Aᵀ·B   (transA='T', side='L')
 * - B := α·B·Aᵀ   (transA='T', side='R')
 * - B := α·Aᴴ·B   (transA='C', side='L')
 * - B := α·B·Aᴴ   (transA='C', side='R')
 * 
 * where A is triangular and B is a general matrix.
 *
 * @param[in] A       Triangular matrix ('d' or 'z' type)
 * @param[in,out] B   Input/output matrix (same type as A)
 * @param[in] side    Multiplication side ('L' for left, 'R' for right) (default = 'L')
 * @param[in] uplo    Triangle selection ('L' for lower, 'U' for upper) (default = 'L')
 * @param[in] transA  Transposition of A ('N', 'T', or 'C') (default = 'N')
 * @param[in] diag    Diagonal type ('N' for non-unit, 'U' for unit diagonal) (default = 'N')
 * @param[in] alpha   Scalar multiplier (type must match A)  (default = 1.0)
 * @param[in] m       Rows in B (default based on side and matrix sizes) (default = -1)
 * @param[in] n       Columns in B (default based on side and matrix sizes) (default = -1)
 * @param[in] ldA     Leading dimension of A  (default = 0)
 * @param[in] ldB     Leading dimension of B  (default = 0)
 * @param[in] offsetA Offset in A (nonnegative) (default = 0)
 * @param[in] offsetB Offset in B (nonnegative) (detault = 0)
 *
 * @note
 * - Implements the BLAS TRMM operation
 * - Handles both real and complex matrices
 * - Supports all transposition and side combinations
 * - Default dimensions are derived from matrix sizes when negative
 * - For complex matrices, 'C' performs conjugate transpose
 *
 * @warning
 * - A must be triangular and B must be general
 * - Matrices must have matching types ('d' or 'z')
 * - Complex alpha requires complex matrices
 * - Leading dimensions must satisfy:
 *   - ldA ≥ max(1, (side=='L') ? m : n)
 *   - ldB ≥ max(1, m)
 * - Offsets must be nonnegative
 * - Default dimension rules must be respected
 *
 * @see BLAS Level 3 TRMM documentation
 */

void blas_trmm(matrix *A, matrix *B, char side, char uplo, char transA, char diag,
               void* alpha, int m, int n, int ldA, int ldB, int offsetA, int offsetB)
{
    number a;

    // int m=-1, n=-1, ldA=0, ldB=0, oA=0, oB=0;
    int oA, oB;

    oA = offsetA;
    oB = offsetB;

    // Default values
    if (side == 0) side = 'L';
    if (uplo == 0) side = 'L';
    if (transA == 0) side = 'N';
    if (diag == 0) side = 'N';
    if (ldA == 0) ldA = 0;
    if (ldB == 0) ldB = 0;
    if (oA < 0) oA = 0;
    if (oB < 0) oB = 0;


    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;

    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (transA != 'N' && transA != 'T' && transA != 'C')
        err_char("transA", "'N', 'T', 'C'");

    if (n < 0){
        n = (side == 'L') ? B->ncols : A->nrows;
        if (side != 'L' && n != A->ncols) ERR_TYPE("A must be square");
        
    }
    if (m < 0){
        m = (side == 'L') ? A->nrows: B->nrows;
        if (side == 'L' && m != A->ncols) ERR_TYPE("A must be square");
    }
    if (m == 0 || n == 0) return;

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1, (side == 'L') ? m : n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, m)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if ((side == 'L' && oA + (m-1)*ldA + m > len(A)) ||
        (side == 'R' && oA + (n-1)*ldA + n > len(A))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + m > len(B)) err_buf_len("B");

    if (alpha && number_from_raw(alpha, &a, MAT_ID(A))) err_type("alpha");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!alpha) a.d = 1.0;
            dtrmm_(&side, &uplo, &transA, &diag, &m, &n, &a.d,
                MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB, &ldB);
            break;

        case COMPLEX:
#ifndef _MSC_VER
   	    if (!alpha) a.z = 1.0;
#else
   	    if (!alpha) a.z = _Cbuild(1.0,0.0);
#endif
            ztrmm_(&side, &uplo, &transA, &diag, &m, &n, &a.z,
                MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB, &ldB);
            break;

        default:
            err_invalid_id;
    }

    return;
}


/**
 * @brief Solves triangular systems with multiple right-hand sides (BLAS TRSM operation)
 * 
    blas_trsm(A, B, side='L', uplo='L', transA='N', diag='N', alpha=1.0, m=None, 
              n=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, offsetB=0)
 * 
 * @details
 * Computes one of six possible triangular system solutions:
 * - B := α·A⁻¹·B    (transA='N', side='L')
 * - B := α·B·A⁻¹    (transA='N', side='R')
 * - B := α·A⁻ᵀ·B    (transA='T', side='L')
 * - B := α·B·A⁻ᵀ    (transA='T', side='R')
 * - B := α·A⁻ᴴ·B    (transA='C', side='L')
 * - B := α·B·A⁻ᴴ    (transA='C', side='R')
 * 
 * where A is triangular and B is a general matrix.
 *
 * @param[in] A       Triangular matrix ('d' or 'z' type)
 * @param[in,out] B   Input/output matrix (same type as A)
 * @param[in] side    Multiplication side ('L' for left, 'R' for right) (default = 'L')
 * @param[in] uplo    Triangle selection ('L' for lower, 'U' for upper) (default = 'L')
 * @param[in] transA  Transposition of A ('N', 'T', or 'C') (default = 'N')
 * @param[in] diag    Diagonal type ('N' for non-unit, 'U' for unit diagonal) (default = 'N')
 * @param[in] alpha   Scalar multiplier (type must match A) (default = -1.0)
 * @param[in] m       Rows in B (default based on side and matrix sizes) (default = -1)
 * @param[in] n       Columns in B (default based on side and matrix sizes) (default = -1)
 * @param[in] ldA     Leading dimension of A (default = 0)
 * @param[in] ldB     Leading dimension of B (default = 0)
 * @param[in] offsetA Offset in A (nonnegative) (default = 0)
 * @param[in] offsetB Offset in B (nonnegative) (default = 0)
 *
 * @note
 * - Implements the BLAS TRSM operation
 * - Handles both real and complex matrices
 * - Supports all transposition and side combinations
 * - Default dimensions are derived from matrix sizes when negative
 * - Does not check if A is nonsingular
 * - For complex matrices, 'C' performs conjugate transpose
 *
 * @warning
 * - A must be triangular and B must be general
 * - Matrices must have matching types ('d' or 'z')
 * - Complex alpha requires complex matrices
 * - Leading dimensions must satisfy:
 *   - ldA ≥ max(1, (side=='L') ? m : n)
 *   - ldB ≥ max(1, m)
 * - Offsets must be nonnegative
 * - Default dimension rules must be respected
 * - Behavior is undefined if A is singular
 *
 * @see BLAS Level 3 TRSM documentation
 */
void blas_trsm(matrix *A, matrix *B, char side, char uplo, char transA, char diag,
          void* alpha, int m, int n, int ldA, int ldB, int offsetA, int offsetB)
{
    number a;
    // int m=-1, n=-1, ldA=0, ldB=0, oA=0, oB=0;
    int oA, oB;

    oA = offsetA;
    oB = offsetB;

    // Default values
    if (side == 0) side = 'L';
    if (uplo == 0) side = 'L';
    if (transA == 0) side = 'N';
    if (diag == 0) side = 'N';
    if (ldA == 0) ldA = 0;
    if (ldB == 0) ldB = 0;
    if (oA < 0) oA = 0;
    if (oB < 0) oB = 0;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;

    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (transA != 'N' && transA != 'T' && transA != 'C')
        err_char("transA", "'N', 'T', 'C'");

    if (n < 0){
        n = (side == 'L') ? B->ncols : A->nrows;
        if (side != 'L' && n != A->ncols) ERR("A must be square");
    }
    if (m < 0){
        m = (side == 'L') ? A->nrows: B->nrows;
        if (side == 'L' && m != A->ncols) ERR("A must be square");
    }
    if (n == 0 || m == 0) return;

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1, (side == 'L') ? m : n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,m)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if ((side == 'L' && oA + (m-1)*ldA + m > len(A)) ||
        (side == 'R' && oA + (n-1)*ldA + n > len(A))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB < 0 || oB + (n-1)*ldB + m > len(B)) err_buf_len("B");

    if (alpha && number_from_raw(alpha, &a, MAT_ID(A))) err_type("alpha");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!alpha) a.d = 1.0;
            dtrsm_(&side, &uplo, &transA, &diag, &m, &n, &a.d,
                MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB, &ldB);
            break;

        case COMPLEX:
#ifndef _MSC_VER
  	    if (!alpha) a.z = 1.0;
#else
  	    if (!alpha) a.z = _Cbuild(1.0,0.0);
#endif
            ztrsm_(&side, &uplo, &transA, &diag, &m, &n, &a.z,
                MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB, &ldB);
            break;

        default:
            err_invalid_id;
    }

    return;
}



/**
 * @brief Solves a triangular system with one right-hand side (BLAS TRSV operation)
 * 
 *  blas_trsv(A, x, uplo='L', trans='N', diag='N', n=A.size[0],
              dA=max(1,A.size[0]), incx=1, offsetA=0, offsetx=0)
 * 
 * @details
 * Computes the solution of a triangular system:
 * - x := A⁻¹·x   (trans='N')
 * - x := A⁻ᵀ·x   (trans='T')
 * - x := A⁻ᴴ·x   (trans='C')
 * 
 * where A is an n×n triangular matrix. The function does not verify
 * whether A is nonsingular.
 *
 * @param[in] A       Triangular matrix ('d' or 'z' type)
 * @param[in,out] x   Input/output vector (same type as A)
 * @param[in] uplo    Triangle selection ('L' for lower, 'U' for upper) (default = 'L')
 * @param[in] trans   Transposition of A ('N', 'T', or 'C') (default = 'N')
 * @param[in] diag    Diagonal type ('N' for non-unit, 'U' for unit diagonal) (default = 'N')
 * @param[in] n       Order of matrix A (default: A.size[0])
 * @param[in] ldA     Leading dimension of A (≥ max(1,n))
 * @param[in] incx    Storage spacing between elements of x (nonzero) (default = 1)
 * @param[in] offsetA Offset in A (nonnegative) (default = 0)
 * @param[in] offsetx Offset in x (nonnegative) (default = 0)
 *
 * @note
 * - Implements the BLAS TRSV operation
 * - Handles both real and complex matrices
 * - Supports all transposition modes
 * - For complex matrices, 'C' performs conjugate transpose
 * - Default n requires A to be square (A.size[0] == A.size[1])
 * - Does not check if A is nonsingular
 *
 * @warning
 * - A must be triangular and x must be a vector
 * - Matrix and vector must have matching types ('d' or 'z')
 * - incx must be nonzero
 * - ldA must satisfy ldA ≥ max(1,n)
 * - Offsets must be nonnegative
 * - Behavior is undefined if A is singular
 *
 * @see BLAS Level 2 TRSV documentation
 */
void blas_trsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
                int n, int ldA, int ix, int offsetA, int offsetx)
{
    // int n=-1, ldA=0, ix=1, oA=0, ox=0;

    int oA, ox;

    oA = offsetA;
    ox = offsetx;

    // Default values
    if (uplo == 0) uplo = 'L';
    if (trans == 0) trans = 'N';
    if (diag == 0) diag = 'N';
    if (ldA == 0) ldA = 0;
    if (oA < 0) oA = 0;
    if (ox < 0) ox = 0;


    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");

    if (ix == 0) err_nz_int("incx");

    if (n < 0){
        if (A->nrows != A->ncols){
            ERR("A is not square");
            return;
        }
        n = A->nrows;
    }
    if (n == 0) return;

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            dtrsv_(&uplo, &trans, &diag, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix);
            break;

        case COMPLEX:
            ztrsv_(&uplo, &trans, &diag, &n, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix);
            break;

        default:
            err_invalid_id;
    }

    return;
}


/**
 * @brief Performs rank-k update of a symmetric matrix (BLAS SYRK operation)
 * 
 *     blas_syrk(A, C, uplo='L', trans='N', alpha=1.0, beta=0.0, n=None, k=None, 
 *              ldA=max(1,A.size[0]), ldC=max(1,C.size[0]), offsetA=0, offsetB=0)
 * 
 * @details
 * Computes a symmetric rank-k update:
 * - C := α·A·Aᵀ + β·C   (trans='N')
 * - C := α·Aᵀ·A + β·C   (trans='T')
 *
 * where C is an n×n symmetric matrix and A is either n×k (trans='N')
 * or k×n (trans='T'). If k=0, the operation reduces to C := β·C.
 *
 * @param[in] A       Input matrix ('d' or 'z' type)
 * @param[in,out] C   Input/output symmetric matrix (same type as A)
 * @param[in] uplo    Triangle selection ('L' for lower, 'U' for upper) (default = 'L')
 * @param[in] trans   Transposition of A ('N' or 'T') (default = 'N')
 * @param[in] alpha   Scalar multiplier (type must match A) (default = 1.0)
 * @param[in] beta    Scalar multiplier (type must match A) (default = 0.0)
 * @param[in] n       Order of matrix C (default based on trans and A's size) (default = -1)
 * @param[in] k       Inner dimension of product (default based on trans and A's size) (default = -1)
 * @param[in] ldA     Leading dimension of A (default = 0)
 * @param[in] ldC     Leading dimension of C (default = 0)
 * @param[in] offsetA Offset in A (nonnegative) (default = 0)
 * @param[in] offsetC Offset in C (nonnegative) (default = 0)
 * 
 * @note
 * - Implements the BLAS SYRK operation
 * - Handles both real and complex matrices
 * - For complex matrices, the result is Hermitian when using complex data
 * - Default dimensions are derived from matrix sizes when negative
 * - Only updates the specified triangle (upper or lower) of C
 *
 * @warning
 * - C must be symmetric/Hermitian
 * - Matrices must have matching types ('d' or 'z')
 * - Complex alpha/beta require complex matrices
 * - Leading dimensions must satisfy:
 *   - ldA ≥ max(1, (trans=='N') ? n : k)
 *   - ldC ≥ max(1, n)
 * - Offsets must be nonnegative
 * - Behavior is undefined if dimensions are incompatible
 *
 * @see BLAS Level 3 SYRK documentation
 */
void blas_syrk(matrix *A, matrix *C, char uplo, char trans, void* alpha, void* beta, 
              int n, int k, int ldA, int ldC, int offsetA, int offsetC)
{
    number a, b;
    // int n=-1, k=-1, ldA=0, ldC=0, oA = 0, oC = 0;

    int oA, oC;

    oA = offsetA;
    oC = offsetC;

    // Default values
    if (uplo == 0) uplo = 'L';
    if (trans == 0) trans = 'N';
    if (ldA == 0) ldA = 0;
    if (ldC == 0) ldC = 0;
    if (oA < 0) oA = 0;
    if (oC < 0) oC = 0;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(C)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (MAT_ID(A) == DOUBLE && trans != 'N' && trans != 'T' &&
        trans != 'C') err_char("trans", "'N', 'T', 'C'");
    if (MAT_ID(A) == COMPLEX && trans != 'N' && trans != 'T')
	err_char("trans", "'N', 'T'");

    if (n < 0) n = (trans == 'N') ? A->nrows : A->ncols;
    if (k < 0) k = (trans == 'N') ? A->ncols : A->nrows;
    if (n == 0) return;

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (k > 0 && ldA < MAX(1, (trans == 'N') ? n : k)) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,n)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (k > 0 && ((trans == 'N' && oA + (k-1)*ldA + n > len(A)) ||
        ((trans == 'T' || trans == 'C') && oA + (n-1)*ldA + k > len(A))))
        err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + n > len(C)) err_buf_len("C");

    if (alpha && number_from_raw(alpha, &a, MAT_ID(A))) err_type("alpha");
    if (beta && number_from_raw(beta, &b, MAT_ID(A))) err_type("beta");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!alpha) a.d = 1.0;
            if (!beta) b.d = 0.0;
            dsyrk_(&uplo, &trans, &n, &k, &a.d, MAT_BUFD(A)+oA, &ldA,
                &b.d, MAT_BUFD(C)+oC, &ldC);
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!alpha) a.z = 1.0;
            if (!beta) b.z = 0.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
            if (!bo) b.z = _Cbuild(0.0,0.0);
#endif
            zsyrk_(&uplo, &trans, &n, &k, &a.z, MAT_BUFZ(A)+oA, &ldA,
                &b.z, MAT_BUFZ(C)+oC, &ldC);
            break;

        default:
            err_invalid_id;
    }
    return;
}


/**
 * @brief General matrix-matrix multiplication (BLAS GEMM operation)
 * 
 *   blas_gemm(A, B, C, transA='N', transB='N', alpha=1.0, beta=0.0, m=None, n=None, 
 *             k=None, ldA=max(1,A.size[0]), dB=max(1,B.size[0]), ldC=max(1,C.size[0]), 
 *             offsetA=0, offsetB=0, offsetC=0)
 * 
 * @details
 * Computes the general matrix product with optional transposition:
 * - C := α·op(A)·op(B) + β·C
 * where op(X) is X, Xᵀ, or Xᴴ based on transX parameters.
 * 
 * Supports all combinations of transpositions (9 total operations).
 * If k=0, reduces to C := β·C.
 *
 * Dimensions:
 * - op(A) is m×k (or k×m if transposed)
 * - op(B) is k×n (or n×k if transposed)
 * - C is m×n
 *
 * @param[in] A       First input matrix ('d' or 'z' type)
 * @param[in] B       Second input matrix (same type as A)
 * @param[in,out] C   Input/output matrix (same type as A)
 * @param[in] transA  A transposition ('N', 'T', or 'C') (default = 'N')
 * @param[in] transB  B transposition ('N', 'T', or 'C') (default = 'N')
 * @param[in] alpha   Scalar multiplier (type must match A) (default = 1.0)
 * @param[in] beta    Scalar multiplier (type must match A) (default = 0.0)
 * @param[in] m       Rows of op(A) and C (default based on A and transA) (default = -1)
 * @param[in] n       Columns of op(B) and C (default based on B and transB) (default = -1)
 * @param[in] k       Inner dimension (default based on A/B and transpositions) (default = -1)
 * @param[in] ldA     Leading dimension of A (default = 0)
 * @param[in] ldB     Leading dimension of B (default = 0)
 * @param[in] ldC     Leading dimension of C (default = 0)
 * @param[in] offsetA Offset in A (nonnegative) (default = 0)
 * @param[in] offsetB Offset in B (nonnegative) (default = 0)
 * @param[in] offsetC Offset in C (nonnegative) (default = 0)
 *
 * @note
 * - Implements the full BLAS GEMM operation
 * - Handles all 9 combinations of transpositions
 * - Supports both real and complex matrices
 * - For complex matrices, 'C' performs conjugate transpose
 * - Default dimensions are derived from matrix sizes when negative
 *
 * @warning
 * - All matrices must have matching types ('d' or 'z')
 * - Complex alpha/beta require complex matrices
 * - Leading dimensions must satisfy:
 *   - ldA ≥ max(1, (transA=='N') ? m : k)
 *   - ldB ≥ max(1, (transB=='N') ? k : n)
 *   - ldC ≥ max(1, m)
 * - Offsets must be nonnegative
 * - Default dimensions must be compatible between A and B
 *
 * @see BLAS Level 3 GEMM documentation
 */
void blas_gemm(matrix *A, matrix *B, matrix *C, char transA, char transB, 
              void* alpha, void* beta, int m, int n, int k, int ldA, int ldB, 
              int ldC, int offsetA, int offsetB, int offsetC)
{
    number a, b;
    // int m=-1, n=-1, k=-1, ldA=0, ldB=0, ldC=0, oA=0, oB=0, oC=0;
    
    int oA, oB, oC;

    oA = offsetA;
    oB = offsetB;
    oC = offsetC;

    // Default values
    if (transA == 0) transA = 'N';
    if (transB == 0) transB = 'N';
    if (ldA == 0) ldA = 0;
    if (ldB == 0) ldB = 0;
    if (ldC == 0) ldC = 0;
    if (oA < 0) oA = 0;
    if (oB < 0) oB = 0;
    if (oC < 0) oC = 0;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(B) || MAT_ID(A) != MAT_ID(C) ||
        MAT_ID(B) != MAT_ID(C)) err_conflicting_ids;

    if (transA != 'N' && transA != 'T' && transA != 'C')
        err_char("transA", "'N', 'T', 'C'");
    if (transB != 'N' && transB != 'T' && transB != 'C')
        err_char("transB", "'N', 'T', 'C'");

    if (m < 0) m = (transA == 'N') ? A->nrows : A->ncols;
    if (n < 0) n = (transB == 'N') ? B->ncols : B->nrows;
    if (k < 0){
        k = (transA == 'N') ? A->ncols : A->nrows;
        if (k != ((transB == 'N') ? B->nrows : B->ncols)) 
            ERR_TYPE("dimensions of A and B do not match");
    }
    if (m == 0 || n == 0) return;

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (k > 0 && ldA < MAX(1, (transA == 'N') ? m : k)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (k > 0 && ldB < MAX(1, (transB == 'N') ? k : n)) err_ld("ldB");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldB");

    if (oA < 0) err_nn_int("offsetA");
    if (k > 0 && ((transA == 'N' && oA + (k-1)*ldA + m > len(A)) ||
        ((transA == 'T' || transA == 'C') &&
        oA + (m-1)*ldA + k > len(A)))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (k > 0 && ((transB == 'N' && oB + (n-1)*ldB + k > len(B)) ||
        ((transB == 'T' || transB == 'C') &&
        oB + (k-1)*ldB + n > len(B)))) err_buf_len("B");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");

    if (alpha && number_from_raw(alpha, &a, MAT_ID(A)))
        err_type("alpha");
    if (beta && number_from_raw(beta, &b, MAT_ID(A)))
        err_type("beta");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!alpha) a.d = 1.0;
            if (!beta) b.d = 0.0;
            dgemm_(&transA, &transB, &m, &n, &k, &a.d,
                MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB, &ldB, &b.d,
                MAT_BUFD(C)+oC, &ldC);
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!alpha) a.z = 1.0;
            if (!beta) b.z = 0.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
            if (!bo) b.z = _Cbuild(0.0,0.0);
#endif
            zgemm_(&transA, &transB, &m, &n, &k, &a.z,
                MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB, &ldB, &b.z,
                MAT_BUFZ(C)+oC, &ldC);
            break;

        default:
            err_invalid_id;
    }

    return;
}


/**
 * @brief General matrix-vector multiplication (BLAS GEMV operation)
 * 
 * blas_gemv(A, x, y, trans='N', alpha=1.0, beta=0.0, m=A.size[0], n=A.size[1], 
 *          ldA=max(1,A.size[0]), incx=1, incy=1, offsetA=0, offsetx=0, offsety=0)
 * 
 * @details
 * Computes one of the following matrix-vector products:
 * - y := α·A·x + β·y    (trans='N')
 * - y := α·Aᵀ·x + β·y   (trans='T')
 * - y := α·Aᴴ·x + β·y   (trans='C')
 * 
 * where A is an m×n matrix. Special cases:
 * - Returns immediately if:
 *   - n=0 and trans is 'T' or 'C'
 *   - m=0 and trans is 'N'
 * - Computes y := β·y if:
 *   - n=0, m>0 and trans is 'N'
 *   - m=0, n>0 and trans is 'T' or 'C'
 *
 * @param[in] A       Input matrix ('d' or 'z' type)
 * @param[in] x       Input vector (same type as A)
 * @param[in,out] y   Input/output vector (same type as A)
 * @param[in] trans   Operation type ('N', 'T', or 'C') (default = 'N')
 * @param[in] alpha   Scalar multiplier (type must match A) (default = 1.0)
 * @param[in] beta    Scalar multiplier (type must match A) (default = 0.0)
 * @param[in] m       Rows of matrix A (default: A.size[0]) (default = -1)
 * @param[in] n       Columns of matrix A (default: A.size[1]) (default = -1)
 * @param[in] ldA     Leading dimension of A (≥ max(1,m)) (default = 0)
 * @param[in] incx    x vector stride (nonzero) (default = 1)
 * @param[in] incy    y vector stride (nonzero) (default = 1)
 * @param[in] offsetA Matrix A offset (nonnegative) (default = 0)
 * @param[in] offsetx Vector x offset (nonnegative) (default = 0)
 * @param[in] offsety Vector y offset (nonnegative) (default = 0)
 *
 * @note
 * - Implements the BLAS GEMV operation
 * - Handles both real and complex matrices
 * - For complex matrices, 'C' performs conjugate transpose
 * - Efficiently handles edge cases with zero dimensions
 * - Uses optimized BLAS routines for computation
 *
 * @warning
 * - All matrices/vectors must have matching types ('d' or 'z')
 * - Complex alpha/beta require complex matrices
 * - ldA must satisfy ldA ≥ max(1,m)
 * - incx and incy must be nonzero
 * - Offsets must be nonnegative
 * - Behavior is undefined if dimensions are incompatible
 *
 * @see BLAS Level 2 GEMV documentation
 */
void blas_gemv(matrix *A, matrix *x, matrix *y, char trans, void* alpha, void* beta, 
            int m, int n, int ldA, int incx, int incy, int offsetA, int offsetx, int offsety)
{
    number a, b;
    // int m=-1, n=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;

    int ix, iy, oA, ox, oy;

    ix = incx;
    iy = incy;
    oA = offsetA;
    ox = offsetx;
    oy = offsety;

    // Default values
    if (trans == 0) trans = 'N';
    if (ldA == 0) ldA = 0;
    if (ix == 0) ix = 1;
    if (iy == 0) iy = 1;
    if (oA < 0) oA = 0;
    if (ox < 0) ox = 0;
    if (oy < 0) oy = 0;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N','T','C'");

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C')))
        return;

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (n > 0 && m > 0 && oA + (n-1)*ldA + m > len(A)) err_buf_len("A");

    if (ox < 0) err_nn_int("offsetx");
    if ((trans == 'N' && n > 0 && ox + (n-1)*abs(ix) + 1 > len(x)) ||
	((trans == 'T' || trans == 'C') && m > 0 &&
        ox + (m-1)*abs(ix) + 1 > len(x))) err_buf_len("x");

    if (oy < 0) err_nn_int("offsety");
    if ((trans == 'N' && oy + (m-1)*abs(iy) + 1 > len(y)) ||
        ((trans == 'T' || trans == 'C') && oy + (n-1)*abs(iy) + 1 > len(y))) 
        err_buf_len("y");

    if (alpha && number_from_raw(alpha, &a, MAT_ID(x)))
        err_type("alpha");
    if (beta && number_from_raw(beta, &b, MAT_ID(x)))
        err_type("beta");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!alpha) a.d=1.0;
            if (!beta) b.d=0.0;
            if (trans == 'N' && n == 0)
                dscal_(&m, &b.d, MAT_BUFD(y)+oy, &iy);
            else if ((trans == 'T' || trans == 'C') && m == 0)
                dscal_(&n, &b.d, MAT_BUFD(y)+oy, &iy);
            else
                dgemv_(&trans, &m, &n, &a.d, MAT_BUFD(A)+oA, &ldA,
                    MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!alpha) a.z=1.0;
            if (!beta) b.z=0.0;
#else
            if (!ao) a.z=_Cbuild(1.0,0.0);
            if (!bo) b.z=_Cbuild(0.0,0.0);
#endif
            if (trans == 'N' && n == 0)
                zscal_(&m, &b.z, MAT_BUFZ(y)+oy, &iy);
            else if ((trans == 'T' || trans == 'C') && m == 0)
                zscal_(&n, &b.z, MAT_BUFZ(y)+oy, &iy);
            else
                zgemv_(&trans, &m, &n, &a.z, MAT_BUFZ(A)+oA, &ldA,
                    MAT_BUFZ(x)+ox, &ix, &b.z, MAT_BUFZ(y)+oy, &iy);
            break;

        default:
            err_invalid_id;
    }
    return;
}