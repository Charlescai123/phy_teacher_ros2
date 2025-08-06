#define BASE_MODULE

#include "base.h"
#include "cvxopt.h"

// Element size array for different types
int E_SIZE[] = {sizeof(int), sizeof(double), sizeof(double complex)};

// Constants for different types
number One[] = {{.i=1}, {.d=1.0}, {.z=1.0+0.0*I}};
number Zero[] = {{.i=0}, {.d=0.0}, {.z=0.0+0.0*I}};

int intOne = 1;

/***********************************************************************/
/****                                                               ****/
/****                      BLAS 1 prototypes                        ****/
/****                                                               ****/
/***********************************************************************/
extern int (*sp_axpy[])(number, void *, void *, int, int, int, void **) ;

extern int (*sp_gemm[])(char, char, number, void *, void *, number, void *,
    int, int, int, int, void **, int, int, int);

extern int (*sp_gemv[])(char, int, int, number, void *, int, void *, int,
    number, void *, int) ;

extern int (*sp_syrk[])(char, char, number, void *, number,
    void *, int, int, int, int, void **) ;

/*************   axpy   *************/
extern void daxpy_(int *, void *, void *, int *, void *, int *) ;
extern void zaxpy_(int *, void *, void *, int *, void *, int *) ;
static void iaxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)y)[i*(*incy)] += *((int_t *)a)*((int_t *)x)[i*(*incx)];
  }
}
void (*axpy_[])(int *, void *, void *, int *, void *, int *) = { iaxpy_, daxpy_, zaxpy_ };

/*************   scal   *************/
extern void dscal_(int *, void *, void *, int *) ;
extern void zscal_(int *, void *, void *, int *) ;
static void iscal_(int *n, void *a, void *x, int *incx) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)x)[i*(*incx)] *= *((int_t *)a);
  }
}
void (*scal_[])(int *, void *, void *, int *) = { iscal_, dscal_, zscal_ };

/***********************************************************************/
/****                                                               ****/
/****                      BLAS 2 prototypes                        ****/
/****                                                               ****/
/***********************************************************************/

/*************   gemv   *************/
extern void dgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
extern void zgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
static void (*gemv_[])(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *) = { NULL, dgemv_, zgemv_ };


/***********************************************************************/
/****                                                               ****/
/****                      BLAS 3 prototypes                        ****/
/****                                                               ****/
/***********************************************************************/

/*************   gemm   *************/
extern void dgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
extern void zgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
static void igemm_(char *transA, char *transB, int *m, int *n, int *k,
    void *alpha, void *A, int *ldA, void *B, int *ldB, void *beta,
    void *C, int *ldC) {
  int i, j, l;
  for (j=0; j<*n; j++) {
    for (i=0; i<*m; i++) {
      ((int_t *)C)[i+j*(*m)] = 0;
      for (l=0; l<*k; l++)
        ((int_t *)C)[i+j*(*m)]+=((int_t *)A)[i+l*(*m)]*((int_t *)B)[j*(*k)+l];
    }
  }
}
void (*gemm_[])(char *, char *, int *, int *, int *, void *, void *, int *,
    void *, int *, void *, void *, int *) = { igemm_, dgemm_, zgemm_ };

/*************   symv   *************/
extern void dsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
extern void zsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
void (*symv[])(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *) = { NULL, dsymv_, zsymv_ };

/*************   syrk   *************/
extern void dsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
extern void zsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
void (*syrk_[])(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *) = { NULL, dsyrk_, zsyrk_ };

/* val_id: 0 = matrix, number = 1
 */
static int
convert_inum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
      *(int_t *)dest = MAT_BUFI(val)[offset]; return 0;
    case DOUBLE:
      *(int_t *)dest = (int_t)round(MAT_BUFD(val)[offset]); return 0;
    default: 
      fprintf(stderr, "Error: cannot cast argument as integer\n");
      return -1;
    }
  } else if (val_id==1) { /* normal number */
      *(int_t *)dest = ((int_t *)val)[offset]; return 0;
      return 0;
  }
    else { /* unsupported type */
        ERR("Error: unsupported type for integer conversion");
        return -1;
    }
}

static int
convert_dnum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* matrix */
    switch (MAT_ID(val)) {
    case INT:    *(double *)dest = MAT_BUFI(val)[offset]; return 0;
    case DOUBLE: *(double *)dest = MAT_BUFD(val)[offset]; return 0;
    default: 
      fprintf(stderr, "Error: cannot cast argument as double\n");
      return -1;
    }
  } else if (val_id == 1) { /* normal number */
    *(double *)dest = ((double *)val)[offset]; return 0;
  } else { /* unsupported type */
    ERR("Error: unsupported type for double conversion");
    return -1;
  }
}

static int
convert_znum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFI(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild((double)MAT_BUFI(val)[offset],0.0); return 0;
#endif
    case DOUBLE:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFD(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild(MAT_BUFD(val)[offset],0.0); return 0;
#endif
    case COMPLEX:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFZ(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = MAT_BUFZ(val)[offset]; return 0;
#endif
    default: 
      fprintf(stderr, "Error: unsupported matrix type for complex conversion\n");
      return -1;
    }
  } else if (val_id == 1) { /* normal number */
    *(double complex *)dest = ((double complex *)val)[offset]; return 0;
  } else { /* unsupported type */
    ERR("Error: unsupported type for complex conversion");
    return -1;
  }
}

int (*convert_num[])(void *, void *, int, int_t) = {
    convert_inum, convert_dnum, convert_znum };

static void mtx_iabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((int_t *)dest)[i] = labs(((int_t *)src)[i]);
}

static void mtx_dabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((double *)dest)[i] = fabs(((double *)src)[i]);
}

static void mtx_zabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
#ifndef _MSC_VER
    ((double *)dest)[i] = cabs(((double complex *)src)[i]);
#else
    ((double *)dest)[i] = cabs(((_Dcomplex *)src)[i]);
#endif
}

void (*mtx_abs[])(void *, void *, int) = { mtx_iabs, mtx_dabs, mtx_zabs };


static void write_inum(void *dest, int i, void *src, int j) {
  ((int_t *)dest)[i]  = ((int_t *)src)[j];
}

static void write_dnum(void *dest, int i, void *src, int j) {
  ((double *)dest)[i]  = ((double *)src)[j];
}

static void write_znum(void *dest, int i, void *src, int j) {
#ifndef _MSC_VER
  ((double complex *)dest)[i]  = ((double complex *)src)[j];
#else
  ((_Dcomplex *)dest)[i]  = ((_Dcomplex *)src)[j];
#endif
}

void (*write_num[])(void *, int, void *, int) = {
    write_inum, write_dnum, write_znum };

/**
 * @brief General matrix-vector product for sparse and dense matrices (GEMV operation)
 * 
 * @details
 * Computes the matrix-vector product with optional transposition:
 * - y := alpha*A*x + beta*y    if trans = 'N'
 * - y := alpha*Aᵀ*x + beta*y   if trans = 'T'
 * - y := alpha*Aᴴ*x + beta*y   if trans = 'C'
 * 
 * Matrix A is m×n dimensional. Special cases:
 * - Returns immediately if:
 *   - n=0 and trans is 'T' or 'C'
 *   - m=0 and trans is 'N'
 * - Computes y := beta*y if:
 *   - n=0, m>0 and trans is 'N'
 *   - m=0, n>0 and trans is 'T' or 'C'
 * 
 * @param[in] A         Input matrix ('d' for real, 'z' for complex; sparse/dense)
 * @param[in] x         Input vector (must match type of A)
 * @param[in,out] y     Input/output vector (must match type of A)
 * @param[in] trans     Operation type ('N': none, 'T': transpose, 
 *                                      'C': conjugate transpose)
 * @param[in] alpha     Scalar coefficient (type must match A; 
 *                                          complex only if A is complex)
 * @param[in] beta      Scalar coefficient (type must match A; 
 *                                          complex only if A is complex)
 * @param[in] m         Row dimension of A (if negative, uses default)
 * @param[in] n         Column dimension of A (if negative, uses default)
 * @param[in] incx      x vector stride (nonzero)
 * @param[in] incy      y vector stride (nonzero)
 * @param[in] offsetA   Matrix A offset (nonnegative)
 * @param[in] offsetx   Vector x offset (nonnegative)
 * @param[in] offsety   Vector y offset (nonnegative)
 * 
 * @note
 * - Implements the BLAS GEMV operation for both sparse and dense matrices
 * - For sparse matrices: m ≤ A.size[0] - (offsetA % A.size[0])
 * - Handles all transposition cases ('N', 'T', 'C')
 * - Properly handles edge cases with zero dimensions
 * 
 * @warning
 * - A, x, and y must be of same type ('d' or 'z')
 * - Complex alpha/beta require complex matrices
 * - incx and incy must be nonzero
 * - All offsets must be nonnegative
 * - For sparse A, dimension constraint must be satisfied
 * 
 * @see BLAS Level 2 Specification for GEMV
 */
void base_gemv(void *A, matrix *x, matrix *y, char trans, void *alpha, void *beta, 
              int m, int n, int incx, int incy, int offsetA, int offsetx, int offsety)
{
    number a, b;
    
    int id;
    // Input validation
    if (!is_matrix(A) && !is_spmatrix(A)) 
        ERR("A must be a dense or sparse matrix");
    
    if (!is_matrix(x)) err_mtrx("x");
    if (!is_matrix(y)) err_mtrx("y");
    
    if (MAT_ID(x) == INT || MAT_ID(y) == INT || X_ID(A) == INT) {
        ERR_TYPE("invalid matrix types");
    }
    
    if (X_ID(A) != MAT_ID(x) || X_ID(A) != MAT_ID(y)) {
        err_conflicting_ids;
    }
    
    if (trans != 'N' && trans != 'T' && trans != 'C') {
        err_char("trans", "'N', 'T', 'C'");
    }
    
    if (incx == 0) err_nz_int("incx");
    if (incy == 0) err_nz_int("incy");
    
    id = MAT_ID(x);
    if (m < 0) m = X_NROWS(A);
    if (n < 0) n = X_NCOLS(A);
    if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C'))) 
        return;
    
    if (offsetA < 0) err_nn_int("offsetA");
    if (n > 0 && m > 0 && offsetA + (n-1)*MAX(1,X_NROWS(A)) + m >
        X_NROWS(A)*X_NCOLS(A)) 
        err_buf_len("A");
    
    if (offsetx < 0) err_nn_int("offsetx");
    if ((trans == 'N' && n > 0 && offsetx + (n-1)*abs(incx) + 1 > MAT_LGT(x)) ||
        ((trans == 'T' || trans == 'C') && m > 0 &&
            offsetx + (m-1)*abs(incx) + 1 > MAT_LGT(x))) err_buf_len("x");
    
    if (offsety < 0) err_nn_int("offsety");
    if ((trans == 'N' && offsety + (m-1)*abs(incy) + 1 > MAT_LGT(y)) ||
        ((trans == 'T' || trans == 'C') &&
            offsety + (n-1)*abs(incy) + 1 > MAT_LGT(y))) err_buf_len("y");
    
    if (alpha && convert_num[MAT_ID(x)](&a, alpha, 1, 0)) err_type("alpha");
    if (beta && convert_num[MAT_ID(x)](&b, beta, 1, 0)) err_type("beta");
    
    if (is_matrix(A)) {
        int ldA = MAX(1,X_NROWS(A));
        if (trans == 'N' && n == 0) {
            scal_[id](&m, (beta ? &b : &Zero[id]), 
                     (unsigned char*)MAT_BUF(y)+offsety*E_SIZE[id], &incy);
        }
        else if ((trans == 'T' || trans == 'C') && m == 0) {
            scal_[id](&n, (beta ? &b : &Zero[id]), 
                     (unsigned char*)MAT_BUF(y)+offsety*E_SIZE[id], &incy);
        }
        else {
            gemv_[id](&trans, &m, &n, (alpha ? &a : &One[id]),
                     (unsigned char*)MAT_BUF(A) + offsetA*E_SIZE[id], &ldA,
                     (unsigned char*)MAT_BUF(x) + offsetx*E_SIZE[id], &incx, 
                     (beta ? &b : &Zero[id]),
                     (unsigned char*)MAT_BUF(y) + offsety*E_SIZE[id], &incy);
        }
    } else {
        if (sp_gemv[id](trans, m, n, (alpha ? a : One[id]), ((spmatrix *)A)->obj,
                       offsetA, (unsigned char*)MAT_BUF(x) + offsetx*E_SIZE[id], incx, 
                       (beta ? b : Zero[id]),
                       (unsigned char*)MAT_BUF(y) + offsety*E_SIZE[id], incy)) {
            err_no_memory;
        }
    }
}


/**
 * @brief General matrix-matrix product (GEMM operation)
 * 
 * @details
 * Computes the matrix-matrix product with optional transposition:
 * - C := alpha*A*B + beta*C     if transA = 'N' and transB = 'N'
 * - C := alpha*Aᵀ*B + beta*C    if transA = 'T' and transB = 'N'
 * - C := alpha*Aᴴ*B + beta*C    if transA = 'C' and transB = 'N'
 * - C := alpha*A*Bᵀ + beta*C    if transA = 'N' and transB = 'T'
 * - C := alpha*Aᵀ*Bᵀ + beta*C   if transA = 'T' and transB = 'T'
 * - C := alpha*Aᴴ*Bᵀ + beta*C   if transA = 'C' and transB = 'T'
 * - C := alpha*A*Bᴴ + beta*C    if transA = 'N' and transB = 'C'
 * - C := alpha*Aᵀ*Bᴴ + beta*C   if transA = 'T' and transB = 'C'
 * - C := alpha*Aᴴ*Bᴴ + beta*C   if transA = 'C' and transB = 'C'
 * 
 * If k=0, reduces to C := beta*C
 * 
 * @param[in] A         Input matrix ('d' for real, 'z' for complex)
 * @param[in] B         Input matrix (must match type of A)
 * @param[in,out] C     Input/output matrix (must match type of A)
 * @param[in] transA    Transposition of A ('N': none, 'T': transpose, 
 *                                      'C': conjugate transpose)
 * @param[in] transB    Transposition of B ('N': none, 'T': transpose, 
 *                                      'C': conjugate transpose)
 * @param[in] alpha     Scalar coefficient (int, float, or complex; 
 *                                      complex only if A is complex)
 * @param[in] beta      Scalar coefficient (int, float, or complex; 
 *                                      complex only if A is complex)
 * @param[in] partial   Boolean flag for sparse C:
 *                      - If true and C is sparse, only updates nonzero elements
 *                      - If false, updates according to full operation
 * 
 * @note
 * - Implements the standard BLAS GEMM operation
 * - For complex matrices, performs complex arithmetic
 * - When partial=true with sparse C, ignores sparsity patterns of A and B
 * 
 * @warning
 * - Matrices A, B, and C must be of same type ('d' or 'z')
 * - Complex alpha/beta require complex matrices
 * - Proper matrix dimensions must be maintained for multiplication
 * - C must be preallocated with correct dimensions
 * 
 * @see BLAS Level 3 Specification for GEMM
 */
void base_gemm(void *A, void *B, void *C, char transA, char transB, 
                       number *alpha, number *beta, bool partial)
{
    number a, b;
    int m, n, k;
    char transA_, transB_;
    
    // Validate input matrices
    if (!(Matrix_Check(A) || SpMatrix_Check(A))) {
        ERR_TYPE("A must be a matrix or sparse matrix");
    }
    if (!(Matrix_Check(B) || SpMatrix_Check(B))) {
        ERR_TYPE("B must be a matrix or sparse matrix");
    }
    if (!(Matrix_Check(C) || SpMatrix_Check(C))) {
        ERR_TYPE("C must be a matrix or sparse matrix");
    }
    
    // Check type consistency
    if (X_ID(A) != X_ID(B) || X_ID(A) != X_ID(C) || X_ID(B) != X_ID(C)) {
        err_conflicting_ids;
    }
    
    // Validate transpose parameters
    if (transA != 'N' && transA != 'T' && transA != 'C') {
        err_char("transA", "'N', 'T', 'C'");
    }
    if (transB != 'N' && transB != 'T' && transB != 'C') {
        err_char("transB", "'N', 'T', 'C'");
    }
    
    // Calculate dimensions
    m = (transA == 'N') ? X_NROWS(A) : X_NCOLS(A);
    n = (transB == 'N') ? X_NCOLS(B) : X_NROWS(B);
    k = (transA == 'N') ? X_NCOLS(A) : X_NROWS(A);
    if (k != ((transB == 'N') ? X_NROWS(B) : X_NCOLS(B))) {
        ERR_TYPE("dimensions of A and B do not match");
    }
    
    // Early return for empty matrices
    if (m == 0 || n == 0) return;
    
    // Convert alpha and beta parameters
    if (alpha && convert_num[X_ID(A)](&a, alpha, 1, 0)) err_type("alpha");
    if (beta && convert_num[X_ID(A)](&b, beta, 1, 0)) err_type("beta");
    
    // Set character versions for compatibility
    transA_ = transA;
    transB_ = transB;
    
    int id = X_ID(A);
    
    // Handle dense matrix case
    if (Matrix_Check(A) && Matrix_Check(B) && Matrix_Check(C)) {
        int ldA = MAX(1, MAT_NROWS(A));
        int ldB = MAX(1, MAT_NROWS(B));
        int ldC = MAX(1, MAT_NROWS(C));
        
        if (id == INT) err_invalid_id;
        
        // Call appropriate BLAS GEMM function
        gemm_[id](&transA_, &transB_, &m, &n, &k, 
                 (alpha ? &a : &One[id]),
                 MAT_BUF(A), &ldA, 
                 MAT_BUF(B), &ldB, 
                 (beta ? &b : &Zero[id]),
                 MAT_BUF(C), &ldC);
    } 
    else {
        // Handle sparse matrix case
        void *z = NULL;
        
        if (sp_gemm[id](transA_, transB_, (alpha ? a : One[id]),
                        Matrix_Check(A) ? MAT_BUF(A) : ((spmatrix *)A)->obj,
                        Matrix_Check(B) ? MAT_BUF(B) : ((spmatrix *)B)->obj,
                        (beta ? b : Zero[id]),
                        Matrix_Check(C) ? MAT_BUF(C) : ((spmatrix *)C)->obj,
                        SpMatrix_Check(A), SpMatrix_Check(B), SpMatrix_Check(C),
                        partial, &z, m, n, k)) {
            err_no_memory;
        }
        
        // Update sparse matrix structure if needed
        if (z) {
            free_ccs(((spmatrix *)C)->obj);
            ((spmatrix *)C)->obj = z;
        }
    }

}


/**
 * @brief Rank-k update of symmetric sparse or dense matrix (BLAS SYRK operation)
 * 
 * base_syrk(A, C, uplo='L', trans='N', alpha=1.0, beta=0.0, partial=False)
 * 
 * @details
 * Computes a symmetric rank-k update:
 * - C := α·A·Aᵀ + β·C   (trans='N')
 * - C := α·Aᵀ·A + β·C   (trans='T')
 * 
 * Supports both dense and sparse matrices. For sparse C with partial=True,
 * only updates nonzero elements of C regardless of A's sparsity pattern.
 *
 * @param[in] A       Input matrix ('d' or 'z' type, sparse or dense)
 * @param[in,out] C   Input/output symmetric matrix (same type/storage as A) 
 * @param[in] uplo    Triangle selection ('L' for lower, 'U' for upper) (default = 'L')
 * @param[in] trans   Transposition of A ('N' or 'T') (default = 'N')
 * @param[in] alpha   Scalar multiplier (type must match A) (default = 1.0)
 * @param[in] beta    Scalar multiplier (type must match A) (default = 0.0)
 * @param[in] partial Sparse optimization flag: (default = false)
 *                    - true: only update nonzero elements of sparse C
 *                    - false: full update
 *
 * @note
 * - Implements extended BLAS SYRK operation with sparse support
 * - Handles both real and complex matrices
 * - For complex matrices, result is Hermitian when using complex data
 * - Partial update only affects sparse C matrices
 * - Only updates the specified triangle (upper/lower) of C
 * - For sparse matrices, maintains original sparsity pattern
 *
 * @warning
 * - C must be symmetric/Hermitian
 * - Matrices must have matching types ('d' or 'z')
 * - Complex alpha/beta require complex matrices
 * - Storage formats (sparse/dense) must be compatible
 * - Partial=true only valid when C is sparse
 * - Undefined behavior if dimensions are incompatible
 *
 * @see BLAS Level 3 SYRK documentation
 * @see Sparse BLAS extensions
 */
void base_syrk(void *A, void *C, char uplo, char trans, void *alpha, void *beta, bool partial)
{
    number a, b;

    if (uplo == 0) uplo = 'L';
    if (trans == 0) trans = 'N';

    if (!(Matrix_Check(A) || SpMatrix_Check(A)))
        ERR_TYPE("A must be a dense or sparse matrix");
    if (!(Matrix_Check(C) || SpMatrix_Check(C)))
        ERR_TYPE("C must be a dense or sparse matrix");

    int id = X_ID(A);
    if (id == INT) ERR_TYPE("invalid matrix types");
    if (id != X_ID(C)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (id == DOUBLE && trans != 'N' && trans != 'T' &&
      trans != 'C') err_char("trans", "'N', 'T', 'C'");
    if (id == COMPLEX && trans != 'N' && trans != 'T')
        err_char("trans", "'N', 'T'");

//   if (partial && !PyBool_Check(partial)) err_bool("partial");

    int n = (trans == 'N') ? X_NROWS(A) : X_NCOLS(A);
    int k = (trans == 'N') ? X_NCOLS(A) : X_NROWS(A);
    if (n == 0) return;

    if (alpha && convert_num[id](&a, alpha, 1, 0)) err_type("alpha");
    if (beta && convert_num[id](&b, beta, 1, 0)) err_type("beta");

    if (Matrix_Check(A) && Matrix_Check(C)) {

        int ldA = MAX(1,MAT_NROWS(A));
        int ldC = MAX(1,MAT_NROWS(C));

    syrk_[id](&uplo, &trans, &n, &k, (alpha ? &a : &One[id]),
            MAT_BUF(A), &ldA, (beta ? &b : &Zero[id]), MAT_BUF(C), &ldC);

    } else {

    void *z = NULL;

    if (sp_syrk[id](uplo, trans,
        (alpha ? a : One[id]),
        Matrix_Check(A) ? MAT_BUF(A) : ((spmatrix *)A)->obj,
            (beta ? b : Zero[id]),
            Matrix_Check(C) ? MAT_BUF(C) : ((spmatrix *)C)->obj,
                SpMatrix_Check(A), SpMatrix_Check(C),
                partial ? 1 : 0,
                    (trans == 'N' ? X_NCOLS(A) : X_NROWS(A)), &z))
        err_no_memory;
    if (z) {
      free_ccs( ((spmatrix *)C)->obj );
      ((spmatrix *)C)->obj = z;
    }
  }

  return;
}


/**
 * @brief Element-wise multiplication of two matrices or numbers
 * 
 * @details
 * Multiplies two matrices element-wise or multiplies a matrix by a number.
 * If both inputs are matrices, they must have the same dimensions.
 * If one input is a number, it multiplies each element of the other matrix.
 * 
 * @param A First input (matrix/spmatrix or number)
 * @param B Second input (matrix/spmatrix or number)
 * @param A_type Type of A (0 - matrix/spmatrix, 1 - number)
 * @param B_type Type of B (0 - matrix/spmatrix, 1 - number)
 * @param A_id Type ID of A (INT, DOUBLE, COMPLEX)
 * @param B_id Type ID of B (INT, DOUBLE, COMPLEX)
 * 
 * @return A new matrix with the result of the element-wise multiplication
 */
void* base_emul(void* A, void* B, int A_type, int B_type, int A_id, int B_id)
{
    if(A_type != 0 && A_type != 1) {
        ERR_TYPE("A must be a matrix/spmatrix or a number");
    }
    if(B_type != 0 && B_type != 1) {
        ERR_TYPE("B must be a matrix/spmatrix or a number");
    }
    // if (!(X_Matrix_Check(A) || is_number(A)) ||
    //     !(X_Matrix_Check(B) || is_number(B)))
    //     ERR_TYPE("arguments must be either matrices or numbers");

    int a_is_number = (A_type == 1) || (Matrix_Check(A) && MAT_LGT(A) == 1);
    int b_is_number = (B_type == 1) || (Matrix_Check(B) && MAT_LGT(B) == 1);

    int ida, idb;
    if (A_type == 1) {
        switch (A_id) {
        case INT: ida = INT; break;
        case DOUBLE: ida = DOUBLE; break;
        case COMPLEX: ida = COMPLEX; break;
        default: ERR_TYPE("invalid ID for A");
        }
    } else { ida = X_ID(A); }
    if (B_type == 1) {
        switch (B_id) {
        case INT: idb = INT; break;
        case DOUBLE: idb = DOUBLE; break;
        case COMPLEX: idb = COMPLEX; break;
        default: ERR_TYPE("invalid ID for B");
        }
    } else { idb = X_ID(B); }

  int id  = MAX( ida, idb );

  number a, b;
  if (a_is_number) convert_num[id](&a, A, A_type, 0);
  if (b_is_number) convert_num[id](&b, B, B_type, 0);

  if (a_is_number && b_is_number &&
      (!X_Matrix_Check(A) && !X_Matrix_Check(B))) {
    number* c = malloc(sizeof(number));
    if (!c) {
        ERR("memory allocation failed");
        return NULL;
    }
    if (id == INT)
    c->i = a.i * b.i;
    else if (id == DOUBLE)
    c->d = a.d * b.d;
    else {
#ifndef _MSC_VER
        c->z = a.z*b.z;
#else
        c->z = _Cmulcc(a.z,b.z);
#endif
    }
    return c;
  }

  if (!(a_is_number || b_is_number)) {
    if (X_NROWS(A) != X_NROWS(B) || X_NCOLS(A) != X_NCOLS(B))
      ERR_TYPE("incompatible dimensions");
  }

  int_t m = ( !a_is_number ? X_NROWS(A) : (!b_is_number ? X_NROWS(B) : 1));
  int_t n = ( !a_is_number ? X_NCOLS(A) : (!b_is_number ? X_NCOLS(B) : 1));

  if ((Matrix_Check(A) || a_is_number) && (Matrix_Check(B) || b_is_number)) {

    matrix *ret = Matrix_New(m, n, id);
    if (!ret) return NULL;

    int_t i;
    for (i=0; i<m*n; i++) {
      if (!a_is_number) convert_num[id](&a, A, 0, i);
      if (!b_is_number) convert_num[id](&b, B, 0, i);

      if (id == INT)
        MAT_BUFI(ret)[i] = a.i*b.i;
      else if (id == DOUBLE)
        MAT_BUFD(ret)[i] = a.d*b.d;
      else
#ifndef _MSC_VER
        MAT_BUFZ(ret)[i] = a.z*b.z;
#else
        MAT_BUFZ(ret)[i] = _Cmulcc(a.z,b.z);
#endif
    }
    return ret;
  }
  else if (SpMatrix_Check(A) && !SpMatrix_Check(B)) {

    spmatrix *ret = SpMatrix_NewFromSpMatrix((spmatrix *)A, id);
    if (!ret) return NULL;

    int_t j, k;
    for (j=0; j<SP_NCOLS(A); j++) {
      for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++) {
        if (!b_is_number) convert_num[id](&b, B, 0, j*m + SP_ROW(A)[k]);

        if (id == DOUBLE)
          SP_VALD(ret)[k] *= b.d;
        else
#ifndef _MSC_VER
          SP_VALZ(ret)[k] *= b.z;
#else
          SP_VALZ(ret)[k] = _Cmulcc(SP_VALZ(ret)[k],b.z);
#endif
      }
    }
    return ret;
  }
  else if (SpMatrix_Check(B) && !SpMatrix_Check(A)) {

    spmatrix *ret = SpMatrix_NewFromSpMatrix((spmatrix *)B, id);
    if (!ret) return NULL;

    int_t j, k;
    for (j=0; j<SP_NCOLS(B); j++) {
      for (k=SP_COL(B)[j]; k<SP_COL(B)[j+1]; k++) {
        if (!a_is_number) convert_num[id](&a, A, 0, j*m + SP_ROW(B)[k]);

        if (id == DOUBLE)
          SP_VALD(ret)[k] *= a.d;
        else
#ifndef _MSC_VER
          SP_VALZ(ret)[k] *= a.z;
#else
          SP_VALZ(ret)[k] = _Cmulcc(SP_VALZ(ret)[k],a.z);
#endif
      }
    }
    return ret;
  }

  else {

    spmatrix *ret = SpMatrix_New(m, n, 0, id);
    if (!ret) return NULL;

    int_t j, ka = 0, kb = 0, kret = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) ka++;
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) kb++;
        else {
          SP_COL(ret)[j+1]++; 
          ka++; 
          kb++;
        }
      }

      ka = SP_COL(A)[j+1];
      kb = SP_COL(B)[j+1];
    }

    for (j=0; j<n; j++) SP_COL(ret)[j+1] += SP_COL(ret)[j];

    int_t *newrow = malloc( sizeof(int_t)*SP_COL(ret)[n] );
    double *newval = malloc( E_SIZE[id]*SP_COL(ret)[n] );
    if (!newrow || !newval) {
      free(newrow); free(newval);
      err_no_memory;
    }
    free( ret->obj->rowind );
    free( ret->obj->values );
    ret->obj->rowind = newrow;
    ret->obj->values = newval;

    ka = 0; kb = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) {
          ka++;
        }
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) {
          kb++;
        }
        else {
          SP_ROW(ret)[kret] = SP_ROW(A)[ka];
          if (id == DOUBLE)
            SP_VALD(ret)[kret] = SP_VALD(A)[ka]*SP_VALD(B)[kb];
          else
#ifndef _MSC_VER
            SP_VALZ(ret)[kret] =
                (X_ID(A) == DOUBLE ? SP_VALD(A)[ka] : SP_VALZ(A)[ka])*
                (X_ID(B) == DOUBLE ? SP_VALD(B)[kb] : SP_VALZ(B)[kb]);
#else
	  SP_VALZ(ret)[kret] = _Cmulcc(
		(X_ID(A) == DOUBLE ? _Cbuild(SP_VALD(A)[ka],0.0) : SP_VALZ(A)[ka]),
		(X_ID(B) == DOUBLE ? _Cbuild(SP_VALD(B)[kb],0.0) : SP_VALZ(B)[kb]));
#endif
          kret++; ka++; kb++;
        }
      }
      ka = SP_COL(A)[j+1];
      kb = SP_COL(B)[j+1];
    }
    return ret;
  }
}



/**
 * @brief Element-wise division of two matrices or numbers
 * 
 * @details
 * Divides two matrices element-wise or divides a matrix by a number.
 * If both inputs are matrices, they must have the same dimensions.
 * If one input is a number, it divides each element of the other matrix.
 * 
 * @param A First input (matrix/spmatrix or number)
 * @param B Second input (matrix/spmatrix or number)
 * @param A_type Type of A (0 - matrix/spmatrix, 1 - number)
 * @param B_type Type of B (0 - matrix/spmatrix, 1 - number)
 * @param A_id Type ID of A (INT, DOUBLE, COMPLEX)
 * @param B_id Type ID of B (INT, DOUBLE, COMPLEX)
 * 
 * @return A new matrix with the result of the element-wise division
 */
void* base_ediv(void* A, void* B, int A_type, int B_type, int A_id, int B_id)
{
  void *ret;
  if(A_type != 0 && A_type != 1) {
    ERR_TYPE("A must be a matrix/spmatrix or a number");
  }
  if(B_type != 0 && B_type != 1) {
    ERR_TYPE("B must be a matrix/spmatrix or a number");
  }

  if (SpMatrix_Check(B))
    ERR_TYPE("elementwise division with sparse matrix\n");

  int a_is_number = (A_type == 1) || (Matrix_Check(A) && MAT_LGT(A) == 1);
  int b_is_number = (B_type == 1) || (Matrix_Check(B) && MAT_LGT(B) == 1);

  int ida, idb;
  if (A_type == 1) {
    switch (A_id) {
    case INT: ida = INT; break;
    case DOUBLE: ida = DOUBLE; break;
    case COMPLEX: ida = COMPLEX; break;
    default: ERR_TYPE("invalid ID for A");
    }
  } else { ida = X_ID(A); }
  if (B_type == 1) {
    switch (B_id) {
    case INT: idb = INT; break;
    case DOUBLE: idb = DOUBLE; break;
    case COMPLEX: idb = COMPLEX; break;
    default: ERR_TYPE("invalid ID for B");
    }
  } else { idb = X_ID(B); }

  int id  = MAX( ida, idb );

  number a, b;
  if (a_is_number) convert_num[id](&a, A, A_type, 0);
  if (b_is_number) convert_num[id](&b, B, B_type, 0);

  if ((a_is_number && b_is_number) &&
      (!X_Matrix_Check(A) && !Matrix_Check(B))) {
    number* c = malloc(sizeof(number));
    if (!c) {
        ERR("memory allocation failed");
        return NULL;
    }
    if (id == INT) {
      if (b.i == 0) err_division_by_zero;
      c->i = a.i/b.i;
      return c;
    }
    else if (id == DOUBLE) {
      if (b.d == 0.0) err_division_by_zero;
      c->d = a.d/b.d;
      return c;
    }
    else {
#ifndef _MSC_VER
      if (b.z == 0.0) err_division_by_zero;
#else
      if (creal(b.z) == 0.0 && cimag(b.z) == 0.0) PY_ERR(PyExc_ArithmeticError, "division by zero");
#endif
#ifndef _MSC_VER
      c->z = a.z/b.z;
#else
      c->z = _Cmulcc(a.z, _Cmulcr(conj(b.z),1.0/norm(b.z)));
#endif
      return c;
    }
  }

  if (!(a_is_number || b_is_number)) {
    if (X_NROWS(A) != MAT_NROWS(B) || X_NCOLS(A) != MAT_NCOLS(B))
      ERR_TYPE("incompatible dimensions");
  }

  int m = ( !a_is_number ? X_NROWS(A) : (!b_is_number ? X_NROWS(B) : 1));
  int n = ( !a_is_number ? X_NCOLS(A) : (!b_is_number ? X_NCOLS(B) : 1));

  if ((Matrix_Check(A) || a_is_number) && (Matrix_Check(B) || b_is_number)) {
    if (!(ret = (matrix *)Matrix_New(m, n, id)))
      return NULL;

    int i;
    for (i=0; i<m*n; i++) {
      if (!a_is_number) convert_num[id](&a, A, 0, i);
      if (!b_is_number) convert_num[id](&b, B, 0, i);

      if (id == INT) {
        if (b.i == 0) goto divzero;
        MAT_BUFI(ret)[i] = a.i/b.i;
      }
      else if (id == DOUBLE) {
        if (b.d == 0) goto divzero;
        MAT_BUFD(ret)[i] = a.d/b.d;
      }
      else {
#ifndef _MSC_VER
        if (b.z == 0) goto divzero;
        MAT_BUFZ(ret)[i] = a.z/b.z;
#else
        if (creal(b.z) == 0 && cimag(b.z)== 0) goto divzero;
        MAT_BUFZ(ret)[i] = _Cmulcc(a.z,_Cmulcr(conj(b.z),1.0/norm(b.z)));
#endif
      }
    }
    return ret;
  }
  else { // (SpMatrix_Check(A) && !SpMatrix_Check(B)) {

    if (!(ret = (spmatrix *)SpMatrix_NewFromSpMatrix((spmatrix *)A, id)))
      return NULL;

    int j, k;
    for (j=0; j<SP_NCOLS(A); j++) {
      for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++) {
        if (!b_is_number) convert_num[id](&b, B, 0, j*m + SP_ROW(A)[k]);

        if (id == DOUBLE) {
          if (b.d == 0.0) goto divzero;
          SP_VALD(ret)[k] /= b.d;
        }
        else {
#ifndef _MSC_VER
         if (b.z == 0) goto divzero;
         SP_VALZ(ret)[k] /= b.z;
#else
         if (creal(b.z) == 0 && cimag(b.z)== 0) goto divzero;
         SP_VALZ(ret)[k] = _Cmulcc(SP_VALZ(ret)[k],_Cmulcr(conj(b.z),1.0/norm(b.z)));
#endif
        }
      }
    }
    return ret;
  }

  divzero:
  err_division_by_zero;
}

/**
 * @brief Core power function for matrix and scalar exponentiation
 * 
 * @details
 * Computes element-wise power operation A^exponent for:
 * - Dense matrices with numeric exponents
 * - Handles both real and complex-valued matrices and exponents
 * - For real values, checks for domain errors (negative base with fractional exponent)
 * - For complex values, checks for zero base with negative/imaginary exponent
 *
 * @param A           Input matrix to raise to power (must be dense matrix)
 * @param exponent    Scalar exponent (int, float, or complex)
 * @param A_type      The type of the input (not used here, but could be used
 *                    for type checking). (0 - matrix/spmatrix, 1 - number)
 * @param A_id        The ID of the input (not used here, but could be used
 *                    for type checking). (0 - INT, 1 - DOUBLE, 2 - COMPLEX)
 * @param exp_id      Type identifier for exponent (0 - INT, 1 - DOUBLE, 2 - COMPLEX)
 *
 * @return Pointer to new matrix containing result, or NULL on error
 *         - Returned matrix type matches highest precision of inputs
 *         - Must be freed with matrix_free()
 */

void* base_pow(void* A, void* exponent, int A_type, int A_id, int exp_id)
{
  if (exp_id != INT && exp_id != DOUBLE && exp_id != COMPLEX) {
    ERR_TYPE("exponent must be a number");
  }

  if (!A) { // NULL
    ERR("error: NULL input"); return NULL;
  }

  number val;
  convert_num[MAX(DOUBLE, exp_id)](&val, exponent, 1, 0);  // Get value of exponent

  int id = MAX(DOUBLE, MAX(A_id, exp_id));

  if (A_type == 1) {
    number* a = (number*) A;
    number* res = malloc(sizeof(number));
    if (!res) {
      fprintf(stderr, "base_pow: memory allocation failed.\n");
      return NULL;
    }

    if (id == INT || id == DOUBLE) {
      double base = (A_id == INT) ? (double)a->i : a->d;
      double exponent_val = (exp_id == INT) ? (double)(((number*)exponent)->i) : val.d;

      if (base == 0.0 && exponent_val < 0.0) {
          fprintf(stderr, "domain error: pow(0, negative)\n");
          free(res);
          return NULL;
      }
        res->d = pow(base, exponent_val);
    }
    else if (id == COMPLEX) {
      double complex base = (A_id == COMPLEX) ? a->z :
                            (A_id == DOUBLE)  ? a->d + 0.0*I :
                                                a->i + 0.0*I;
      double complex exponent_val = (exp_id == COMPLEX) ? val.z :
                                    (exp_id == DOUBLE)  ? val.d + 0.0*I :
                                                          val.i + 0.0*I;
      if (creal(base) == 0.0 && cimag(base) == 0.0 &&
          (cimag(exponent_val) != 0.0 || creal(exponent_val) < 0.0)) {
          fprintf(stderr, "domain error: cpow(0, complex/negative)\n");
          free(res);
          return NULL;
      }
      res->z = cpow(base, exponent_val);
    }
    return res;
  } else if (Matrix_Check(A)) {
    matrix *Y = Matrix_NewFromMatrix((matrix *)A, id);
    if (!Y) return NULL;

    int i;
    for (i=0; i<MAT_LGT(Y); i++) {
      if (id == DOUBLE) {
        if ((MAT_BUFD(Y)[i] == 0.0 && val.d < 0.0) ||
            (MAT_BUFD(Y)[i] < 0.0 && val.d < 1.0 && val.d > 0.0)) {
          ERR("domain error");
        }

        MAT_BUFD(Y)[i] = pow(MAT_BUFD(Y)[i], val.d);
      } else {
  #ifndef _MSC_VER
        if (MAT_BUFZ(Y)[i] == 0.0 && (cimag(val.z) != 0.0 || creal(val.z)<0.0)) {
  #else
        if ((creal(MAT_BUFZ(Y)[i]) == 0.0 && cimag(MAT_BUFZ(Y)[i]) == 0.0) && (cimag(val.z) != 0.0 || creal(val.z)<0.0)) {
  #endif
          ERR("domain error");
        }
        MAT_BUFZ(Y)[i] = cpow(MAT_BUFZ(Y)[i], val.z);
      }
    }

    return (void *)Y;
  }

  fprintf(stderr, "base_pow: unsupported A_type=%d\n", A_type);
  return NULL;
}

/**
 * @brief Core exponential function for numeric types and matrices
 * 
 * @details
 * Computes the exponential function for:
 * - Scalar numbers (int, float, complex)
 * - Dense matrices (element-wise operation)
 * - Handles both real and complex numeric types
 *
 * @param A The input number or matrix.
 * @param A_type The type of the input (not used here, but could be used
 *               for type checking). (0 - matrix/spmatrix, 1 - number)
 * @param A_id The ID of the input (not used here, but could be used
 *             for type checking). (0 - INT, 1 - DOUBLE, 2 - COMPLEX)
 *
 * @return A new number or matrix containing the exponential of the input.
 */

void* base_exp(void* A, int A_type, int A_id)
{
  if (A_type == 1) {  // number
    number* a = (number*) A;
    number* res = malloc(sizeof(number));
    if (!res) err_no_memory;

    if (A_id == INT || A_id == DOUBLE) {
      res->d = exp(A_id == INT ? (double)a->i : a->d);
    } else if (A_id == COMPLEX) {
      res->z = cexp(a->z);
    } else {
        ERR("unsupported scalar type");
    }
      return res;
  }
  else if (Matrix_Check(A)) {  // matrix
    matrix *ret = Matrix_New(MAT_NROWS(A),MAT_NCOLS(A),
        (MAT_ID(A) == COMPLEX ? COMPLEX : DOUBLE));
    if (!ret) return NULL;

    int i;
    if (MAT_ID(ret) == DOUBLE)
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFD(ret)[i] = exp(MAT_ID(A) == DOUBLE ? MAT_BUFD(A)[i] :
        MAT_BUFI(A)[i]);
    else
      for (i=0; i<MAT_LGT(ret); i++)
        MAT_BUFZ(ret)[i] = cexp(MAT_BUFZ(A)[i]);

    return (void *)ret;
  }
  else ERR_TYPE("argument must a be a number or dense matrix");
}


/**
 * \brief Computes the square root of a number or a matrix.
 * @param A The input number or matrix.
 * @param A_type The type of the input (not used here, but could be used
 *               for type checking). (0 - matrix/spmatrix, 1 - number)
 * @param A_id The ID of the input (not used here, but could be used
 *             for type checking). (0 - INT, 1 - DOUBLE, 2 - COMPLEX)
 * @return A new number or matrix containing the square root of the input.
 */
void* base_sqrt(void* A, int A_type, int A_id)
{
  if (!A) { // NULL
    ERR("error: NULL input"); return NULL;
  }
  if (A_type == 1) { // number
    number* ret = (number*)A;

    if (A_id == DOUBLE) {    // DOUBLE
      if (ret->d < 0.0) {
        ERR("domain error: sqrt of negative real\n");
        return NULL;
      }
      ret->d = sqrt(ret->d);
      return ret;
    } else if (A_id == INT) {    // INT
      if (ret->i < 0) {
        ERR("domain error: sqrt of negative int\n");
        return NULL;
      }
      ret->i = sqrt(ret->i);
      return ret;
    } else if (A_id == COMPLEX) {
      ret->z = csqrt(ret->z);
      return ret;
    } else {
      fprintf(stderr, "invalid number id\n");
      return NULL;
    }
  }
  else if (Matrix_Check(A) && (MAT_ID(A) == INT || MAT_ID(A) == DOUBLE)) {
    if (MAT_LGT(A) == 0)
      return (matrix *)Matrix_New(MAT_NROWS(A),MAT_NCOLS(A),DOUBLE);

    double val = (MAT_ID(A) == INT ? MAT_BUFI(A)[0] : MAT_BUFD(A)[0]);

    int i;
    for (i=1; i<MAT_LGT(A); i++) {
      if (MAT_ID(A) == INT)
        val = MIN(val,(MAT_BUFI(A)[i]));
      else
        val = MIN(val,(MAT_BUFD(A)[i]));
    }

    if (val >= 0.0) {
      matrix *ret = Matrix_New(MAT_NROWS(A), MAT_NCOLS(A), DOUBLE);
      if (!ret) return NULL;
      for (i=0; i<MAT_LGT(A); i++)
        MAT_BUFD(ret)[i] = sqrt((MAT_ID(A)== INT ?
            MAT_BUFI(A)[i] : MAT_BUFD(A)[i]));
      return ret;
    }
    else ERR("domain error");
  }
  else if (Matrix_Check(A) && MAT_ID(A) == COMPLEX) {
    matrix *ret = Matrix_New(MAT_NROWS(A), MAT_NCOLS(A), COMPLEX);
    if (!ret) return NULL;

    int i;
    for (i=0; i<MAT_LGT(A); i++)
      MAT_BUFZ(ret)[i] = csqrt(MAT_BUFZ(A)[i]);

    return ret;
  }
  else ERR_TYPE("argument must a be a number or dense matrix");
}