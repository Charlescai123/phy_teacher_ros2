#ifndef __BASE__
#define __BASE__

#include "cvxopt.h"
#include "misc.h"
#include "math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Element size array for different types
extern int E_SIZE[];

// Constants for different types
extern number One[];
extern number Zero[];

extern int intOne;

/***********************************************************************/
/****                                                               ****/
/****                   Base library prototypes                     ****/
/****                                                               ****/
/***********************************************************************/
void base_gemv(void *A, matrix *x, matrix *y, char trans, void *alpha, void *beta, 
              int m, int n, int incx, int incy, int offsetA, int offsetx, int offsety);
void base_gemm(void *A, void *B, void *C, char transA, char transB, number *alpha, 
              number *beta, bool partial);
void base_syrk(void *A, void *C, char uplo, char trans, void *alpha, void *beta, bool partial);
void* base_emul(void* A, void* B, int A_type, int B_type, int A_id, int B_id);
void* base_ediv(void* A, void* B, int A_type, int B_type, int A_id, int B_id);
void* base_pow(void* A, void* exponent, int A_type, int A_id, int exp_id);
void* base_exp(void* A, int A_type, int A_id);
void* base_sqrt(void* A, int A_type, int A_id);

#endif