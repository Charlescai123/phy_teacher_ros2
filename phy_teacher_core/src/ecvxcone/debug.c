#include "cvxopt.h"
#include "solver.h"
#include "misc.h"
#include "blas.h"

void debug_matrix_by_project2vector(matrix *m);
void print_matrix_(matrix *m);
void print_scaling(scaling *W);

/**
 * Debugging function to print a matrix by projecting it onto a vector of ones.
 * This is useful for debugging purposes to visualize the matrix.
 *
 * @param m Pointer to the matrix to debug.
 */
void debug_matrix_by_project2vector(matrix *m) 
{
    printf("Debugging matrix by projecting onto vector of ones:\n");
    number temp_n;
    double alpha = 1.0;
    double beta = 0.0;
    temp_n.d = 1.0;
    matrix *vector_one = Matrix_New_Val(m->ncols ,1, DOUBLE, temp_n);
    matrix *res = Matrix_New_Val(m->nrows, 1, DOUBLE, temp_n);
    blas_gemv(m, vector_one, res, 'N', &alpha, &beta, -1, -1, 0, 1, 1, 0, 0, 0);

    print_matrix_(res);

    // Free the allocated matrices
    Matrix_Free(vector_one);
    Matrix_Free(res);
}

/**
 * Debugging function to print a matrix.
 * This function prints the matrix in a readable format.
 *
 * @param m Pointer to the matrix to be printed.
 */
void print_matrix_(matrix *m) 
{
    double *buf = MAT_BUFD(m);
    printf("Matrix (%dx%d):\n", m->nrows, m->ncols);
    for (int r = 0; r < m->nrows; ++r) {
        for (int c = 0; c < m->ncols; ++c) {
            printf("%.16f ", buf[c * m->nrows + r]);  // column-major
        }
        printf("\n");
    }
    printf("End of matrix********************************************\n\n");
}

/**
 * Debugging function to print the scaling structure.
 * This function prints the contents of the scaling structure, including matrices and arrays.
 *
 * @param W Pointer to the scaling structure to be printed.
 */
void print_scaling(scaling *W) 
{
    for(int i = 0; i < W->r_count; ++i) {
        printf("W[%d]:\n", i);
        for (int j = 0; j < W->r[i]->nrows; ++j) {
            for (int k = 0; k < W->r[i]->ncols; ++k) {
                printf("%.16f ", MAT_ELEMD(W->r[i], j, k));
            }
            printf("\n");
        }
        printf("End of W[%d] matrix********************************************\n\n", i);
    }
}