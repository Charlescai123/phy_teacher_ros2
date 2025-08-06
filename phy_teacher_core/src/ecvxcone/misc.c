#include "base.h"
#include "blas.h"
#include "lapack.h"
#include "solver.h"
#include "cvxopt.h"
#include "misc.h"
#include "float.h"
#include <unistd.h>

/**
 * @brief Compute jnrm2 for a second-order cone vector
 *
 * Computes sqrt(x0 - a) * sqrt(x0 + a), where:
 *  - x0 = x[offset]
 *  - a = Euclidean norm of x[offset+1 ... offset+n-1]
 *
 * @param x      input matrix
 * @param n      length of the vector in the second-order cone
 * @param offset start index of this SOC block in x (default = 0)
 * @return       double result of jnrm2, or NAN if domain error
 */
double misc_jnrm2(matrix* x, int n, int offset) {
    if (offset < 0) offset = 0;
    if (n < 0) n = len(x);
    double a = blas_nrm2(x, n-1, 1, offset+1);
    double x0 = MAT_BUFD(x)[offset];
    double t1 = x0 - a;
    double t2 = x0 + a;
    return sqrt(t1) * sqrt(t2);
}


/**
 * @brief Applies Nesterov-Todd scaling or its inverse to a vector
 * 
 * misc_scale(x, W, trans = 'N', inverse = 'N')
 * 
 * @details
 * Computes one of four scaling operations based on parameters:
 * - x := W*x       (trans='N', inverse='N')
 * - x := Wᵀ*x      (trans='T', inverse='N')
 * - x := W⁻¹*x     (trans='N', inverse='I')
 * - x := W⁻ᵀ*x     (trans='T', inverse='I')
 *
 * The scaling matrix W is defined by a dictionary containing:
 * - dnl, dnli:  Positive vector and its inverse (optional, nonlinear solver only)
 * - d, di:      Positive vector and its inverse
 * - v:          List of 2nd-order cone vectors with unit hyperbolic norms
 * - beta:       List of positive scaling factors
 * - r:          List of square matrices
 * - rti:        List of inverse-transpose matrices of r[k]
 *
 * @param[in,out] x      Dense 'd' matrix to scale (input/output)
 * @param[in] W          Scaling dictionary containing NT parameters
 * @param[in] trans      Transpose flag ('N' or 'T')
 * @param[in] inverse    Inverse flag ('N' or 'I')
 *
 * @note
 * - Implements Nesterov-Todd scaling for conic optimization
 * - Handles both regular and inverse transformations
 * - Supports transposed operations
 * - dnl/dnli entries are optional and nonlinear-solver specific
 *
 * @warning
 * - x must be a dense 'd' matrix
 * - W must contain all required fields (except optional dnl/dnli)
 * - trans and inverse must be valid flags ('N', 'T', or 'I')
 * - All vectors/matrices must have compatible dimensions
 */
void misc_scale(matrix *x, scaling *W, char trans, char inverse) 
{
    matrix *d, *vk, *rk;

    // Set default character values
    if (trans == 0) trans = 'N';
    if (inverse == 0) inverse = 'N';

    int m, n, xr, xc, ind = 0, int0 = 0, int1 = 1, i, k, inc, len, ld, 
        maxn, N;
    double b, dbl0 = 0.0, dbl1 = 1.0, dblm1 = -1.0, dbl2 = 2.0, dbl5 = 0.5,
        *wrk;

    xr = x->nrows;
    xc = x->ncols;

    /*
     * Scaling for nonlinear component xk is xk := dnl .* xk; inverse is
     * xk ./ dnl = dnli .* xk, where dnl = W['dnl'], dnli = W['dnli'].
     */

    if ((d = (inverse == 'N') ? (matrix *) W->dnl : (matrix *) W->dnli)){
        m = len(d);
        for (i = 0; i < xc; ++i)
            dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(d), &int1,
                MAT_BUFD(x) + i*xr, &int1);
        ind += m;
    }

    /*
     * Scaling for 'l' component xk is xk := d .* xk; inverse scaling is
     * xk ./ d = di .* xk, where d = W['d'], di = W['di'].
     */

    if (!(d = (inverse == 'N') ? (matrix *) W->d : (matrix *) W->di)){
        ERR("missing item W['d'] or W['di']");
        return;
    }
    m = len(d);
    for (i = 0; i < xc; i++)
        dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(d), &int1, MAT_BUFD(x)
            + i*xr + ind, &int1);
    ind += m;

    /*
     * Scaling for 'q' component is
     *
     *     xk := beta * (2*v*v' - J) * xk
     *         = beta * (2*v*(xk'*v)' - J*xk)
     *
     * where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
     *
     * Inverse scaling is
     *
     *     xk := 1/beta * (2*J*v*v'*J - J) * xk
     *         = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk).
     */

    N = W->v_count;
    if (!(wrk = (double *) calloc(xc, sizeof(double)))) err_no_memory;
    for (k = 0; k < N; ++k){
        vk = (matrix *) W->v[k];
        m = vk->nrows;
        if (inverse == 'I')
            dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);
        ld = MAX(xr, 1);
        dgemv_("T", &m, &xc, &dbl1, MAT_BUFD(x) + ind, &ld, MAT_BUFD(vk), 
            &int1, &dbl0, wrk, &int1);
        dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);
        dger_(&m, &xc, &dbl2, MAT_BUFD(vk), &int1, wrk, &int1,
            MAT_BUFD(x) + ind, &ld);
        if (inverse == 'I')
            dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);

        b = W->beta[k];
        if (inverse == 'I') b = 1.0 / b;
        for (i = 0; i < xc; i++)
            dscal_(&m, &b, MAT_BUFD(x) + ind + i*xr, &int1);
        ind += m;
    }
    free(wrk);


    /*
     * Scaling for 's' component xk is
     *
     *     xk := vec( r' * mat(xk) * r )  if trans = 'N'
     *     xk := vec( r * mat(xk) * r' )  if trans = 'T'.
     *
     * r is kth element of W['r'].
     *
     * Inverse scaling is
     *
     *     xk := vec( rti * mat(xk) * rti' )  if trans = 'N'
     *     xk := vec( rti' * mat(xk) * rti )  if trans = 'T'.
     *
     * rti is kth element of W['rti'].
     */

    N = W->r_count;
    for (k = 0, maxn = 0; k < N; ++k){
        rk = (inverse == 'N') ? W->r[k] : W->rti[k];
        maxn = MAX(maxn, rk->nrows);
    }
    if (!(wrk = (double *) calloc(maxn*maxn, sizeof(double)))) err_no_memory;
    for (k = 0; k < N; k++){
        rk = (inverse == 'N') ? W->r[k] : W->rti[k];
        n = rk->nrows;
        for (i = 0; i < xc; i++){

            /* scale diagonal of rk by 0.5 */
            inc = n + 1;
            dscal_(&n, &dbl5, MAT_BUFD(x) + ind + i*xr, &inc);

            /* wrk = r*tril(x) if inverse is 'N' and trans is 'T' or
             *                 inverse is 'I' and trans is 'N'
             * wrk = tril(x)*r otherwise. */
            len = n*n;
            dcopy_(&len, MAT_BUFD(rk), &int1, wrk, &int1);
            ld = MAX(1, n);
            dtrmm_( (( inverse == 'N' && trans == 'T') || ( inverse == 'I'
                && trans == 'N')) ? "R" : "L", "L", "N", "N", &n, &n,
                &dbl1, MAT_BUFD(x) + ind + i*xr, &ld, wrk, &ld);

            /* x := (r*wrk' + wrk*r') if inverse is 'N' and trans is 'T'
             *                        or inverse is 'I' and trans is 'N'
             * x := (r'*wrk + wrk'*r) otherwise. */
            dsyr2k_("L", ((inverse == 'N' && trans == 'T') ||
                (inverse == 'I' && trans == 'N')) ? "N" : "T", &n, &n,
                &dbl1, MAT_BUFD(rk), &ld, wrk, &ld, &dbl0, MAT_BUFD(x) +
                ind + i*xr, &ld);
        }
        ind += n*n;
    }
    free(wrk);
}

/**
 * @brief Multiplies with square root of the Hessian of the logarithmic barrier
 * 
 * misc_scale2(lmbda, x, dims, mnl = 0, inverse = 'N')
 * 
 * @details
 * Computes the product of x with the square root of the Hessian matrix H:
 * - x := H(λ^{1/2}) * x   (inverse = 'N')
 * - x := H(λ^{-1/2}) * x  (inverse = 'I')
 * 
 * where H is the Hessian of the logarithmic barrier function and λ are the
 * scaling parameters.
 *
 * @param[in] lmbda    Scaling parameters vector
 * @param[in,out] x    Input/output vector to be transformed
 * @param[in] dims     Dictionary containing cone dimensions:
 *                     - "l": linear dimension
 *                     - "q": list of quadratic cone dimensions
 *                     - "s": list of semidefinite cone dimensions
 * @param[in] mnl      Number of nonlinear variables (default = 0)
 * @param[in] inverse  Transformation mode ('N' for direct, 'I' for inverse)
 *                     (default = 'N')
 * 
 * @return None (result stored in x)
 *
 * @note
 * - Implements Nesterov-Todd scaling for conic optimization
 * - Handles both direct and inverse square root transformations
 * - Respects the cone structure defined in dims
 * - Uses efficient BLAS operations for the transformations
 *
 * @warning
 * - lmbda and x must have matching cone structure
 * - dims must contain valid dimension entries
 * - inverse must be either 'N' or 'I'
 * - All vectors must be properly allocated
 * - Undefined behavior if dimensions mismatch
 *
 * @see The Theory of Self-Dual Cones and Cone Programming by Nesterov and Todd
 */
void misc_scale2(matrix *lmbda, matrix *x, DIMs *dims, int mnl, char inverse)
{
    // Set default character values
    if (inverse == 0) inverse = 'N';

    double a, lx, x0, b, *c = NULL, *sql = NULL;
    int m = 0, mk, i, j, len, int0 = 0, int1 = 1, maxn = 0, ind2;

    /*
     * For nonlinear and 'l' blocks:
     *
     *     xk := xk ./ l  (invers is 'N')
     *     xk := xk .* l  (invers is 'I')
     *
     * where l is the first mnl + dims['l'] components of lmbda.
     */

    m += dims->l;
    if (inverse == 'N')
        dtbsv_("L", "N", "N", &m, &int0, MAT_BUFD(lmbda), &int1,
             MAT_BUFD(x), &int1);
    else
        dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(lmbda), &int1,
             MAT_BUFD(x), &int1);

    /*
     * For 'q' blocks, if inverse is 'N',
     *
     *     xk := 1/a * [ l'*J*xk;
     *         xk[1:] - (xk[0] + l'*J*xk) / (l[0] + 1) * l[1:] ].
     *
     *  If inverse is 'I',
     *
     *     xk := a * [ l'*xk;
     *         xk[1:] + (xk[0] + l'*xk) / (l[0] + 1) * l[1:] ].
     *
     * a = sqrt(lambda_k' * J * lambda_k), l = lambda_k / a.
     */

    for (i = 0; i < dims->q_size; i++){
        mk = dims->q[i];
        len = mk - 1;
        a = dnrm2_(&len, MAT_BUFD(lmbda) + m + 1, &int1);
        a = sqrt(MAT_BUFD(lmbda)[m] + a) * sqrt(MAT_BUFD(lmbda)[m] - a);
        if (inverse == 'N')
            lx = ( MAT_BUFD(lmbda)[m] * MAT_BUFD(x)[m] -
                ddot_(&len, MAT_BUFD(lmbda) + m + 1, &int1, MAT_BUFD(x) + m
                    + 1, &int1) ) / a;
        else
            lx = ddot_(&mk, MAT_BUFD(lmbda) + m, &int1, MAT_BUFD(x) + m,
                &int1) / a;
        x0 = MAT_BUFD(x)[m];
        MAT_BUFD(x)[m] = lx;
        b = (x0 + lx) / (MAT_BUFD(lmbda)[m]/a + 1.0) / a;
        if (inverse == 'N')  b *= -1.0;
        daxpy_(&len, &b, MAT_BUFD(lmbda) + m + 1, &int1,
            MAT_BUFD(x) + m + 1, &int1);
        if (inverse == 'N')  a = 1.0 / a;
        dscal_(&mk, &a, MAT_BUFD(x) + m, &int1);
        m += mk;
    }


    /*
     *  For the 's' blocks, if inverse is 'N',
     *
     *      xk := vec( diag(l)^{-1/2} * mat(xk) * diag(k)^{-1/2}).
     *
     *  If inverse is 'I',
     *
     *     xk := vec( diag(l)^{1/2} * mat(xk) * diag(k)^{1/2}).
     *
     * where l is kth block of lambda.
     *
     * We scale upper and lower triangular part of mat(xk) because the
     * inverse operation will be applied to nonsymmetric matrices.
     */

    for (i = 0; i < dims->s_size; i++){
        maxn = MAX(maxn, dims->s[i]);
    }
    if (!(c = (double *) calloc(maxn, sizeof(double))) ||
        !(sql = (double *) calloc(maxn, sizeof(double)))){
        free(c); free(sql);
        err_no_memory;
    }
    ind2 = m;
    for (i = 0; i < dims->s_size; i++){
        mk = dims->s[i];
        for (j = 0; j < mk; j++)
            sql[j] = sqrt(MAT_BUFD(lmbda)[ind2 + j]);
        for (j = 0; j < mk; j++){
            dcopy_(&mk, sql, &int1, c, &int1);
            b = sqrt(MAT_BUFD(lmbda)[ind2 + j]);
            dscal_(&mk, &b, c, &int1);
            if (inverse == 'N')
                dtbsv_("L", "N", "N", &mk, &int0, c, &int1, MAT_BUFD(x) +
                    m + j*mk, &int1);
            else
                dtbmv_("L", "N", "N", &mk, &int0, c, &int1, MAT_BUFD(x) +
                    m + j*mk, &int1);
        }
        m += mk*mk;
        ind2 += mk;
    }
    free(c); free(sql);
}


// static char doc_pack[] =
//     "Copy x to y using packed storage.\n\n"
//     "pack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0)\n\n"
//     "The vector x is an element of S, with the 's' components stored in\n"
//     "unpacked storage.  On return, x is copied to y with the 's' \n"
//     "components stored in packed storage and the off-diagonal entries \n"
//     "scaled by sqrt(2).";

// static PyObject* pack(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x, *y;
//     PyObject *O, *Ok, *dims;
//     double a;
//     int i, k, nlq = 0, ox = 0, oy = 0, np, iu, ip, int1 = 1, len, n;
//     char *kwlist[] = {"x", "y", "dims", "mnl", "offsetx", "offsety", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iii", kwlist, &x,
//         &y, &dims, &nlq, &ox, &oy)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     nlq += (int) PyLong_AsLong(O);
// #else
//     nlq += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         nlq += (int) PyLong_AsLong(Ok);
// #else
//         nlq += (int) PyInt_AsLong(Ok);
// #endif
//     }
//     dcopy_(&nlq, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, np = 0, iu = ox + nlq, ip = oy + nlq; i < (int)
//         PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         n = (int) PyLong_AsLong(Ok);
// #else
//         n = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < n; k++){
//             len = n-k;
//             dcopy_(&len, MAT_BUFD(x) + iu + k*(n+1), &int1,  MAT_BUFD(y) +
//                 ip, &int1);
//             MAT_BUFD(y)[ip] /= sqrt(2.0);
//             ip += len;
//         }
//         np += n*(n+1)/2;
//         iu += n*n;
//     }

//     a = sqrt(2.0);
//     dscal_(&np, &a, MAT_BUFD(y) + oy + nlq, &int1);

//     return Py_BuildValue("");
// }


// static char doc_pack2[] =
//     "In-place version of pack().\n\n"
//     "pack2(x, dims, mnl = 0)\n\n"
//     "In-place version of pack(), which also accepts matrix arguments x.\n"
//     "The columns of x are elements of S, with the 's' components stored\n"
//     "in unpacked storage.  On return, the 's' components are stored in\n"
//     "packed storage and the off-diagonal entries are scaled by sqrt(2).";

// static PyObject* pack2(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x;
//     PyObject *O, *Ok, *dims;
//     double a = sqrt(2.0), *wrk;
//     int i, j, k, nlq = 0, iu, ip, len, n, maxn, xr, xc;
//     char *kwlist[] = {"x", "dims", "mnl", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
//         &dims, &nlq)) return NULL;

//     xr = x->nrows;
//     xc = x->ncols;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     nlq += (int) PyLong_AsLong(O);
// #else
//     nlq += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         nlq += (int) PyLong_AsLong(Ok);
// #else
//         nlq += (int) PyInt_AsLong(Ok);
// #endif
//     }

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, maxn = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
// #else
//         maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
// #endif
//     }
//     if (!maxn) return Py_BuildValue("");
//     if (!(wrk = (double *) calloc(maxn * xc, sizeof(double))))
//         return PyErr_NoMemory();

//     for (i = 0, iu = nlq, ip = nlq; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         n = (int) PyLong_AsLong(Ok);
// #else
//         n = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < n; k++){
//             len = n-k;
//             dlacpy_(" ", &len, &xc, MAT_BUFD(x) + iu + k*(n+1), &xr, wrk, 
//                 &maxn);
//             for (j = 1; j < len; j++)
//                 dscal_(&xc, &a, wrk + j, &maxn);
//             dlacpy_(" ", &len, &xc, wrk, &maxn, MAT_BUFD(x) + ip, &xr);
//             ip += len;
//         }
//         iu += n*n;
//     }

//     free(wrk);
//     return Py_BuildValue("");
// }


// static char doc_unpack[] =
//     "Unpacks x into y.\n\n"
//     "unpack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0)\n\n"
//     "The vector x is an element of S, with the 's' components stored in\n"
//     "unpacked storage and off-diagonal entries scaled by sqrt(2).\n"
//     "On return, x is copied to y with the 's' components stored in\n"
//     "unpacked storage.";

// static PyObject* unpack(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x, *y;
//     PyObject *O, *Ok, *dims;
//     double a = 1.0 / sqrt(2.0);
//     int m = 0, ox = 0, oy = 0, int1 = 1, iu, ip, len, i, k, n;
//     char *kwlist[] = {"x", "y", "dims", "mnl", "offsetx", "offsety", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iii", kwlist, &x,
//         &y, &dims, &m, &ox, &oy)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     m += (int) PyLong_AsLong(O);
// #else
//     m += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         m += (int) PyLong_AsLong(Ok);
// #else
//         m += (int) PyInt_AsLong(Ok);
// #endif
//     }
//     dcopy_(&m, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, ip = ox + m, iu = oy + m; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         n = (int) PyLong_AsLong(Ok);
// #else
//         n = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < n; k++){
//             len = n-k;
//             dcopy_(&len, MAT_BUFD(x) + ip, &int1, MAT_BUFD(y) + iu +
//                 k*(n+1), &int1);
//             ip += len;
//             len -= 1;
//             dscal_(&len, &a, MAT_BUFD(y) + iu + k*(n+1) + 1, &int1);
//         }
//         iu += n*n;
//     }

//     return Py_BuildValue("");
// }


/**
 * Converts lower triangular matrix to symmetric.
 * 
 * Fills in the upper triangular part of the symmetric matrix stored
 * in x[offset : offset+n*n] using 'L' storage.
 * 
 * @param x        Pointer to the matrix structure
 * @param n        Size of the matrix (n x n)
 * @param offset   Starting offset in the matrix buffer (default = 0)
 */

void misc_symm(matrix *x, int n, int offset)
{
    int ox = offset, k, len, int1 = 1;
    if (n > 1) for (k = 0; k < n; k++){
        len = n-k-1;
        dcopy_(&len, MAT_BUFD(x) + ox + k*(n+1) + 1, &int1, MAT_BUFD(x) +
            ox + (k+1)*(n+1)-1, &n);
    }
    return;
}


/**
 * @brief Computes the symmetric product x := (y o x) in cone space
 * 
 * misc_sprod(x, y, dims, mnl = 0, diag = 'N')
 * 
 * @details
 * Computes the symmetric product operation between vectors x and y that respects
 * the cone structure defined by dims. The operation handles three cases:
 * 
 * 1. Nonlinear and linear blocks:
 *    yk o xk = yk .* xk (element-wise product)
 * 
 * 2. Quadratic cone blocks:
 *              [ l0   l1'  ]
 *    yk o xk = [           ] * xk
 *              [ l1   l0*I ]
 *    where yk = (l0, l1)
 * 
 * 3. Semidefinite cone blocks:
 *    yk o xk = 0.5 * (Yk*mat(xk) + mat(xk)*Yk)
 *    where Yk = mat(yk) if diag='N' or Yk = diag(yk) if diag='D'
 *
 * @param[in,out] x    Input/output vector in cone space
 * @param[in] y        Input vector in cone space
 * @param[in] dims     Dictionary containing cone dimensions:
 *                     - "l": linear dimension
 *                     - "q": list of quadratic cone dimensions
 *                     - "s": list of semidefinite cone dimensions
 * @param[in] mnl      Number of nonlinear variables (default = 0)
 * @param[in] diag     Diagonal flag ('N' for full, 'D' for diagonal storage) 
 *                     (default = 'N')
 *
 * @return None (result stored in x)
 *
 * @note
 * - Uses BLAS/LAPACK routines (dtbmv, ddot, dscal, daxpy, dsyr2k) for efficiency
 * - For 's' blocks with diag='D', only diagonal elements are stored/processed
 * - Handles Python 2/3 compatibility internally
 * - Allocates temporary workspace for semidefinite blocks
 *
 * @warning
 * - x and y must have matching cone structure
 * - dims must contain valid dimension entries
 * - diag must be either 'N' or 'D'
 * - Caller must ensure x has sufficient storage for result
 * - Memory allocation failures will raise Python memory error
 */

void misc_sprod(matrix *x, matrix *y, DIMs *dims, int mnl, char diag)
{
    int i, j, k, mk, length, maxn, ind = 0, ind2, int0 = 0, int1 = 1, ld;
    double a, *A = NULL, dbl2 = 0.5, dbl0 = 0.0;

    if (mnl < 0) mnl = 0;

    ind = mnl;
    if(diag == 0) diag = 'N';

    /*
     * For nonlinear and 'l' blocks:
     *
     *     yk o xk = yk .* xk
     */

    ind += dims->l;
    dtbmv_("L", "N", "N", &ind, &int0, MAT_BUFD(y), &int1, MAT_BUFD(x),
        &int1);

    /*
     * For 'q' blocks:
     *
     *                [ l0   l1'  ]
     *     yk o xk =  [           ] * xk
     *                [ l1   l0*I ]
     *
     * where yk = (l0, l1).
     */

    for (i = 0; i < dims->q_size; ++i){
        mk = dims->q[i];
        a = ddot_(&mk, MAT_BUFD(y) + ind, &int1, MAT_BUFD(x) + ind, &int1);
        length = mk - 1;
        dscal_(&length, MAT_BUFD(y) + ind, MAT_BUFD(x) + ind + 1, &int1);
        daxpy_(&length, MAT_BUFD(x) + ind, MAT_BUFD(y) + ind + 1, &int1,
            MAT_BUFD(x) + ind + 1, &int1);
        MAT_BUFD(x)[ind] = a;
        ind += mk;
    }


    /*
     * For the 's' blocks:
     *
     *    yk o sk = .5 * ( Yk * mat(xk) + mat(xk) * Yk )
     *
     * where Yk = mat(yk) if diag is 'N' and Yk = diag(yk) if diag is 'D'.
     */

    for (i = 0, maxn = 0; i < dims->s_size; ++i)
        maxn = MAX(maxn, dims->s[i]);
    
    if (diag == 'N'){
        if (!(A = (double *) calloc(maxn * maxn, sizeof(double)))) err_no_memory;
        for (i = 0; i < dims->s_size; ind += mk*mk, ++i){
            mk = dims->s[i];
            length = mk * mk;
            dcopy_(&length, MAT_BUFD(x) + ind, &int1, A, &int1);

            if (mk > 1) for (k = 0; k < mk; k++){
                length = mk - k - 1;
                dcopy_(&length, A + k*(mk+1) + 1, &int1, A + (k+1)*(mk+1)-1,
                    &mk);
                dcopy_(&length, MAT_BUFD(y) + ind + k*(mk+1) + 1, &int1,
                    MAT_BUFD(y) + ind + (k+1)*(mk+1)-1, &mk);
            }

            ld = MAX(1, mk);
            dsyr2k_("L", "N", &mk, &mk, &dbl2, A, &ld, MAT_BUFD(y) + ind,
                &ld, &dbl0, MAT_BUFD(x) + ind, &ld);
        }
    }
    else {
        if (!(A = (double *) calloc(maxn, sizeof(double)))) err_no_memory;
        for (i = 0, ind2 = ind; i < dims->s_size; ind += mk*mk,
            ind2 += mk, i++){
            mk = dims->s[i];
            for (k = 0; k < mk; k++){
                length = mk - k;
                dcopy_(&length, MAT_BUFD(y) + ind2 + k, &int1, A, &int1);
                for (j = 0; j < length; j++) A[j] += MAT_BUFD(y)[ind2 + k];
                dscal_(&length, &dbl2, A, &int1);
                dtbmv_("L", "N", "N", &length, &int0, A, &int1, MAT_BUFD(x) +
                    ind + k * (mk+1), &int1);
            }
        }
    }

    free(A);
}


// static char doc_sinv[] =
//     "The inverse of the product x := (y o x) when the 's' components of \n"
//     "y are diagonal.\n\n"
//     "sinv(x, y, dims, mnl = 0)";

// static PyObject* sinv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x, *y;
//     PyObject *dims, *O, *Ok;
//     int i, j, k, mk, len, maxn, ind = 0, ind2, int0 = 0, int1 = 1;
//     double a, c, d, alpha, *A = NULL, dbl2 = 0.5;
//     char *kwlist[] = {"x", "y", "dims", "mnl", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|i", kwlist, &x, &y,
//         &dims, &ind)) return NULL;


//     /*
//      * For nonlinear and 'l' blocks:
//      *
//      *     yk o\ xk = yk .\ xk
//      */

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     ind += (int) PyLong_AsLong(O);
// #else
//     ind += (int) PyInt_AsLong(O);
// #endif
//     dtbsv_("L", "N", "N", &ind, &int0, MAT_BUFD(y), &int1, MAT_BUFD(x),
//         &int1);


//     /*
//      * For 'q' blocks:
//      *
//      *                        [  l0   -l1'               ]
//      *     yk o\ xk = 1/a^2 * [                          ] * xk
//      *                        [ -l1    (a*I + l1*l1')/l0 ]
//      *
//      * where yk = (l0, l1) and a = l0^2 - l1'*l1.
//      */

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         mk = (int) PyLong_AsLong(Ok);
// #else
//         mk = (int) PyInt_AsLong(Ok);
// #endif
//         len = mk - 1;
//         a = dnrm2_(&len, MAT_BUFD(y) + ind + 1, &int1);
//         a = (MAT_BUFD(y)[ind] + a) * (MAT_BUFD(y)[ind] - a);
//         c = MAT_BUFD(x)[ind];
//         d = ddot_(&len, MAT_BUFD(x) + ind + 1, &int1,
//             MAT_BUFD(y) + ind + 1, &int1);
//         MAT_BUFD(x)[ind] = c * MAT_BUFD(y)[ind] - d;
//         alpha = a / MAT_BUFD(y)[ind];
//         dscal_(&len, &alpha, MAT_BUFD(x) + ind + 1, &int1);
//         alpha = d / MAT_BUFD(y)[ind] - c;
//         daxpy_(&len, &alpha, MAT_BUFD(y) + ind + 1, &int1, MAT_BUFD(x) +
//             ind + 1, &int1);
//         alpha = 1.0 / a;
//         dscal_(&mk, &alpha, MAT_BUFD(x) + ind, &int1);
//         ind += mk;
//     }


//     /*
//      * For the 's' blocks:
//      *
//      *    yk o\ sk = xk ./ gamma
//      *
//      * where  gammaij = .5 * (yk_i + yk_j).
//      */

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, maxn = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
// #else
//         maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
// #endif
//     }
//     if (!(A = (double *) calloc(maxn, sizeof(double))))
//         return PyErr_NoMemory();
//     for (i = 0, ind2 = ind; i < (int) PyList_Size(O); ind += mk*mk,
//         ind2 += mk, i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         mk = (int) PyLong_AsLong(Ok);
// #else
//         mk = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < mk; k++){
//             len = mk - k;
//             dcopy_(&len, MAT_BUFD(y) + ind2 + k, &int1, A, &int1);
//             for (j = 0; j < len; j++) A[j] += MAT_BUFD(y)[ind2 + k];
//             dscal_(&len, &dbl2, A, &int1);
//             dtbsv_("L", "N", "N", &len, &int0, A, &int1, MAT_BUFD(x) + ind
//                 + k * (mk+1), &int1);
//         }
//     }

//     free(A);
//     return Py_BuildValue("");
// }



// static char doc_trisc[] =
//     "Sets the upper triangular part of the 's' components of x equal to\n"
//     "zero and scales the strictly lower triangular part\n\n"
//     "trisc(x, dims, offset = 0)";

// static PyObject* trisc(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x;
//     double dbl0 = 0.0, dbl2 = 2.0;
//     int ox = 0, i, k, nk, len, int1 = 1;
//     PyObject *dims, *O, *Ok;
//     char *kwlist[] = {"x", "dims", "offset", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
//         &dims, &ox)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     ox += (int) PyLong_AsLong(O);
// #else
//     ox += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         ox += (int) PyLong_AsLong(Ok);
// #else
//         ox += (int) PyInt_AsLong(Ok);
// #endif
//     }

//     O = PyDict_GetItemString(dims, "s");
//     for (k = 0; k < (int) PyList_Size(O); k++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) k);
// #if PY_MAJOR_VERSION >= 3
//         nk = (int) PyLong_AsLong(Ok);
// #else
//         nk = (int) PyInt_AsLong(Ok);
// #endif
//         for (i = 1; i < nk; i++){
//             len = nk - i;
//             dscal_(&len, &dbl0, MAT_BUFD(x) + ox + i*(nk+1) - 1, &nk);
//             dscal_(&len, &dbl2, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
//         }
//         ox += nk*nk;
//     }

//     return Py_BuildValue("");
// }


// static char doc_triusc[] =
//     "Scales the strictly lower triangular part of the 's' components of\n"
//     "x by 0.5.\n\n"
//     "triusc(x, dims, offset = 0)";

// static PyObject* triusc(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x;
//     double dbl5 = 0.5;
//     int ox = 0, i, k, nk, len, int1 = 1;
//     PyObject *dims, *O, *Ok;
//     char *kwlist[] = {"x", "dims", "offset", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
//         &dims, &ox)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     ox += (int) PyLong_AsLong(O);
// #else
//     ox += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         ox += (int) PyLong_AsLong(Ok);
// #else
//         ox += (int) PyInt_AsLong(Ok);
// #endif
//     }

//     O = PyDict_GetItemString(dims, "s");
//     for (k = 0; k < (int) PyList_Size(O); k++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) k);
// #if PY_MAJOR_VERSION >= 3
//         nk = (int) PyLong_AsLong(Ok);
// #else
//         nk = (int) PyInt_AsLong(Ok);
// #endif
//         for (i = 1; i < nk; i++){
//             len = nk - i;
//             dscal_(&len, &dbl5, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
//         }
//         ox += nk*nk;
//     }

//     return Py_BuildValue("");
// }

/**
 * @brief Computes the inner product of two vectors in the composite cone space S
 * 
 * @details
 * Calculates the specialized inner product between vectors x and y that respects
 * the cone structure defined by dims. The computation handles:
 * - Linear components (including nonlinear variables mnl)
 * - Quadratic cone components
 * - Semidefinite cone components (with proper scaling of off-diagonal elements)
 *
 * For semidefinite blocks, the inner product is computed as:
 * ⟨X,Y⟩ = Σ_{i,j} X_{i,j}Y_{i,j} with off-diagonal terms multiplied by 2
 *
 * @param[in] x     First input vector in cone space
 * @param[in] y     Second input vector in cone space (must match x structure)
 * @param[in] dims  Cone dimensions structure:
 *                  - l: linear dimension
 *                  - q: array of quadratic cone dimensions
 *                  - q_size: number of quadratic cones
 *                  - s: array of semidefinite cone dimensions
 *                  - s_size: number of semidefinite cones
 * @param[in] mnl   Number of nonlinear variables (default = 0)
 *
 * @return The computed inner product ⟨x,y⟩_S
 *
 * @note
 * - Uses BLAS ddot for efficient inner product computations
 * - For semidefinite blocks, off-diagonal terms are doubled
 * - Handles all three cone types (linear, quadratic, semidefinite)
 * - Vectors must have matching structure
 *
 * @warning
 * - x and y must be properly formatted for the cone dimensions
 * - dims must contain valid dimension information
 * - Undefined behavior if vectors have mismatched structure
 * - All cone components must be stored contiguously
 *
 * @see BLAS ddot documentation for inner product implementation
 */
double misc_sdot(matrix *x, matrix *y, DIMs *dims, int mnl)
{
    /*  Inner product of two vectors in S
        sdot(x, y, dims, mnl=0);
    */
    int m = mnl, int1 = 1, i, k, nk, inc, len;
    double a;

    if (m < 0) m = 0; // Ensure m is non-negative

    // Add dims->l to m
    m += dims->l;

    // Add sum of dims->q to m
    for (i = 0; i < dims->q_size; i++) {
        m += dims->q[i];
    }

    // Compute initial dot product
    a = ddot_(&m, MAT_BUFD(x), &int1, MAT_BUFD(y), &int1);

    // Process dims->s array
    for (k = 0; k < dims->s_size; k++) {
        nk = dims->s[k];
        inc = nk + 1;
        
        // Add dot product for diagonal elements
        a += ddot_(&nk, MAT_BUFD(x) + m, &inc, MAT_BUFD(y) + m, &inc);
        
        // Add dot products for off-diagonal elements (multiplied by 2)
        for (i = 1; i < nk; i++) {
            len = nk - i;
            a += 2.0 * ddot_(&len, MAT_BUFD(x) + m + i, &inc,
                MAT_BUFD(y) + m + i, &inc);
        }
        
        // Update offset for next matrix block
        m += nk * nk;
    }

    return a;
}



/**
 * @brief Computes the maximum feasible step size for x to remain in the cone
 * 
 * @details
 * Calculates the minimum value t such that x + t*e remains in the composite cone,
 * where e is defined differently for each cone type:
 * - Nonlinear and linear ('l') blocks: vector of ones
 * - Quadratic ('q') cones: first unit vector
 * - Semidefinite ('s') cones: identity matrix
 *
 * Optionally computes eigenvalues/eigenvectors for semidefinite blocks when
 * sigma is provided.
 *
 * @param[in] x      Input matrix in cone space
 * @param[in] dims   Cone dimensions structure:
 *                   - l: linear dimension
 *                   - q: array of quadratic cone dimensions
 *                   - q_size: number of quadratic cones
 *                   - s: array of semidefinite cone dimensions
 *                   - s_size: number of semidefinite cones
 * @param[in] mnl    Number of nonlinear variables (default = 0)
 * @param[in] sigma  Optional matrix for eigenvalues (for 's' blocks)
 *
 * @return Maximum feasible step size t
 *
 * @note
 * - Uses LAPACK routines (dsyevd/dsyevr) for eigenvalue computations
 * - Handles three cone types separately with appropriate definitions of e
 * - For 'q' cones, uses second-order cone structure
 * - For 's' cones, computes minimal eigenvalue when sigma=NULL
 *
 * @warning
 * - x must be properly formatted for the cone dimensions
 * - dims must contain valid dimension information
 * - When sigma is provided, it must have sufficient storage for eigenvalues
 * - Memory allocation failures will trigger error handling
 *
 * @see LAPACK dsyevd/dsyevr documentation for eigenvalue computation details
 */
double misc_max_step(matrix* x, DIMs* dims, int mnl, matrix* sigma)
{
    int i, mk, length, maxn, ind = mnl, ind2, int1 = 1, ld, info, lwork,
        *iwork = NULL, liwork, iwl, m;
    
    double t = -FLT_MAX, dbl0 = 0.0, *work = NULL, wl, *Q = NULL, *w = NULL;

    if (mnl < 0) ind = 0; // Ensure mnl is non-negative

    // 1. Handle nonlinear and 'l' part: vector of ones
    ind += dims->l;
    for (i = 0; i < ind; i++) t = MAX(t, -MAT_BUFD(x)[i]);

    // 2. Handle 'q' second-order cone part
    for (i = 0; i < dims->q_size; ++i) {
        mk = dims->q[i];
        length = mk - 1;
        t = MAX(t, dnrm2_(&length, MAT_BUFD(x) + ind + 1, &int1) -
            MAT_BUFD(x)[ind]);
        ind += mk;
    }

    // 3. Handle 's' semidefinite cone part
    // Find maximum dimension among 's' blocks for memory allocation
    for (i = 0, maxn = 0; i < dims->s_size; ++i) {
        maxn = MAX(maxn, dims->s[i]);
    }
    
    // If no 's' blocks or all have dimension 0, return current t
    if (!maxn) return (ind) ? t : 0.0;
    
    // Determine workspace size for LAPACK routines
    lwork = -1;
    liwork = -1;
    ld = MAX(1, maxn);
    if (sigma){
        dsyevd_("V", "L", &maxn, NULL, &ld, NULL, &wl, &lwork, &iwl,
            &liwork, &info);
    }
    else {
        if (!(Q = (double *) calloc(maxn * maxn, sizeof(double))) ||
            !(w = (double *) calloc(maxn, sizeof(double)))){
            free(Q); free(w);
            err_no_memory;
        }
        dsyevr_("N", "I", "L", &maxn, NULL, &ld, &dbl0, &dbl0, &int1,
            &int1, &dbl0, &maxn, NULL, NULL, &int1, NULL, &wl, &lwork,
            &iwl, &liwork, &info);
    }

    // Allocate workspace
    lwork = (int) wl;
    liwork = iwl;
    if (!(work = (double *) calloc(lwork, sizeof(double))) ||
        (!(iwork = (int *) calloc(liwork, sizeof(int))))){
        free(Q);  free(w);  free(work); free(iwork);
        err_no_memory;
    }

    // Process each 's' block
    for (i = 0, ind2 = 0; i < dims->s_size; ++i){
        mk = dims->s[i];
        if (mk){
            if (sigma){
                dsyevd_("V", "L", &mk, MAT_BUFD(x) + ind, &mk,
                    MAT_BUFD(sigma) + ind2, work, &lwork, iwork, &liwork,
                    &info);
                t = MAX(t, -MAT_BUFD(sigma)[ind2]);
            }
            else {
                length = mk*mk;
                dcopy_(&length, MAT_BUFD(x) + ind, &int1, Q, &int1);
                ld = MAX(1, mk);
                dsyevr_("N", "I", "L", &mk, Q, &mk, &dbl0, &dbl0, &int1,
                    &int1, &dbl0, &m, w, NULL, &int1, NULL, work, &lwork,
                    iwork, &liwork, &info);
                t = MAX(t, -w[0]);
            }
        }
        ind += mk*mk;
        ind2 += mk;
    }

    // Free allocated memory
    free(work);  free(iwork);  free(Q);  free(w);

    return (ind) ? t : 0.0;
}

/**
 * @brief Computes the S-norm (specialized norm) of a structured matrix
 * 
 * @details
 * Calculates the S-norm of matrix x, defined as the square root of the
 * S-dot product of x with itself. The S-norm is a specialized norm that
 * respects the matrix structure defined by dims:
 * - Handles linear, quadratic cone, and semidefinite cone components
 * - Properly accounts for non-negative variables (mnl)
 *
 * @param[in] x     Input matrix (structured according to dims)
 * @param[in] dims  Structure containing matrix dimensions:
 *                  - l: linear dimension
 *                  - q: array of quadratic cone dimensions
 *                  - q_size: number of quadratic cones
 *                  - s: array of semidefinite cone dimensions
 *                  - s_size: number of semidefinite cones
 * @param[in] mnl   Number of non-negative variables (default = 0)
 *
 * @return The S-norm of the matrix (sqrt of S-dot product with itself)
 */
double misc_snrm2(matrix *x, DIMs *dims, int mnl)
{
    // Compute the S-dot product of x with itself
    double inner_product = misc_sdot(x, x, dims, mnl);
    // Return the square root of the dot product (i.e., the norm)
    return sqrt(inner_product);
}


/**
 * @brief Computes the element-wise square product x := y ∘ y for a structured matrix
 * 
 * @details
 * Computes the Hadamard (element-wise) product of y with itself, storing the result in x.
 * Handles special structure where:
 * - The 's' components of y are diagonal
 * - Only the diagonals of x and y are stored
 * - Processes three types of matrix blocks:
 *   1. Linear blocks (mnl + dims->l)
 *   2. Quadratic cones (dims->q)
 *   3. Semidefinite cones (dims->s)
 *
 * @param[out] x    Output matrix (stores result y ∘ y)
 * @param[in] y     Input matrix (source values)
 * @param[in] dims  Structure containing matrix dimensions:
 *                  - l: linear dimension
 *                  - q: array of quadratic cone dimensions
 *                  - q_size: number of quadratic cones
 *                  - s: array of semidefinite cone dimensions
 *                  - s_size: number of semidefinite cones
 * @param[in] mnl   Number of non-negative variables (default = 0)
 *
 * @note
 * - Uses BLAS operations for efficient computation:
 *   1. blas_copy for initial copy operation
 *   2. blas_tbmv for triangular band matrix operations
 *   3. blas_nrm2 for Euclidean norm calculations
 *   4. blas_scal for scaling operations
 * - For quadratic cones, computes squared norms and scales appropriately
 * - For semidefinite cones, processes only diagonal elements
 * - All operations preserve the structured format of the matrices
 *
 * @warning
 * - Matrices x and y must be properly allocated and compatible in size
 * - dims structure must contain valid dimension information
 * - All diagonal elements must be stored contiguously
 * - For quadratic cones, assumes standard second-order cone structure
 */
void misc_ssqr(matrix *x, matrix *y, DIMs *dims, int mnl) 
{
    
    int ind = mnl + dims->l;

    // blas.copy(y, x)
    blas_copy(y, x, len(y), 1, 1, 0, 0);
    
    // blas.tbmv(y, x, n = mnl + dims['l'], k = 0, ldA = 1)
    blas_tbmv(y, x, 'L', 'N', 'N', ind, 0, 1, 1, 0, 0);
    
    // for m in dims['q']:
    for (int i = 0; i < dims->q_size; ++i) {
        int m = dims->q[i];
        
        // x[ind] = blas.nrm2(y, offset = ind, n = m)**2
        double nrm = blas_nrm2(y, m, 1, ind);
        MAT_BUFD(x)[ind] = nrm * nrm;

        // blas.scal(2.0*y[ind], x, n = m-1, offset = ind+1)
        // blas_scal(m-1, 2.0 * y[ind], &x[ind+1], 1);
        double y_val = MAT_BUFD(y)[ind] * 2.0;
        blas_scal(&y_val, x, m - 1, 1, ind + 1);
        ind += m;
    }
    
    // blas.tbmv(y, x, n = sum(dims['s']), k = 0, ldA = 1, offsetA = ind, offsetx = ind)
    int s_sum = sum_array(dims->s, dims->s_size);
    blas_tbmv(y, x, 0, 0, 0, s_sum, 0, 1, 1, ind, ind);
}


/**
 * Computes x' * J * y where J = [1, 0; 0, -I]
 * 
 * @param x Input matrix/vector x
 * @param y Input matrix/vector y  
 * @param n Length of vectors. If negative, uses length of x (default = -1)
 * @param offsetx Starting offset in x vector (default = 0)
 * @param offsety Starting offset in y vector (default = 0)
 * @return Result of x^T J y
 */
double misc_jdot(matrix* x, matrix* y, int n, int offsetx, int offsety) 
{
    if (offsetx < 0) offsetx = 0;
    if (offsety < 0) offsety = 0;

    // Validate input matrices
    if (!x || !y) ERR("Error: NULL matrix pointer\n");
    
    // Use x length if n is negative
    if (n < 0) {
        if(len(x) != len(y)) {
            ERR_TYPE("x and y must have the same length");
        }
        n = len(x) ;
    }

    // Check bounds
    if (offsetx >= len(x) || offsety >= len(y)) err_buf_len("x or y");
    
    if (n == 0) return 0.0;  // If n is zero, return zero

    double term1 = MAT_BUFD(x)[offsetx] * MAT_BUFD(y)[offsety];  // x0 * y0
    number dot_res = blas_dot(x, y, n - 1, 1, 1, offsetx + 1, offsety + 1);

    return term1 - dot_res.d;  // x0 * y0 - sum(xi * yi) for i > 0
}

/**
 * @brief Computes the Nesterov-Todd scaling W at points s and z
 * 
 * @details
 * Returns the Nesterov-Todd scaling W at points s and z, and stores the 
 * scaled variable in lmbda. The scaling is computed as:
 * 
 *     W * z = W^{-T} * s = lmbda
 *
 * Handles nonlinear blocks, linear blocks, and quadratic blocks.
 *
 * @param[in] s       First input matrix/spmatrix
 * @param[in] z       Second input matrix/spmatrix
 * @param[in] lmbda   lmbda matrix/spmatrix
 * @param[in] dims    Dimensions structure
 * @param[in] mnl     Number of nonlinear variables (default = 0)
 *
 * @return Scaling structure containing computed scaling factors
 */
scaling* misc_compute_scaling(matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int mnl) 
{

    scaling *W = malloc(sizeof(scaling));
    Scaling_Init(W);
    // memset(&W, 0, sizeof(scaling));
    
    // For the nonlinear block:
    //
    //     W['dnl'] = sqrt( s[:mnl] ./ z[:mnl] )
    //     W['dnli'] = sqrt( z[:mnl] ./ s[:mnl] )
    //     lambda[:mnl] = sqrt( s[:mnl] .* z[:mnl] )

    if (mnl < 0) {
        mnl = 0;    // Default to 0 if mnl is nonpositive
    } else {
        matrix* s_mnl = matrix_slice(s, 0, mnl);
        matrix* z_mnl = matrix_slice(z, 0, mnl);

        matrix* tmp = base_ediv(s_mnl, z_mnl, 0, 0, DOUBLE, DOUBLE);
        W->dnl = base_sqrt(tmp, 0, DOUBLE);
        Matrix_Free(tmp);

        double exponent = -1.0;
        W->dnli = base_pow(W->dnl, &exponent, 0, DOUBLE, DOUBLE);
        tmp = base_emul(s_mnl, z_mnl, 0, 0, DOUBLE, DOUBLE);

        matrix* res = base_sqrt(tmp, 0, DOUBLE);
        for (int i = 0; i < mnl; ++i) {
            MAT_BUFD(lmbda)[i] = MAT_BUFD(res)[i];
        }

        // Free all
        Matrix_Free(tmp);
        Matrix_Free(res);
        Matrix_Free(s_mnl);
        Matrix_Free(z_mnl);
    }

    /* For the 'l' block:
    *
    *     W['d'] = sqrt(sk ./ zk)
    *     W['di'] = sqrt(zk ./ sk)
    *     lambdak = sqrt(sk .* zk)
    *
    * where:
    * - sk and zk are the first dims['l'] entries of s and z
    * - lambda_k is stored in the first dims['l'] positions of lmbda
    */
             
    int m = dims->l;
    // void *res = base_ediv(s, z, )
    // if (m == 0) {
        // W->d = NULL;
        // W->di = NULL;
    // } 
    
    matrix *s_mnl_m = matrix_slice(s, mnl, m);
    matrix *z_mnl_m = matrix_slice(z, mnl, m);
    matrix *tmp = base_ediv(s_mnl_m, z_mnl_m, 0, 0, DOUBLE, DOUBLE);
    W->d = base_sqrt(tmp, 0, DOUBLE);
    Matrix_Free(tmp);

    double exponent = -1.0;
    W->di = base_pow(W->d, &exponent, 0, DOUBLE, DOUBLE);
    tmp = base_emul(s_mnl_m, z_mnl_m, 0, 0, DOUBLE, DOUBLE);
    matrix *res = base_sqrt(tmp, 0, DOUBLE);
    for (int i = 0; i < m; ++i) {
        MAT_BUFD(lmbda)[mnl + i] = MAT_BUFD(res)[i];
    }
    // Free all
    Matrix_Free(tmp);
    Matrix_Free(res);
    Matrix_Free(s_mnl_m);
    Matrix_Free(z_mnl_m);

    /*
    * For the 'q' blocks, compute lists 'v' and 'beta':
    *
    * - The vector v[k] has unit hyperbolic norm:
    *     sqrt(v[k]' * J * v[k]) = 1, where J = [1, 0; 0, -I]
    *   
    * - beta[k] is a positive scalar
    *
    * The hyperbolic Householder matrix H = 2*v[k]*v[k]' - J defined by v[k]
    * satisfies:
    *
    *     (beta[k] * H) * zk = (beta[k] * H) \ sk = lambda_k
    *
    * where:
    * - sk = s[indq[k]:indq[k+1]]
    * - zk = z[indq[k]:indq[k+1]]
    * - lambda_k is stored in lmbda[indq[k]:indq[k+1]]
    */
           
    int ind = mnl + dims->l;
    W->v_count = dims->q_size;
    W->v = (matrix**)malloc(dims->q_size * sizeof(matrix*));
    for (int k = 0; k < dims->q_size; ++k) {
        W->v[k] = Matrix_New(dims->q[k], 1, DOUBLE);
    }
    
    W->b_count= dims->q_size;
    W->beta = (double*)calloc(dims->q_size, sizeof(double));

    for (int k = 0; k < dims->q_size; ++k) {
        m = dims->q[k];
        matrix *v = W->v[k];

        // a = sqrt( sk' * J * sk )  where J = [1, 0; 0, -I]
        double aa = misc_jnrm2(s, m, ind);

        // b = sqrt( zk' * J * zk )
        double bb = misc_jnrm2(z, m, ind);

        // beta[k] = ( a / b )**1/2
        W->beta[k] = sqrt(aa / bb);

        // c = sqrt( (sk/a)' * (zk/b) + 1 ) / sqrt(2)    
        number tmp_num = blas_dot(s, z, m, 1, 1, ind, ind);
        double cc = sqrt((tmp_num.d / aa / bb + 1.0) / 2.0);

        // vk = 1/(2*c) * ( (sk/a) + J * (zk/b) )
        blas_copy(z, v, m, 1, 1, ind, 0);

        double alpha = -1.0/bb;
        blas_scal(&alpha, v, -1, 1, 0);
        MAT_BUFD(v)[0] *= -1.0;

        number alpha_n;
        alpha_n.d = 1.0 / aa;
        alpha = 1.0 / (2.0 * cc);
        blas_axpy(s, v, &alpha_n, m, 1, 1, ind, 0);
        blas_scal(&alpha, v, -1, 1, 0);

        // v[k] = 1/sqrt(2*(vk0 + 1)) * ( vk + e ),  e = [1; 0]
        MAT_BUFD(v)[0] += 1.0;
        alpha = 1.0 / sqrt(2.0 * MAT_BUFD(v)[0]);
        blas_scal(&alpha, v, -1, 1, 0);
            
        // To get the scaled variable lambda_k
        // 
        //     d =  sk0/a + zk0/b + 2*c
        //     lambda_k = [ c; 
        //                  (c + zk0/b)/d * sk1/a + (c + sk0/a)/d * zk1/b ]
        //     lambda_k *= sqrt(a * b)

        MAT_BUFD(lmbda)[ind] = cc;
        double saa = MAT_BUFD(s)[ind] / aa;
        double zbb = MAT_BUFD(z)[ind] / bb;
        double dd = 2*cc + saa + zbb;
        double tmp = (zbb+cc) / dd / aa;
        double tmp2 = sqrt(aa * bb);
        alpha_n.d = (cc + saa)/dd/bb;
        blas_copy(s, lmbda, m-1, 1, 1, ind+1, ind+1);
        blas_scal(&tmp, lmbda, m-1, 1, ind+1);
        blas_axpy(z, lmbda, &alpha_n, m-1, 1, 1, ind+1, ind+1);
        blas_scal(&tmp2, lmbda, m, 1, ind);

        ind += m;
    }

    /* For the 's' blocks: compute two lists 'r' and 'rti':
    *
    *     r[k]' * sk^{-1} * r[k] = diag(lambda_k)^{-1}
    *     r[k]' * zk * r[k] = diag(lambda_k)
    *
    * where:
    * - sk and zk are the entries inds[k] : inds[k+1] of s and z,
    *   reshaped into symmetric matrices
    *
    * rti[k] is the inverse of r[k]', so:
    *
    *     rti[k]' * sk * rti[k] = diag(lambda_k)^{-1}
    *     rti[k]' * zk^{-1} * rti[k] = diag(lambda_k)
    *
    * The vectors lambda_k are stored in:
    *     lmbda[ dims['l'] + sum(dims['q']) : -1 ]
    */
            
    W->r_count = dims->s_size;
    W->r = (matrix**)malloc(dims->s_size * sizeof(matrix*));
    W->rti = (matrix**)malloc(dims->s_size * sizeof(matrix*));
    
    for (int k = 0; k < dims->s_size; ++k) {
        W->r[k] = Matrix_New(dims->s[k], dims->s[k], DOUBLE);
        W->rti[k] = Matrix_New(dims->s[k], dims->s[k], DOUBLE);
    }
    
    int max_s = max_array(dims->s, dims->s_size);
    // if (max_s == 0) max_s = 1; // Ensure at least size 1
    
    matrix *work = Matrix_New(max_s * max_s, 1, DOUBLE);
    matrix *Ls = Matrix_New(max_s * max_s, 1, DOUBLE);
    matrix *Lz = Matrix_New(max_s * max_s, 1, DOUBLE);

    int ind2 = ind;
    for (int k = 0; k < dims->s_size; ++k) {
        m = dims->s[k];
        matrix *r = W->r[k];
        matrix *rti = W->rti[k];

        // Factor sk = Ls*Ls'; store Ls in ds[inds[k]:inds[k+1]].
        blas_copy(s, Ls, m*m, 1, 1, ind2, 0);
        lapack_potrf(Ls, 'L', m, m, 0);

        // Factor zs[k] = Lz*Lz'; store Lz in dz[inds[k]:inds[k+1]].
        blas_copy(z, Lz, m*m, 1, 1, ind2, 0);
        lapack_potrf(Lz, 'L', m, m, 0);
	 
        // SVD Lz'*Ls = U*diag(lambda_k)*V'.  Keep U in work. 
        double alpha = 0.0;
        for (int i = 0; i < m; i++) {
            blas_scal(&alpha, Ls, i, 1, i*m);
        }

        blas_copy(Ls, work, m*m, 1, 1, 0, 0);
        blas_trmm(Lz, work, 'L', 'L', 'T', 'N', NULL, m, m, m, m, 0, 0);

        lapack_gesvd(work, lmbda, 'O', 'N', NULL, NULL, m, m, m, 0, 0, 0, ind, 0, 0);

        // r = Lz^{-T} * U 
        blas_copy(work, r, m*m, 1, 1, 0, 0);
        blas_trsm(Lz, r, 'L', 'L', 'T', 'N', NULL, m, m, m, 0, 0, 0);

        // rti = Lz * U 
        blas_copy(work, rti, m*m, 1, 1, 0, 0);
        blas_trmm(Lz, rti, 'L', 'L', 'N', 'N', NULL, m, m, m, 0, 0, 0);

        // r := r * diag(sqrt(lambda_k))
        // rti := rti * diag(1 ./ sqrt(lambda_k))
        for (int i = 0; i < m; ++i) {
            double a = sqrt(MAT_BUFD(lmbda)[ind + i]);
            double tmp_a = 1.0 / a;
            blas_scal(&a, r, m, 1, m*i);
            blas_scal(&tmp_a, rti, m, 1, m*i);
        }
        ind += m;
        ind2 += m*m;
    }
    
    // Clean up temporary matrices
    Matrix_Free(work);
    Matrix_Free(Ls);
    Matrix_Free(Lz);

    return W;
}


/**
 * @brief Updates the Nesterov-Todd scaling matrix W and scaled variable lmbda
 * 
 * @details
 * Updates the Nesterov-Todd scaling matrix W and the scaled variable lmbda
 * so that on exit:
 * 
 *     W * zt = W^{-T} * st = lmbda
 *
 * On entry, the nonlinear, 'l', and 'q' components of the arguments s and z
 * contain W^{-T}*st and W*zt, i.e., the new iterates in the current scaling.
 *
 * The 's' components contain the factors Ls, Lz in a factorization of the new
 * iterates in the current scaling, W^{-T}*st = Ls*Ls', W*zt = Lz*Lz'.
 *
 * @param[in,out] W      Scaling structure containing dnl, d, v, beta, r, rti
 * @param[in,out] lmbda  Scaled variable (output)
 * @param[in,out] s      New iterate in current scaling (input/output)
 * @param[in,out] z      New iterate in current scaling (input/output)
 */
void misc_update_scaling(scaling *W, matrix *lmbda, matrix *s, matrix *z) 
{

    /* Nonlinear and 'l' blocks:
    *
    *    d := d .* sqrt(s ./ z)
    *    lmbda := lmbda .* sqrt(s) .* sqrt(z)
    */

    int mnl, ml, m;
    if (W->has_dnl) {
        mnl = W->dnl_count;
    } else {
        mnl = 0;
    }
    ml = W->d_count;
    m = mnl + ml;
    
    matrix *s_m = matrix_slice(s, 0, m);
    matrix *z_m = matrix_slice(z, 0, m);
    matrix *tmp1 = base_sqrt(s_m, 0, DOUBLE);
    matrix *tmp2 = base_sqrt(z_m, 0, DOUBLE);

    for(int i = 0; i < m; ++i) {
        MAT_BUFD(s)[i] = MAT_BUFD(tmp1)[i];
        MAT_BUFD(z)[i] = MAT_BUFD(tmp2)[i];
    }

    Matrix_Free(tmp1);
    Matrix_Free(tmp2);
    Matrix_Free(s_m);
    Matrix_Free(z_m);
 
    // d := d .* s .* z 
    if (W->has_dnl) {
        blas_tbmv(s, W->dnl, 'L', 'N', 'N', mnl, 0, 1, 1, 0, 0);
        blas_tbsv(z, W->dnl, 'L', 'N', 'N', mnl, 0, 1, 1, 0, 0);
        matrix *tmp = base_pow(W->dnl, &((double){-1.0}), 0, DOUBLE, DOUBLE);
        for(int i = 0; i < MAT_LGT(W->dnl); ++i) {
            MAT_BUFD(W->dnl)[i] = MAT_BUFD(tmp)[i];
        }
        Matrix_Free(tmp);
    }
    blas_tbmv(s, W->d, 'L', 'N', 'N', ml, 0, 1, 1, mnl, 0);
    blas_tbsv(z, W->d, 'L', 'N', 'N', ml, 0, 1, 1, mnl, 0);
    matrix *tmp = base_pow(W->d, &((double){-1.0}), 0, DOUBLE, DOUBLE);
    for(int i = 0; i < MAT_LGT(W->d); ++i) {
        MAT_BUFD(W->d)[i] = MAT_BUFD(tmp)[i];
    }
    Matrix_Free(tmp);
         
    // lmbda := s .* z
    blas_copy(s, lmbda, m, 1, 1, 0, 0);
    blas_tbmv(z, lmbda, 'L', 'N', 'N', m, 0, 1, 1, 0, 0);

    /* 'q' blocks:
    *
    * Let st and zt be the new variables in the old scaling:
    *
    *     st = s_k,   zt = z_k
    *
    * and a = sqrt(st' * J * st),  b = sqrt(zt' * J * zt).
    *
    * 1. Compute the hyperbolic Householder transformation 2*q*q' - J 
    *    that maps st/a to zt/b:
    *
    *        c = sqrt( (1 + st'*zt/(a*b)) / 2 )
    *        q = (st/a + J*zt/b) / (2*c)
    *
    *    The new scaling point is:
    *
    *        wk := betak * sqrt(a/b) * (2*v[k]*v[k]' - J) * q
    *
    *    with betak = W['beta'][k]
    *
    * 2. The scaled variable:
    *
    *        lambda_k0 = sqrt(a*b) * c
    *        lambda_k1 = sqrt(a*b) * ( (2vk*vk' - J) * (-d*q + u/2) )_1
    *
    *    where:
    *
    *        u = st/a - J*zt/b
    *        d = ( vk0 * (vk'*u) + u0/2 ) / (2*vk0 *(vk'*q) - q0 + 1)
    *
    * 3. Update scaling:
    *
    *        v[k] := wk^(1/2)
    *              = 1 / sqrt(2*(wk0 + 1)) * (wk + e)
    *        beta[k] *= sqrt(a/b)
    */

    int ind = m;
    for (int k = 0; k < W->v_count; ++k) {
        matrix *v = W->v[k];
        m = len(v);

        double *v_data = (double *)v->buffer;
        double *s_data = (double *)s->buffer;
        double *z_data = (double *)z->buffer;

        // ln = sqrt( lambda_k' * J * lambda_k )
        // double ln = misc_jnrm2(lmbda, m, ind);

        // a = sqrt( sk' * J * sk ) = sqrt( st' * J * st ) 
        // s := s / a = st / a
        double aa = misc_jnrm2(s, m, ind);
        double alpha = 1.0 / aa;
        blas_scal(&alpha, s, m, 1, ind);

        // b = sqrt( zk' * J * zk ) = sqrt( zt' * J * zt )
        // z := z / a = zt / b
        double bb = misc_jnrm2(z, m, ind);

        alpha = 1.0 / bb;
        blas_scal(&alpha, z, m, 1, ind);

        // c = sqrt( ( 1 + (st'*zt) / (a*b) ) / 2 )
        number c_tmp = blas_dot(s, z, m, 1, 1, ind, ind);
        double cc = sqrt((1.0 + c_tmp.d) / 2.0);

        // vs = v' * st / a 
        number vs = blas_dot(v, s, m, 1, 1, ind, ind);

        // vz = v' * J *zt / b
        double vz = misc_jdot(v, z, m, 0, ind);

        // vq = v' * q where q = (st/a + J * zt/b) / (2 * c)
        double vq = (vs.d + vz) / 2.0 / cc;

        // vu = v' * u  where u =  st/a - J * zt/b 
        double vu = vs.d - vz;

        // lambda_k0 = c
        MAT_BUFD(lmbda)[ind] = cc;

        // wk0 = 2 * vk0 * (vk' * q) - q0 
        double wk0 = 2.0 * v_data[0] * vq - (s_data[ind] + z_data[ind]) / 2.0 / cc;

        // d = (v[0] * (vk' * u) - u0/2) / (wk0 + 1)
        double dd = (v_data[0] * vu - s_data[ind]/2.0 + z_data[ind]/2.0) / (wk0 + 1.0);

        // lambda_k1 = 2 * v_k1 * vk' * (-d*q + u/2) - d*q1 + u1/2
        blas_copy(v, lmbda, m-1, 1, 1, 1, ind+1);
        alpha = 2.0 * (-dd * vq + 0.5 * vu);
        blas_scal(&alpha, lmbda, m-1, 1, ind+1);

        number alpha_n;
        alpha_n.d = 0.5 * (1.0 - dd/cc);
        blas_axpy(s, lmbda, &alpha_n, m-1, 1, 1, ind+1, ind+1);
        blas_axpy(z, lmbda, &alpha_n, m-1, 1, 1, ind+1, ind+1);

        // Scale so that sqrt(lambda_k' * J * lambda_k) = sqrt(aa*bb).
        alpha = sqrt(aa * bb);
        blas_scal(&alpha, lmbda, m, 1, ind);
            
        // v := (2*v*v' - J) * q 
        //    = 2 * (v'*q) * v' - (J* st/a + zt/b) / (2*c)
        alpha = 2.0 * vq;
        blas_scal(&alpha, v, -1, 1, 0);
        v_data[0] -= s_data[ind] / 2.0 / cc;
        alpha_n.d = 0.5 / cc;
        blas_axpy(s, v, &alpha_n, m-1, 1, 1, ind+1, 1);
        blas_axpy(z, v, &alpha_n, m, 1, 1, ind, 0);

        // v := v^{1/2} = 1/sqrt(2 * (v0 + 1)) * (v + e)
        v_data[0] += 1.0;
        alpha = 1.0 / sqrt(2.0 * v_data[0]);
        blas_scal(&alpha, v, -1, 1, 0);

        // beta[k] *= ( aa / bb )**1/2
        W->beta[k] *= sqrt(aa / bb);
            
        ind += m;
    }

    // 's' blocks
    // 
    // Let st, zt be the updated variables in the old scaling:
    // 
    //     st = Ls * Ls', zt = Lz * Lz'.
    //
    // where Ls and Lz are the 's' components of s, z.
    //
    // 1.  SVD Lz'*Ls = Uk * lambda_k^+ * Vk'.
    //
    // 2.  New scaling is 
    //
    //         r[k] := r[k] * Ls * Vk * diag(lambda_k^+)^{-1/2}
    //         rti[k] := r[k] * Lz * Uk * diag(lambda_k^+)^{-1/2}.
    //

    // Calculate maximum dimension for workspace
    int max_r_sizes[W->r_count];
    for (int k = 0; k < W->r_count; k++) {
        max_r_sizes[k] = W->r[k]->nrows;
    }
    int max_r_size = max_array(max_r_sizes, W->r_count);
    matrix *work = Matrix_New(max_r_size * max_r_size, 1, DOUBLE);
    double *lmbda_data = (double *)lmbda->buffer;
    
    // Calculate starting indices
    int v_total = 0;
    for (int k = 0; k < W->v_count; k++) {
        v_total += len(W->v[k]);
    }
    
    ind = mnl + ml + v_total;
    int ind2 = ind;
    int ind3 = 0;
    
    for (int k = 0; k < W->r_count; k++) {
        matrix *r = W->r[k];
        matrix *rti = W->rti[k];
        int m = r->nrows;

        // r := r*sk = r*Ls
        blas_gemm(r, s, work, 'N', 'N', NULL, NULL, m, m, m, 0, m, m, 0, ind2, 0);
        blas_copy(work, r, m*m, 1, 1, 0, 0);

        // rti := rti*zk = rti*Lz
        blas_gemm(rti, z, work, 'N', 'N', NULL, NULL, m, m, m, 0, m, m, 0, ind2, 0);
        blas_copy(work, rti, m*m, 1, 1, 0, 0);

        // SVD Lz'*Ls = U * lmbds^+ * V'; store U in sk and V' in zk.
        // blas_gemm(z, s, work, 'T', 'N', NULL, NULL, m, m, m, m, m, m, 0, ind2, 0); // wo cao ni ma bi
        blas_gemm(z, s, work, 'T', 'N', NULL, NULL, m, m, m, m, m, m, ind2, ind2, 0);
        lapack_gesvd(work, lmbda, 'A', 'A', s, z, m, m, m, m, m, 0, ind, ind2, ind2);

        // r := r*V
        blas_gemm(r, z, work, 'N', 'T', NULL, NULL, m, m, m, 0, m, m, 0, ind2, 0);
        blas_copy(work, r, m*m, 1, 1, 0, 0);

        // rti := rti*U
        blas_gemm(rti, s, work, 'N', 'N', NULL, NULL, m, m, m, 0, m, m, 0, ind2, 0);
        blas_copy(work, rti, m*m, 1, 1, 0, 0);

        // r := r*lambda^{-1/2}; rti := rti*lambda^{-1/2}
        for (int i = 0; i < m; i++) {
            double a = 1.0 / sqrt(lmbda_data[ind + i]);
            blas_scal(&a, r, m, 1, m * i);
            blas_scal(&a, rti, m, 1, m * i);
        }

        ind += m;
        ind2 += m * m;
        ind3 += m;
    }
    
    // Clean up workspace
    Matrix_Free(work);
}


void misc_compute_scaling2(scaling *W, matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int mnl) 
{

    if (W == NULL) ERR("misc_compute_scaling2: W is NULL");
    
    // For the nonlinear block:
    //
    //     W['dnl'] = sqrt( s[:mnl] ./ z[:mnl] )
    //     W['dnli'] = sqrt( z[:mnl] ./ s[:mnl] )
    //     lambda[:mnl] = sqrt( s[:mnl] .* z[:mnl] )

    if (mnl < 0) {
        mnl = 0;    // Default to 0 if mnl is nonpositive
    } else {
        matrix* s_mnl = matrix_slice(s, 0, mnl);
        matrix* z_mnl = matrix_slice(z, 0, mnl);

        matrix* tmp = base_ediv(s_mnl, z_mnl, 0, 0, DOUBLE, DOUBLE);
        W->dnl = base_sqrt(tmp, 0, DOUBLE);
        Matrix_Free(tmp);

        double exponent = -1.0;
        W->dnli = base_pow(W->dnl, &exponent, 0, DOUBLE, DOUBLE);
        tmp = base_emul(s_mnl, z_mnl, 0, 0, DOUBLE, DOUBLE);

        matrix* res = base_sqrt(tmp, 0, DOUBLE);
        for (int i = 0; i < mnl; ++i) {
            MAT_BUFD(lmbda)[i] = MAT_BUFD(res)[i];
        }

        // Free all
        Matrix_Free(tmp);
        Matrix_Free(res);
        Matrix_Free(s_mnl);
        Matrix_Free(z_mnl);
    }

    /* For the 'l' block:
    *
    *     W['d'] = sqrt(sk ./ zk)
    *     W['di'] = sqrt(zk ./ sk)
    *     lambdak = sqrt(sk .* zk)
    *
    * where:
    * - sk and zk are the first dims['l'] entries of s and z
    * - lambda_k is stored in the first dims['l'] positions of lmbda
    */
             
    int m = dims->l;
    matrix *s_mnl_m = matrix_slice(s, mnl, m);
    matrix *z_mnl_m = matrix_slice(z, mnl, m);
    matrix *tmp = base_ediv(s_mnl_m, z_mnl_m, 0, 0, DOUBLE, DOUBLE);
    W->d = base_sqrt(tmp, 0, DOUBLE);
    Matrix_Free(tmp);

    double exponent = -1.0;
    W->di = base_pow(W->d, &exponent, 0, DOUBLE, DOUBLE);
    tmp = base_emul(s_mnl_m, z_mnl_m, 0, 0, DOUBLE, DOUBLE);
    matrix *res = base_sqrt(tmp, 0, DOUBLE);
    for (int i = 0; i < m; ++i) {
        MAT_BUFD(lmbda)[mnl + i] = MAT_BUFD(res)[i];
    }
    // Free all
    Matrix_Free(tmp);
    Matrix_Free(res);
    Matrix_Free(s_mnl_m);
    Matrix_Free(z_mnl_m);

    /*
    * For the 'q' blocks, compute lists 'v' and 'beta':
    *
    * - The vector v[k] has unit hyperbolic norm:
    *     sqrt(v[k]' * J * v[k]) = 1, where J = [1, 0; 0, -I]
    *   
    * - beta[k] is a positive scalar
    *
    * The hyperbolic Householder matrix H = 2*v[k]*v[k]' - J defined by v[k]
    * satisfies:
    *
    *     (beta[k] * H) * zk = (beta[k] * H) \ sk = lambda_k
    *
    * where:
    * - sk = s[indq[k]:indq[k+1]]
    * - zk = z[indq[k]:indq[k+1]]
    * - lambda_k is stored in lmbda[indq[k]:indq[k+1]]
    */
           
    int ind = mnl + dims->l;
    for (int k = 0; k < dims->q_size; ++k) {
        m = dims->q[k];
        matrix *v = W->v[k];

        // a = sqrt( sk' * J * sk )  where J = [1, 0; 0, -I]
        double aa = misc_jnrm2(s, m, ind);

        // b = sqrt( zk' * J * zk )
        double bb = misc_jnrm2(z, m, ind);

        // beta[k] = ( a / b )**1/2
        W->beta[k] = sqrt(aa / bb);

        // c = sqrt( (sk/a)' * (zk/b) + 1 ) / sqrt(2)    
        number tmp_num = blas_dot(s, z, m, 1, 1, ind, ind);
        double cc = sqrt((tmp_num.d / aa / bb + 1.0) / 2.0);

        // vk = 1/(2*c) * ( (sk/a) + J * (zk/b) )
        blas_copy(z, v, m, 1, 1, ind, 0);

        double alpha = -1.0/bb;
        blas_scal(&alpha, v, -1, 1, 0);
        MAT_BUFD(v)[0] *= -1.0;

        number alpha_n;
        alpha_n.d = 1.0 / aa;
        alpha = 1.0 / (2.0 * cc);
        blas_axpy(s, v, &alpha_n, m, 1, 1, ind, 0);
        blas_scal(&alpha, v, -1, 1, 0);

        // v[k] = 1/sqrt(2*(vk0 + 1)) * ( vk + e ),  e = [1; 0]
        MAT_BUFD(v)[0] += 1.0;
        alpha = 1.0 / sqrt(2.0 * MAT_BUFD(v)[0]);
        blas_scal(&alpha, v, -1, 1, 0);
            
        // To get the scaled variable lambda_k
        // 
        //     d =  sk0/a + zk0/b + 2*c
        //     lambda_k = [ c; 
        //                  (c + zk0/b)/d * sk1/a + (c + sk0/a)/d * zk1/b ]
        //     lambda_k *= sqrt(a * b)

        MAT_BUFD(lmbda)[ind] = cc;
        double saa = MAT_BUFD(s)[ind] / aa;
        double zbb = MAT_BUFD(z)[ind] / bb;
        double dd = 2*cc + saa + zbb;
        double tmp = (zbb+cc) / dd / aa;
        double tmp2 = sqrt(aa * bb);
        alpha_n.d = (cc + saa)/dd/bb;
        blas_copy(s, lmbda, m-1, 1, 1, ind+1, ind+1);
        blas_scal(&tmp, lmbda, m-1, 1, ind+1);
        blas_axpy(z, lmbda, &alpha_n, m-1, 1, 1, ind+1, ind+1);
        blas_scal(&tmp2, lmbda, m, 1, ind);

        ind += m;
    }

    /* For the 's' blocks: compute two lists 'r' and 'rti':
    *
    *     r[k]' * sk^{-1} * r[k] = diag(lambda_k)^{-1}
    *     r[k]' * zk * r[k] = diag(lambda_k)
    *
    * where:
    * - sk and zk are the entries inds[k] : inds[k+1] of s and z,
    *   reshaped into symmetric matrices
    *
    * rti[k] is the inverse of r[k]', so:
    *
    *     rti[k]' * sk * rti[k] = diag(lambda_k)^{-1}
    *     rti[k]' * zk^{-1} * rti[k] = diag(lambda_k)
    *
    * The vectors lambda_k are stored in:
    *     lmbda[ dims['l'] + sum(dims['q']) : -1 ]
    */
    
    int max_s = max_array(dims->s, dims->s_size);
    
    matrix *work = Matrix_New(max_s * max_s, 1, DOUBLE);
    matrix *Ls = Matrix_New(max_s * max_s, 1, DOUBLE);
    matrix *Lz = Matrix_New(max_s * max_s, 1, DOUBLE);

    int ind2 = ind;
    for (int k = 0; k < dims->s_size; ++k) {
        m = dims->s[k];
        matrix *r = W->r[k];
        matrix *rti = W->rti[k];

        // Factor sk = Ls*Ls'; store Ls in ds[inds[k]:inds[k+1]].
        blas_copy(s, Ls, m*m, 1, 1, ind2, 0);
        lapack_potrf(Ls, 'L', m, m, 0);

        // Factor zs[k] = Lz*Lz'; store Lz in dz[inds[k]:inds[k+1]].
        blas_copy(z, Lz, m*m, 1, 1, ind2, 0);
        lapack_potrf(Lz, 'L', m, m, 0);
	 
        // SVD Lz'*Ls = U*diag(lambda_k)*V'.  Keep U in work. 
        double alpha = 0.0;
        for (int i = 0; i < m; i++) {
            blas_scal(&alpha, Ls, i, 1, i*m);
        }

        blas_copy(Ls, work, m*m, 1, 1, 0, 0);
        blas_trmm(Lz, work, 'L', 'L', 'T', 'N', NULL, m, m, m, m, 0, 0);

        lapack_gesvd(work, lmbda, 'O', 'N', NULL, NULL, m, m, m, 0, 0, 0, ind, 0, 0);

        // r = Lz^{-T} * U 
        blas_copy(work, r, m*m, 1, 1, 0, 0);
        blas_trsm(Lz, r, 'L', 'L', 'T', 'N', NULL, m, m, m, 0, 0, 0);

        // rti = Lz * U 
        blas_copy(work, rti, m*m, 1, 1, 0, 0);
        blas_trmm(Lz, rti, 'L', 'L', 'N', 'N', NULL, m, m, m, 0, 0, 0);

        // r := r * diag(sqrt(lambda_k))
        // rti := rti * diag(1 ./ sqrt(lambda_k))
        for (int i = 0; i < m; ++i) {
            double a = sqrt(MAT_BUFD(lmbda)[ind + i]);
            double tmp_a = 1.0 / a;
            blas_scal(&a, r, m, 1, m*i);
            blas_scal(&tmp_a, rti, m, 1, m*i);
        }
        ind += m;
        ind2 += m*m;
    }
    
    // Clean up temporary matrices
    Matrix_Free(work);
    Matrix_Free(Ls);
    Matrix_Free(Lz);
}


// KKT Inner solve function
void solve_function(matrix *x, matrix *y, matrix *z, KKTCholContext *ctx, DIMs *dims) 
{
    // Solve
    //
    //     [ 0          A'  GG'*W^{-1} ]   [ ux   ]   [ bx        ]
    //     [ A          0   0          ] * [ uy   ] = [ by        ]
    //     [ W^{-T}*GG  0   -I         ]   [ W*uz ]   [ W^{-T}*bz ]
    //
    // and return ux, uy, W*uz.
    //
    // On entry, x, y, z contain bx, by, bz.  On exit, they contain
    // the solution ux, uy, W*uz.
    //
    // If we change variables ux = Q1*v + Q2*w, the system becomes 
    // 
    //     [ K11 K12 R ]   [ v  ]   [Q1'*(bx+GG'*W^{-1}*W^{-T}*bz)]
    //     [ K21 K22 0 ] * [ w  ] = [Q2'*(bx+GG'*W^{-1}*W^{-T}*bz)]
    //     [ R^T 0   0 ]   [ uy ]   [by                           ]
    // 
    //     W*uz = W^{-T} * ( GG*ux - bz ).

    // bzp := W^{-T} * bz in packed storage 

    misc_scale(z, ctx->W, 'T', 'I'); 
    misc_pack(z, ctx->bzp, dims, ctx->mnl, 0, 0);    

    // x := [Q1, Q2]' * (x + Gs' * bzp)
    //    = [Q1, Q2]' * (bx + Gs' * W^{-T} * bz)
    double beta = 1.0;
    blas_gemv(ctx->Gs, ctx->bzp, x, 'T', NULL, &beta, ctx->cdim_pckd, -1, 0, 1, 1, 0, 0, 0);
    lapack_ormqr(ctx->QA, ctx->tauA, x, 'L', 'T', -1, -1, -1, 0, 0, 0, 0);

    // y := x[:p] 
    //    = Q1' * (bx + Gs' * W^{-T} * bz)
    blas_copy(y, ctx->yy, -1, 1, 1, 0, 0);
    blas_copy(x, y, ctx->p, 1, 1, 0, 0);

    // x[:p] := v = R^{-T} * by 
    blas_copy(ctx->yy, x, -1, 1, 1, 0, 0);
    lapack_trtrs(ctx->QA, x, 'U', 'T', 'N', ctx->p, -1, 0, 0, 0, 0);

    // x[p:] := K22^{-1} * (x[p:] - K21*x[:p])
    //        = K22^{-1} * (Q2' * (bx + Gs' * W^{-T} * bz) - K21*v)
    double alpha = -1.0;
    beta = 1.0;
    blas_gemv(ctx->K, x, x, 'N', &alpha, &beta, ctx->n - ctx->p, ctx->p, 0, 1, 1, ctx->p, 0, ctx->p);
    lapack_potrs(ctx->K, x, 'L', ctx->n - ctx->p, -1, 0, 0, ctx->p * (ctx->n+1), ctx->p);

    // y := y - [K11, K12] * x
    //    = Q1' * (bx + Gs' * W^{-T} * bz) - K11*v - K12*w
    alpha = -1.0;
    beta = 1.0;
    blas_gemv(ctx->K, x, y, 'N', &alpha, &beta, ctx->p, ctx->n, 0, 1, 1, 0, 0, 0);

    // y := R^{-1}*y
    //    = R^{-1} * (Q1' * (bx + Gs' * W^{-T} * bz) - K11*v 
    //      - K12*w)
    lapack_trtrs(ctx->QA, y, 'U', 'N', 'N', ctx->p, -1, 0, 0, 0, 0);

    // x := [Q1, Q2] * x
    lapack_ormqr(ctx->QA, ctx->tauA, x, 'L', 'N', -1, -1, -1, 0, 0, 0, 0);

    // bzp := Gs * x - bzp.
    //      = W^{-T} * ( GG*ux - bz ) in packed storage.
    // Unpack and copy to z.
    alpha = 1.0;
    beta = -1.0;

    blas_gemv(ctx->Gs, x, ctx->bzp, 'N', &alpha, &beta, ctx->cdim_pckd, -1, 0, 1, 1, 0, 0, 0);
    misc_unpack(ctx->bzp, z, dims, ctx->mnl, 0, 0);
}

// Inner factor function
void factor_function(scaling *W, matrix *H, matrix *Df, KKTCholContext *ctx, DIMs *dims) 
{
    // Compute 
    //
    //     K = [Q1, Q2]' * (H + GG' * W^{-1} * W^{-T} * GG) * [Q1, Q2]
    //
    // and take the Cholesky factorization of the 2,2 block
    //
    //     Q_2' * (H + GG^T * W^{-1} * W^{-T} * GG) * Q2.

    // Gs = W^{-T} * GG in packed storage.
    // if (ctx->W == NULL) {
    //     ctx->W = W;
    // }
    ctx->W = W;

    if (ctx->mnl) {
        matrix_slice_assign(ctx->Gs, Df, 0, ctx->mnl, 0, ctx->n);
    }

    if (Matrix_Check(ctx->G) && !SpMatrix_Check(ctx->G)) {
        // G is a dense matrix
        matrix_slice_assign(ctx->Gs, ctx->G, ctx->mnl, ctx->cdim, 0, ctx->n);
    } else if (SpMatrix_Check(ctx->G) && !Matrix_Check(ctx->G)) {
        // G is a sparse matrix
        spmatrix *spG = (spmatrix*)ctx->G;
        matrix *dense_G = dense(spG);
        matrix_slice_assign(ctx->Gs, dense_G, ctx->mnl, ctx->cdim, 0, ctx->n);
        Matrix_Free(dense_G);
    } else {
        ERR_TYPE("Invalid matrix type for G.");
    }

    misc_scale(ctx->Gs, W, 'T', 'I');
    misc_pack2(ctx->Gs, dims, ctx->mnl);

    // K = [Q1, Q2]' * (H + Gs' * Gs) * [Q1, Q2].
    blas_syrk(ctx->Gs, ctx->K, 'L', 'T', NULL, NULL, -1, ctx->cdim_pckd, 0, 0, 0, 0);

    if (H != NULL) {
        matrix_add(ctx->K, H);
    }
    misc_symm(ctx->K, ctx->n, 0);
    lapack_ormqr(ctx->QA, ctx->tauA, ctx->K, 'L', 'T', -1, -1, -1, 0, 0, 0, 0);
    lapack_ormqr(ctx->QA, ctx->tauA, ctx->K, 'R', 'N', -1, -1, -1, 0, 0, 0, 0);

    // Cholesky factorization of 2,2 block of K.
    lapack_potrf(ctx->K, 'L', ctx->n - ctx->p, 0, ctx->p * (ctx->n + 1));
    return;
}

// Main function
KKTCholContext* kkt_chol(void *G, DIMs *dims, void *A, int mnl) 
{
    /*
    Solution of KKT equations by reduction to a 2 x 2 system, a QR 
    factorization to eliminate the equality constraints, and a dense 
    Cholesky factorization of order n-p. 
    
    Computes the QR factorization
    
        A' = [Q1, Q2] * [R; 0]
    
    and returns a function that (1) computes the Cholesky factorization 
    
        Q_2^T * (H + GG^T * W^{-1} * W^{-T} * GG) * Q2 = L * L^T, 
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H    A'   GG'    ]   [ ux ]   [ bx ]
        [ A    0    0      ] * [ uy ] = [ by ].
        [ GG   0    -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    */

    // Allocate and initialize context
    KKTCholContext *ctx = malloc(sizeof(KKTCholContext));
    if (ctx == NULL) err_no_memory;
    KKTCholContext_Init(ctx);   // Initialize context structure

    ctx->G = G;
    ctx->mnl = mnl;
    ctx->cdim = mnl + dims->l + sum_array(dims->q, dims->q_size) 
                            + sum_square_array(dims->s, dims->s_size);
    ctx->cdim_pckd = mnl + dims->l + sum_array(dims->q, dims->q_size) 
                            + sum_triangular_array(dims->s, dims->s_size);

    // A' = [Q1, Q2] * [R; 0]  (Q1 is n x p, Q2 is n x n-p).
    if (Matrix_Check(A) && !SpMatrix_Check(A)) {
        matrix *A_mat = (matrix*)A;
        ctx->p = A_mat->nrows;  // A is a dense matrix
        ctx->n = A_mat->ncols;
        ctx->QA = matrix_transpose(A_mat);
    } else if (SpMatrix_Check(A) && !Matrix_Check(A)) {
        // A is a sparse matrix
        spmatrix *spA = (spmatrix*)A;
        ctx->p = spA->obj->nrows;
        ctx->n = spA->obj->ncols;

        spmatrix *spA_trans = spmatrix_trans(spA);
        ctx->QA = dense(spA_trans);
        SpMatrix_Free(spA_trans);
        
    } else {
        free(ctx);
        ERR_TYPE("Invalid matrix type for A.");
    }

    ctx->tauA = Matrix_New(ctx->p, 1, DOUBLE);
    lapack_geqrf(ctx->QA, ctx->tauA, -1, -1, 0, 0);

    ctx->Gs = Matrix_New(ctx->cdim, ctx->n, DOUBLE);
    ctx->K = Matrix_New(ctx->n, ctx->n, DOUBLE);
    ctx->bzp = Matrix_New(ctx->cdim_pckd, 1, DOUBLE);
    ctx->yy = Matrix_New(ctx->p, 1, DOUBLE);

    // Return factory function that captures context
    return ctx;
}



/**
 * Copy x to y using packed storage.
 *
 * The vector x is an element of S, with the 's' components stored in
 * unpacked storage.  On return, x is copied to y with the 's' 
 * components stored in packed storage and the off-diagonal entries 
 * scaled by sqrt(2).
 *
 * @param x        Input matrix
 * @param y        Output matrix  
 * @param dims     Cone dimensions structure
 * @param mnl      Additional offset for nonlinear variables (default 0)
 * @param offsetx  Offset in x (default 0)
 * @param offsety  Offset in y (default 0)
 */

void misc_pack(matrix *x, matrix *y, DIMs *dims, int mnl, int offsetx, int offsety)
{
    double a;
    int i, k, nlq = mnl, ox = offsetx, oy = offsety, np, iu, ip, int1 = 1, len, n;

    // Default values
    if (mnl < 0) nlq = 0;  
    if (offsetx < 0) ox = 0; 
    if (offsety < 0) oy = 0; 

    // Calculate nlq by adding 'l' dimensions
    nlq += dims->l;

    // Add all 'q' dimensions
    for (i = 0; i < dims->q_size; ++i){
        nlq += dims->q[i];
    }

    // Copy the linear and quadratic parts
    dcopy_(&nlq, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

    // Process semidefinite blocks
    for (i = 0, np = 0, iu = ox + nlq, ip = oy + nlq; i < dims->s_size; ++i){
        n = dims->s[i];
        for (k = 0; k < n; k++){
            len = n-k;
            dcopy_(&len, MAT_BUFD(x) + iu + k*(n+1), &int1,  MAT_BUFD(y) +
                ip, &int1);
            MAT_BUFD(y)[ip] /= sqrt(2.0);
            ip += len;
        }
        np += n*(n+1)/2;
        iu += n*n;
    }

    // Scale the packed semidefinite part
    a = sqrt(2.0);
    dscal_(&np, &a, MAT_BUFD(y) + oy + nlq, &int1);

    return;
}


/**
 * In-place version of pack().
 *
 * misc_pack2(x, dims, mnl = 0)
 *
 * In-place version of pack(), which also accepts matrix arguments x.
 * The columns of x are elements of S, with the 's' components stored
 * in unpacked storage.  On return, the 's' components are stored in
 * packed storage and the off-diagonal entries are scaled by sqrt(2).
 *
 * @param x    Matrix with 's' components in unpacked storage
 * @param dims Cone dimensions structure
 * @param mnl  Additional offset (default = 0)
 */
void misc_pack2(matrix *x, DIMs *dims, int mnl)
{
    double a = sqrt(2.0), *wrk;
    int i, j, k, nlq = mnl, iu, ip, len, n, maxn, xr, xc;

    // Default values
    if (mnl < 0) nlq = 0;  

    // Get matrix dimensions
    xr = x->nrows;
    xc = x->ncols;

    // Calculate nlq by adding 'l' dimension
    nlq += dims->l;

    // Add all 'q' dimensions
    for (i = 0; i < dims->q_size; i++){
        nlq += dims->q[i];
    }

    // Find maximum 's' dimension
    maxn = 0;
    for (i = 0; i < dims->s_size; i++){
        maxn = MAX(maxn, dims->s[i]);
    }
    
    // If no 's' blocks, return immediately
    if (!maxn) return;

    // Allocate workspace
    if (!(wrk = (double *) calloc(maxn * xc, sizeof(double))))
        err_no_memory;

    for (i = 0, iu = nlq, ip = nlq; i < dims->s_size; ++i){
        n = dims->s[i];

        // For each column k in the n x n block
        for (k = 0; k < n; k++){
            len = n-k;
            dlacpy_(" ", &len, &xc, MAT_BUFD(x) + iu + k*(n+1), &xr, wrk, 
                &maxn);
            for (j = 1; j < len; j++)
                dscal_(&xc, &a, wrk + j, &maxn);
            dlacpy_(" ", &len, &xc, wrk, &maxn, MAT_BUFD(x) + ip, &xr);
            ip += len;
        }
        iu += n*n;
    }
    free(wrk);
    return;
}


/**
 * @brief Unpacks x into y
 * 
 * @param x Input matrix - element of S with 's' components stored in packed storage 
 *          and off-diagonal entries scaled by sqrt(2)
 * @param y Output matrix - will contain x with 's' components stored in unpacked storage
 * @param dims Cone dimensions structure containing 'l', 'q', and 's' dimensions
 * @param mnl Number of nonlinear variables (default = 0)
 * @param offsetx Offset in x vector (default = 0)  
 * @param offsety Offset in y vector (default = 0)
 * 
 * @details
 * The vector x is an element of S, with the 's' components stored in
 * packed storage and off-diagonal entries scaled by sqrt(2).
 * On return, x is copied to y with the 's' components stored in
 * unpacked storage.
 * 
 * Processing order:
 * 1. Copy linear ('l') and quadratic ('q') components directly
 * 2. For semidefinite ('s') components:
 *    - Unpack from packed storage to full matrix storage
 *    - Scale off-diagonal entries by 1/sqrt(2)
 *    - Maintain lower triangular structure
 */
void misc_unpack(matrix *x, matrix *y, DIMs *dims, int mnl, int offsetx, int offsety)
{
    double a = 1.0 / sqrt(2.0);
    int m = mnl, ox = offsetx, oy = offsety, int1 = 1, iu, ip, len, i, k, n;

    // Default values
    if (mnl < 0) m = 0;  
    if (offsetx < 0) ox = 0; 
    if (offsety < 0) oy = 0; 

    /* Calculate total size of linear and quadratic parts */
    m += dims->l;
    
    /* Add dimensions of all quadratic cones */
    for (i = 0; i < dims->q_size; i++){
        m += dims->q[i];
    }

    /* Copy linear and quadratic parts directly */
    dcopy_(&m, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

    /* Process semidefinite cones */
    /* ip: current position in packed storage (input x) */
    /* iu: current position in unpacked storage (output y) */
    for (i = 0, ip = ox + m, iu = oy + m; i < dims->s_size; ++i){
        n = dims->s[i];

        /* Unpack each column of the semidefinite block */
        for (k = 0; k < n; ++k){
            len = n-k;
            dcopy_(&len, MAT_BUFD(x) + ip, &int1, MAT_BUFD(y) + iu +
                k*(n+1), &int1);
            ip += len;
            len -= 1;
            dscal_(&len, &a, MAT_BUFD(y) + iu + k*(n+1) + 1, &int1);
        }
        iu += n*n;
    }
    return;
}


/**
 * @brief Matrix-vector multiplication for structured matrices.
 *
 * @details
 * A is a matrix or spmatrix of size (m, n) where 
 * 
 *     N = dims->l + sum(dims->q) + sum( k^2 for k in dims->s ) 
 *
 * representing a mapping from R^n to S.  
 * 
 * If trans is 'N': 
 * 
 *     y := alpha*A*x + beta * y   (trans = 'N').
 * 
 * x is a vector of length n.  y is a vector of length N.
 * 
 * If trans is 'T':
 * 
 *     y := alpha*A'*x + beta * y  (trans = 'T').
 * 
 * x is a vector of length N.  y is a vector of length n.
 * 
 * The 's' components in S are stored in unpacked 'L' storage.
 *
 * @param[in] A         Input matrix or spmatrix
 * @param[in] x         Input vector
 * @param[in,out] y     Input/output vector
 * @param[in] dims      Dimensions structure containing l, q, and s arrays
 * @param[in] trans     Transposition flag ('N' or 'T') (default = 'N')
 * @param[in] alpha     Scalar multiplier (default = 1.0)
 * @param[in] beta      Scalar multiplier (default = 0.0)
 * @param[in] n         Column dimension of A (default = -1, which means A.size[1])
 * @param[in] offsetA   Matrix A offset (default = 0)
 * @param[in] offsetx   Vector x offset (default = 0)
 * @param[in] offsety   Vector y offset (default = 0)
 */
void misc_sgemv(void *A, matrix *x, matrix *y, DIMs *dims, char trans, double alpha, 
                double beta, int n, int offsetA, int offsetx, int offsety)
{
    int m, i;
    int sum_q = 0, sum_s = 0;

    if (!A || !x || !y || !dims) ERR("Invalid input parameters.\n");
    if (!is_matrix(A) && !is_spmatrix(A)) ERR("A must be a matrix or spmatrix.\n");

    // Default values
    if (trans == 0) trans = 'N';
    
    // Compute sum of dims->q array
    for (i = 0; i < dims->q_size; ++i) {
        sum_q += dims->q[i];
    }
    
    // Compute sum of k^2 for k in dims->s array
    for (i = 0; i < dims->s_size; ++i) {
        sum_s += dims->s[i] * dims->s[i];
    }
    
    // Calculate m = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] )
    m = dims->l + sum_q + sum_s;
    
    // Default value for n if not provided (n = A.size[1])
    if (n < 0) {
        if (is_matrix(A)) {
            n = ((matrix *)A)->ncols;
        } else if (is_spmatrix(A)) {
            n = SP_NCOLS(A);
        }
    }
    
    // If trans is 'T' and alpha is non-zero, call trisc
    if (trans == 'T' && alpha != 0.0) {
        misc_trisc(x, dims, offsetx);
    }
    
    // Call base.gemv with the computed parameters
    base_gemv(A, x, y, trans, &alpha, &beta, m, n, 1, 1, offsetA, offsetx, offsety);
    
    // If trans is 'T' and alpha is non-zero, call triusc
    if (trans == 'T' && alpha != 0.0) {
        misc_triusc(x, dims, offsetx);
    }
}


/**
 * @brief Modifies the input matrix by zeroing its upper triangular 's' components and scaling the strictly lower triangular part.
 * 
 * This function processes the matrix `x` by:
 * 1. Setting the upper triangular part of each 's' block to zero.
 * 2. Scaling the strictly lower triangular part by 2.
 * 
 * @param[in,out] x The matrix to be modified. Must be a valid `matrix` object.
 * @param[in] dims A `DIMs` structure containing:
 *   - `l` (linear part size)
 *   - `q` (array of SOCP cone sizes)
 *   - `s` (array of SDP cone sizes)
 * @param[in] offset Starting offset for processing (default = 0 if negative).
 * 
 * @note The function modifies `x` in-place.
 * @warning The input matrix `x` must have sufficient size to accommodate the dimensions specified in `dims`.
 */
void misc_trisc(matrix *x, DIMs *dims, int offset)
{
    double dbl0 = 0.0, dbl2 = 2.0;
    int ox = offset, i, k, nk, len, int1 = 1;

    // Default value for offset
    if (ox < 0) ox = 0;

    // Add the 'l' component to offset
    ox += dims->l;

    // Add all 'q' components to offset
    for (i = 0; i < dims->q_size; i++){
        ox += dims->q[i];
    }

    for (k = 0; k < dims->s_size; k++){
        nk = dims->s[k];
        for (i = 1; i < nk; i++){
            len = nk - i;
            dscal_(&len, &dbl0, MAT_BUFD(x) + ox + i*(nk+1) - 1, &nk);
            dscal_(&len, &dbl2, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
        }
        ox += nk*nk;
    }
    return;
}


/**
 * @brief Documentation string for the triusc function.
 * 
 * Scales the strictly lower triangular part of the 's' components of x by 0.5.
 * 
 * @par Usage:
 * @code
 * triusc(x, dims, offset = 0)
 * @endcode
 * 
 * @param x The input matrix to be modified.
 * @param dims A structure containing dimensions (l, q, s).
 * @param offset Starting index for processing (default = 0).
 */
void misc_triusc(matrix *x, DIMs *dims, int offset)
{
    double dbl5 = 0.5;
    int ox = offset, i, k, nk, len, int1 = 1;

    // Default value for offset
    if (ox < 0) ox = 0;

    // Add the 'l' component to offset
    ox += dims->l;

    // Add all 'q' components to offset
    for (i = 0; i < dims->q_size; i++){
        ox += dims->q[i];
    }

    for (k = 0; k < dims->s_size; k++){
        nk = dims->s[k];
        for (i = 1; i < nk; i++){
            len = nk - i;
            dscal_(&len, &dbl5, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
        }
        ox += nk*nk;
    }
    return;
}


/**
 * The inverse of the product x := (y o\ x) when the 's' components of 
 * y are diagonal.
 * 
 * misc_sinv(x, y, dims, mnl = 0)
 */
void misc_sinv(matrix *x, matrix *y, DIMs *dims, int mnl)
{
    int i, j, k, mk, len, maxn, ind = mnl, ind2, int0 = 0, int1 = 1;
    double a, c, d, alpha, *A = NULL, dbl2 = 0.5;

    if (!x || !y || !dims) return;

    if (mnl < 0) mnl = 0;

    /*
     * For nonlinear and 'l' blocks:
     *
     *     yk o\ xk = yk .\ xk
     */

    ind += dims->l;
    dtbsv_("L", "N", "N", &ind, &int0, MAT_BUFD(y), &int1, MAT_BUFD(x),
        &int1);


    /*
     * For 'q' blocks:
     *
     *                        [  l0   -l1'               ]
     *     yk o\ xk = 1/a^2 * [                          ] * xk
     *                        [ -l1    (a*I + l1*l1')/l0 ]
     *
     * where yk = (l0, l1) and a = l0^2 - l1'*l1.
     */

    for (i = 0; i < dims->q_size; i++){
        mk = dims->q[i];
        len = mk - 1;
        a = dnrm2_(&len, MAT_BUFD(y) + ind + 1, &int1);
        a = (MAT_BUFD(y)[ind] + a) * (MAT_BUFD(y)[ind] - a);
        c = MAT_BUFD(x)[ind];
        d = ddot_(&len, MAT_BUFD(x) + ind + 1, &int1,
            MAT_BUFD(y) + ind + 1, &int1);
        MAT_BUFD(x)[ind] = c * MAT_BUFD(y)[ind] - d;
        alpha = a / MAT_BUFD(y)[ind];
        dscal_(&len, &alpha, MAT_BUFD(x) + ind + 1, &int1);
        alpha = d / MAT_BUFD(y)[ind] - c;
        daxpy_(&len, &alpha, MAT_BUFD(y) + ind + 1, &int1, MAT_BUFD(x) +
            ind + 1, &int1);
        alpha = 1.0 / a;
        dscal_(&mk, &alpha, MAT_BUFD(x) + ind, &int1);
        ind += mk;
    }


    /*
     * For the 's' blocks:
     *
     *    yk o\ sk = xk ./ gamma
     *
     * where  gammaij = .5 * (yk_i + yk_j).
     */

    for (i = 0, maxn = 0; i < dims->s_size; i++){
        maxn = MAX(maxn, dims->s[i]);
    }

    if (!(A = (double *) calloc(maxn, sizeof(double)))) err_no_memory;
    for (i = 0, ind2 = ind; i < dims->s_size; ind += mk*mk,
        ind2 += mk, i++){
        mk = dims->s[i];
        for (k = 0; k < mk; k++){
            len = mk - k;
            dcopy_(&len, MAT_BUFD(y) + ind2 + k, &int1, A, &int1);
            for (j = 0; j < len; j++) A[j] += MAT_BUFD(y)[ind2 + k];
            dscal_(&len, &dbl2, A, &int1);
            dtbsv_("L", "N", "N", &len, &int0, A, &int1, MAT_BUFD(x) + ind
                + k * (mk+1), &int1);
        }
    }

    free(A);
    return;
}