#include "solver.h"
#include "cvxopt.h"
#include "misc.h"

ECVXConeSettings ecvxcone_settings = {
    .debug = false,
    .kktreg = -1.0, // -1 indicates None/unset
    .maxiters = 100,
    // .abstol = 1e-7,  // Original default values
    // .reltol = 1e-6,  // Original default values
    // .feastol = 1e-7, // Original default values
    .abstol = 1e-2, // Adjusted for better solving time
    .reltol = 1e-2, // Adjusted for better solving time
    .feastol = 1e-3, // Adjusted for better solving time
    .show_progress = false,
    .refinement = -1, // -1 indicates None/unset
    .EXPON = 3,
    .STEP = 0.99,
    .kktsolver = "chol" // Default KKT solver
};

ECVXConeData ecvxcone_data = {
    .c = NULL,
    .b = NULL,
    .h = NULL,
    .G = NULL,
    .A = NULL
};

void validate_ecvxcone_settings(DIMs *dims, ECVXConeSettings *stgs);
void validate_problem_data(matrix *c, void *G, matrix *h, void *A, matrix *b, int cdim);
void validate_cone_dimensions(DIMs* dims);
void validate_kktsolver(DIMs* dims, const char* kktsolver);
extern void init_solver_matrices(matrix *c, matrix *b, int cdim, int cdim_diag, int sum_dims_s);
ECVXConeWorkspace *ECVXConeWorkspace_Init(matrix *c, void *G, matrix *h, void *A, matrix *b, 
                                        PrimalStart *primalstart, DualStart *dualstart, DIMs *dims);
ECVXConeWorkspace* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, 
                                ECVXConeSettings* settings);
scaling *init_identity_scaling(DIMs *dims);

/**
 * Initialize the CVXConeResult structure.
 * This function sets all fields of the result structure to their initial values.
 *
 * @param result Pointer to the CVXConeResult structure to initialize.
 */
ECVXConeWorkspace* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, 
                                ECVXConeSettings* settings) 
{
    ECVXConeWorkspace *ecvxcone_ws = ECVXConeWorkspace_Init(c, G, h, A, b, NULL, NULL, dims);
    init_solver_matrices(c, b, ecvxcone_ws->cdim, ecvxcone_ws->cdim_diag, ecvxcone_ws->sum_dims_s);
    
    validate_cone_dimensions(dims); // Validate the cone dimensions
    validate_ecvxcone_settings(dims, settings);    // Validate the settings
    validate_problem_data(c, G, h, A, b, ecvxcone_ws->cdim); // Validate the problem data

    return ecvxcone_ws;
}

/**
 * Validate the settings for ECVXCONE.
 * This function checks if the settings are valid and raises errors if they are not.
 *
 * @param stgs Pointer to the ECVXConeSettings structure to validate.
 */
void validate_ecvxcone_settings(DIMs *dims, ECVXConeSettings *stgs) 
{
    if (stgs->kktreg != -1.0) { // -1.0 represents None/unset
        if (stgs->kktreg < 0.0) {
            ERR("options['kktreg'] must be a nonnegative scalar");
        }
    }

    if (stgs->maxiters < 1) {    // Times of iterations
        ERR("options['maxiters'] must be a positive integer");
    }

    if (stgs->reltol <= 0.0 && stgs->abstol <= 0.0) {   // Relative and absolute tolerance
        ERR("at least one of options['reltol'] and options['abstol'] must be positive");
    }

    // Set default refinement
    if (stgs->refinement == -1) { // -1 represents None/unset
        if (dims->q_size > 0 || dims->s_size > 0) {
            stgs->refinement = 1;
        } else {
            stgs->refinement = 0;
        }
    } else if (stgs->refinement < 0) {
        ERR_TYPE( "options['refinement'] must be a nonnegative integer");
    }

    // Validate kktsolver
    validate_kktsolver(dims, stgs->kktsolver);
}

/**
 * Validate the cone dimensions.
 * This function checks if the dimensions of the cones are valid and raises errors if they are not.
 *
 * @param dims Pointer to the DIMs structure containing the cone dimensions.
 */
void validate_kktsolver(DIMs* dims, const char* kktsolver) 
{
    // Default solver selection
    char* default_kktsolver = NULL;
    if (kktsolver == NULL) {
        if (dims && (dims->q_size > 0 || dims->s_size > 0)) {
            default_kktsolver = "qr";
        } else {
            default_kktsolver = "chol2";
        }
        kktsolver = default_kktsolver;
    }
    
    // Check if kktsolver is one of the default string solvers
    bool is_string_solver = false;
    if (kktsolver && ((char*)kktsolver)[0] != '\0') { // Assume string if not function pointer
        for (int i = 0; i < 1; ++i) {
            if (strcmp((char*)kktsolver, defaultsolvers[i]) == 0) {
                is_string_solver = true;
                break;
            }
        }
        if (!is_string_solver) {
            snprintf(msg, sizeof(msg), "'%s' is not supported for kktsolver", (char*)kktsolver);
            ERR(msg);
        }
    }
    if (kktsolver == NULL || strlen(kktsolver) == 0) {
        ERR("kktsolver must be a non-empty string");
    }
}

/**
 * Validate the problem data for ConeLP.
 * This function checks if the provided matrices and dimensions are valid.
 *
 * @param c Coefficient matrix for the linear term.
 * @param G Coefficient matrix for the constraints.
 * @param h Right-hand side vector.
 * @param A Coefficient matrix for the constraints.
 * @param b Right-hand side vector for the constraints.
 * @param cdim Cone dimension.
 */
void validate_problem_data(matrix *c, void *G, matrix *h, void *A, matrix *b, int cdim) 
{
    // Validate c
    if (!Matrix_Check(c) || c->id != DOUBLE || c->ncols != 1) {
        ERR_TYPE("'c' must be a 'd' matrix with one column");
    }

    // Validate h 
    if (!Matrix_Check(h) || h->id != DOUBLE || h->ncols != 1) {
        ERR_TYPE("'h' must be a 'd' matrix with one column");
    }
    if (h->nrows != cdim) {
        snprintf(msg, sizeof(msg), "Error: 'h' must be a 'd' matrix of size (%d,1)", cdim);
        ERR_TYPE(msg);
    }

    // Validate G
    bool matrixG = Matrix_Check(G) || SpMatrix_Check(G);
    bool matrixA = Matrix_Check(A) || SpMatrix_Check(A);

    if ((!matrixG || (!matrixA && A != NULL))) {
        ERR("use of function valued G, A requires a user-provided kktsolver");
    }

    if (matrixG) {
        if (Matrix_Check(G) && !SpMatrix_Check(G)) {
            // G is a dense matrix
            matrix *G_mat = (matrix *)G;
            if (G_mat->id != DOUBLE || G_mat->nrows != cdim || G_mat->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'G' must be a 'd' matrix of size (%d, %d)", cdim, c->nrows);
                ERR_TYPE(msg);
            }
        } else if (SpMatrix_Check(G) && !Matrix_Check(G)) {
            // G is a sparse matrix
            spmatrix *G_sp = (spmatrix *)G;
            if (G_sp->obj->id != DOUBLE || G_sp->obj->nrows != cdim || G_sp->obj->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'G' must be a 'd' matrix of size (%d, %d)", cdim, c->nrows);
                ERR_TYPE(msg);
            }
        }
    } 

    // Validate A
    if (A == NULL) {
        // Create empty sparse matrix
        A = SpMatrix_New(0, c->nrows, 0, DOUBLE);
        matrixA = true;
    }
    
    if (matrixA) {
        if (Matrix_Check(A) && !SpMatrix_Check(A)) {
            // A is a dense matrix
            matrix *A_mat = (matrix *)A;
            if (A_mat->id != DOUBLE || A_mat->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'A' must be a 'd' matrix with %d columns", c->nrows);
                ERR_TYPE(msg);
            }
        } else if (SpMatrix_Check(A) && !Matrix_Check(A)) {
            // A is a sparse matrix
            spmatrix *A_sp = (spmatrix *)A;
            if (A_sp->obj->id != DOUBLE  || A_sp->obj->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'A' must be a 'd' matrix with %d columns", c->nrows);
                ERR_TYPE(msg);
            }
        }
    } 

    // Validate b
    if (b == NULL) {
        b = Matrix_New(0, 1, DOUBLE); // Create an empty vector
    }
    if (!Matrix_Check(b) || b->id != DOUBLE || b->ncols != 1) {
        snprintf(msg, sizeof(msg), "Error: 'b' must be a 'd' matrix with one column");
        ERR_TYPE(msg);
    }

    int A_nrows = 0;
    if (Matrix_Check(A) && !SpMatrix_Check(A)) {
        // A is a dense matrix
        matrix *A_mat = (matrix *)A;
        A_nrows = A_mat->nrows;
    } else if (SpMatrix_Check(A) && !Matrix_Check(A)) {
        // A is a sparse matrix
        spmatrix *A_sp = (spmatrix *)A;
        A_nrows = A_sp->obj->nrows;
    }

    if (matrixA && b->nrows != A_nrows) {
        snprintf(msg, sizeof(msg), "Error: 'b' must have length %d", A_nrows);
        ERR_TYPE(msg);
    }
    
}

/**
 * Validate the dimensions of the cone programming problem.
 * This function checks if the dimensions are valid and raises errors if they are not.
 *
 * @param dims Pointer to the DIMs structure containing the dimensions.
 */
void validate_cone_dimensions(DIMs* dims) 
{
// Validate dims structure
    if (dims->l < 0) {
        ERR_TYPE("'dims['l']' must be a nonnegative integer");
    }
    
    // Check q dimensions
    for (int k = 0; k < dims->q_size; ++k) {
        if (dims->q[k] < 1) {
            ERR_TYPE("'dims['q']' must be a list of positive integers");
        }
    }
    
    // Check s dimensions
    for (int k = 0; k < dims->s_size; ++k) {
        if (dims->s[k] < 0) {
            ERR_TYPE("'dims['s']' must be a list of nonnegative integers");
        }
    }
}

/**
 * Initialize the scaling structure with identity matrices.
 * This function sets up the scaling structure for initialization and iterations.
 *
 * @param dims Pointer to the DIMs structure containing the dimensions.
 * @return Pointer to the initialized scaling structure.
 */
scaling *init_identity_scaling(DIMs *dims) 
{
    scaling *W_init = malloc(sizeof(scaling));
    Scaling_Init(W_init);

    number number_one;
    number_one.d = 1.0; // Initialize number for value 1.0
    W_init->d = Matrix_New_Val(dims->l, 1, DOUBLE, number_one);
    W_init->di = Matrix_New_Val(dims->l, 1, DOUBLE, number_one);
    W_init->d_count = dims->l;

    W_init->v = malloc(dims->q_size * sizeof(matrix*));
    W_init->beta = malloc(dims->q_size * sizeof(double));
    W_init->v_count = dims->q_size;
    W_init->b_count = dims->q_size;
    for (int i = 0; i < dims->q_size; ++i) {
        W_init->v[i] = Matrix_New(dims->q[i], 1, DOUBLE);
        W_init->beta[i] = 1.0;
        MAT_BUFD(W_init->v[i])[0] = 1.0;
    }

    W_init->r = malloc(dims->s_size * sizeof(matrix*));
    W_init->rti = malloc(dims->s_size * sizeof(matrix*));
    W_init->r_count = dims->s_size;
    for (int i = 0; i < dims->s_size; ++i) {
        int m = dims->s[i];
        W_init->r[i] = Matrix_New(m, m, DOUBLE);
        W_init->rti[i] = Matrix_New(m, m, DOUBLE);
        for (int j = 0; j < m; ++j) {
            MAT_BUFD(W_init->r[i])[j * m + j] = 1.0;
            MAT_BUFD(W_init->rti[i])[j * m + j] = 1.0;
        }
    }
    return W_init;
}


/**
 * Initialize the ECVXConeWorkspace structure.
 * This function sets up the workspace with the provided data and dimensions.
 *
 * @param c Coefficient matrix for the linear term.
 * @param G Coefficient matrix for the constraints.
 * @param h Right-hand side vector.
 * @param A Coefficient matrix for the constraints.
 * @param b Right-hand side vector for the constraints.
 * @param primalstart Pointer to the primal start values.
 * @param dualstart Pointer to the dual start values.
 * @param dims Pointer to the DIMs structure containing the dimensions.
 * 
 * @return Pointer to the initialized ECVXConeWorkspace structure.
 */
ECVXConeWorkspace *ECVXConeWorkspace_Init(matrix *c, void *G, matrix *h, void *A, matrix *b, 
                                        PrimalStart *primalstart, DualStart *dualstart, DIMs *dims) 
{
    ECVXConeWorkspace *ecvxcone_ws = (ECVXConeWorkspace*)malloc(sizeof(ECVXConeWorkspace));
    if (!ecvxcone_ws) err_no_memory;

    ECVXConeData *data = (ECVXConeData*)malloc(sizeof(ECVXConeData));
    if (!data) err_no_memory;
    data->c = c;
    data->G = G;
    data->h = h;
    data->A = A;
    data->b = b;   
    ecvxcone_ws->data = data;

    ECVXConeResult *result = ECVXConeResult_Init(); 
    if (!result) err_no_memory;

    ecvxcone_ws->dims = dims;
    ecvxcone_ws->result = result;

    ecvxcone_ws->primalstart = primalstart;
    ecvxcone_ws->dualstart = dualstart;

    ecvxcone_ws->sum_dims_q = sum_array(dims->q, dims->q_size);
    ecvxcone_ws->sum_dims_s = sum_array(dims->s, dims->s_size);

    // Calculate cone dimensions
    ecvxcone_ws->cdim = dims->l;
    ecvxcone_ws->cdim_pckd = dims->l;
    ecvxcone_ws->cdim_diag = dims->l;

    for (int i = 0; i < dims->q_size; ++i) {
        ecvxcone_ws->cdim += dims->q[i];
        ecvxcone_ws->cdim_pckd += dims->q[i];
        ecvxcone_ws->cdim_diag += dims->q[i];
    }

    for (int i = 0; i < dims->s_size; ++i) {
        ecvxcone_ws->cdim += dims->s[i] * dims->s[i];
        ecvxcone_ws->cdim_pckd += dims->s[i] * (dims->s[i] + 1) / 2;
        ecvxcone_ws->cdim_diag += dims->s[i];
    }

    // Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G
    ecvxcone_ws->indq = (int*)malloc((dims->q_size + 1) * sizeof(int));
    if (ecvxcone_ws->indq == NULL) err_no_memory;
    ecvxcone_ws->indq[0] = dims->l;
    for (int k = 0; k < dims->q_size; ++k) {
        ecvxcone_ws->indq[k + 1] = ecvxcone_ws->indq[k] + dims->q[k];
    }
    
    // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G
    ecvxcone_ws->inds = (int*)malloc((dims->s_size + 1) * sizeof(int));
    if (ecvxcone_ws->inds == NULL) err_no_memory;
    ecvxcone_ws->inds[0] = ecvxcone_ws->indq[dims->q_size];
    for (int k = 0; k < dims->s_size; ++k) {
        ecvxcone_ws->inds[k + 1] = ecvxcone_ws->inds[k] + dims->s[k] * dims->s[k];
    }

    // Initialize scaling structure
    if (primalstart == NULL || dualstart == NULL) {
        ecvxcone_ws->W_init = init_identity_scaling(dims);
    } else {
        ecvxcone_ws->W_init = NULL; // No dummy scaling structure needed
    }

    return ecvxcone_ws;
}