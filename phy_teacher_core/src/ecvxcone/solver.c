#include "cvxopt.h"
#include "misc.h"     
#include "solver.h"
#include <math.h>

void f6(matrix* x_, matrix* y_, matrix* z_, matrix* tau, matrix* s_, matrix* kappa, 
        matrix *c, matrix *b, matrix *h, void *G, void *A, DIMs *dims,  scaling *W, 
        KKTCholContext *kkt_ctx, int cdim_diag, double dg, double dgi, int iters, 
        int REFINEMENT, int DEBUG);
void f6_no_ir(matrix* x_, matrix* y_, matrix* z_, matrix* tau, matrix* s_, matrix* kappa, matrix *c, 
              matrix *b, DIMs *dims, scaling *W, KKTCholContext *kkt_ctx, int cdim_diag, double dgi);
void res(matrix *ux, matrix *uy, matrix *uz, matrix *utau, matrix *us, matrix *ukappa, 
         matrix *vx, matrix *vy, matrix *vz, matrix *vtau, matrix *vs, matrix *vkappa, 
         matrix *c, matrix *b, matrix *h, void* G, void* A, DIMs *dims, scaling *W, 
         double dg, int cdim_diag);
void Af_gemv(matrix *x_, matrix *y_, void *A, char trans, void* alpha, void* beta);
void Gf_gemv(matrix *x_, matrix *y_, void *G, DIMs *dims, char trans, void* alpha, void* beta); 
void xy_copy(matrix *x, matrix *y);

// const char* defaultsolvers[] = {"ldl", "ldl2", "qr", "chol", "chol2"};
const char* defaultsolvers[] = {"chol"};
char msg[256];

// Primal and dual variables
matrix *x = NULL;     // Primal variable (decision variable)
matrix *y = NULL;     // Dual variable for equality constraints (Lagrange multiplier for Ax = b)
matrix *s = NULL;     // Slack variable for inequality constraints (s ≥ 0 or s ∈ cone)
matrix *z = NULL;     // Dual variable for inequality constraints (associated with cone dual K*)

// Newton direction variables (used during each iteration step)
matrix *dx = NULL;    // Search direction for x
matrix *dy = NULL;    // Search direction for y
matrix *ds = NULL;    // Search direction for s
matrix *dz = NULL;    // Search direction for z

// Extended variables (used in self-dual embedding or infeasibility detection)
matrix *dkappa = NULL;  // Search direction for κ (helps in feasibility or regularization)
matrix *dtau = NULL;    // Search direction for τ (scaling or feasibility tracking)

// Intermediate matrices for computations
static matrix *rx = NULL;
static matrix *hrx = NULL;
static matrix *ry = NULL;
static matrix *hry = NULL;
static matrix *rz = NULL;
static matrix *hrz = NULL;
static matrix *sigs = NULL;
static matrix *sigz = NULL;
static matrix *lmbda = NULL;
static matrix *lmbdasq = NULL;

static matrix *x1 = NULL;
static matrix *y1_ = NULL;     // y1_ is used to avoid name conflict with y1 in math.h
static matrix *z1 = NULL;
static matrix *th = NULL;

static matrix *wx = NULL;
static matrix *wy = NULL;
static matrix *wz = NULL;
static matrix *ws = NULL;
static matrix *wtau = NULL;
static matrix *wkappa = NULL;

static matrix *wx2 = NULL;
static matrix *wy2 = NULL;
static matrix *wz2 = NULL;
static matrix *ws2 = NULL;
static matrix *wtau2 = NULL;
static matrix *wkappa2 = NULL;

static matrix *wz3 = NULL;
static matrix *ws3 = NULL;

void init_solver_matrices(matrix *c, matrix *b, int cdim, int cdim_diag, 
                          int sum_dims_s, int REFINEMENT, int DEBUG) 
{
    // Initialize matrices for the solver
    x = Matrix_NewFromMatrix(c, c->id);
    y = Matrix_NewFromMatrix(b, b->id);
    s = Matrix_New(cdim, 1, DOUBLE);
    z = Matrix_New(cdim, 1, DOUBLE);
    dx = Matrix_NewFromMatrix(c, c->id);
    dy = Matrix_NewFromMatrix(b, b->id);
    ds = Matrix_New(cdim, 1, DOUBLE);
    dz = Matrix_New(cdim, 1, DOUBLE);
    dkappa = Matrix_New(1, 1, DOUBLE);
    dtau = Matrix_New(1, 1, DOUBLE);

    // Initialize intermediate matrices
    rx = Matrix_NewFromMatrix(c, c->id);
    hrx = Matrix_NewFromMatrix(c, c->id);
    ry = Matrix_NewFromMatrix(b, b->id);
    hry = Matrix_NewFromMatrix(b, b->id);
    rz = Matrix_New(cdim, 1, DOUBLE);
    hrz = Matrix_New(cdim, 1, DOUBLE);
    sigs = Matrix_New(sum_dims_s, 1, DOUBLE);
    sigz = Matrix_New(sum_dims_s, 1, DOUBLE);
    lmbda = Matrix_New(cdim_diag + 1, 1, DOUBLE);
    lmbdasq = Matrix_New(cdim_diag + 1, 1, DOUBLE);

    wz3 = Matrix_New(cdim, 1, DOUBLE);
    ws3 = Matrix_New(cdim, 1, DOUBLE);

    x1 = Matrix_NewFromMatrix(c, c->id);
    y1_ = Matrix_NewFromMatrix(b, b->id);
    z1 = Matrix_New(cdim, 1, DOUBLE);
    th = Matrix_New(cdim, 1, DOUBLE);

    if (REFINEMENT || DEBUG) {
        wx = Matrix_NewFromMatrix(c, c->id);
        wy = Matrix_NewFromMatrix(b, b->id);
        wz = Matrix_New(cdim, 1, DOUBLE);
        ws = Matrix_New(cdim, 1, DOUBLE);
        wtau = Matrix_New(1, 1, DOUBLE);
        wkappa = Matrix_New(1, 1, DOUBLE);
    }
    if (REFINEMENT) {
        wx2 = Matrix_NewFromMatrix(c, c->id);
        wy2 = Matrix_NewFromMatrix(b, b->id);
        wz2 = Matrix_New(cdim, 1, DOUBLE);
        ws2 = Matrix_New(cdim, 1, DOUBLE);
        wtau2 = Matrix_New(1, 1, DOUBLE);
        wkappa2 = Matrix_New(1, 1, DOUBLE);
    }           
}


int conelp(ECVXConeWorkspace* ecvxcone_ws, ECVXConeSettings* settings) 
{   
    // Use custom options if provided, otherwise use global options
    bool DEBUG = settings->debug;
    double KKTREG = settings->kktreg;
    double FEASTOL = settings->feastol;
    double ABSTOL = settings->abstol;
    double RELTOL = settings->reltol;
    int MAXITERS = settings->maxiters;
    bool SHOW_PROGRESS = settings->show_progress;
    int REFINEMENT = settings->refinement;
    int EXPON = settings->EXPON;
    double STEP = settings->STEP;

    PrimalStart* primalstart = ecvxcone_ws->primalstart;
    DualStart* dualstart = ecvxcone_ws->dualstart;
    ECVXConeResult* result = ecvxcone_ws->result;
    matrix *c = ecvxcone_ws->data->c;
    void *G = ecvxcone_ws->data->G;
    matrix *h = ecvxcone_ws->data->h;
    void *A = ecvxcone_ws->data->A;
    matrix *b = ecvxcone_ws->data->b;
    DIMs* dims = ecvxcone_ws->dims;

    // sum of the dimensions (q and s)
    int sum_dims_q = ecvxcone_ws->sum_dims_q;
    int sum_dims_s = ecvxcone_ws->sum_dims_s;

    // Cone dimensions
    // int cdim = ecvxcone_ws->cdim;
    int cdim_pckd = ecvxcone_ws->cdim_pckd;
    int cdim_diag = ecvxcone_ws->cdim_diag;

    // Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G
    int *indq = ecvxcone_ws->indq;

    // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G
    // int *inds = ecvxcone_ws->inds;

    // kktsolver(W) returns a routine for solving 3x3 block KKT system
    //
    //     [ 0   A'  G'*W^{-1} ] [ ux ]   [ bx ]
    //     [ A   0   0         ] [ uy ] = [ by ].
    //     [ G   0   -W'       ] [ uz ]   [ bz ]

    KKTCholContext *kkt_ctx = NULL;
    scaling *W = NULL;
    if (!KKTREG && (b->nrows > c->nrows || b->nrows + cdim_pckd < c->nrows)) {
        ERR("Error: Rank(A) < p or Rank([G; A]) < n\n");
    }
    if (strcmp(settings->kktsolver, "chol") == 0) {
        kkt_ctx = kkt_chol(G, dims, A, 0);
    } else {
        snprintf(msg, sizeof(msg), "Unsupported KKT solver: %s", settings->kktsolver);
        ERR(msg);
    }
    
    double resx0, resy0, resz0;

    double norm_c = sqrt(blas_dot(c, c, -1, 1, 1, 0, 0).d);          // dot product c⋅c，
    double norm_b = sqrt(blas_dot(b, b, -1, 1, 1, 0, 0).d);          // dot product b⋅b，
    double norm_h = misc_snrm2(h, dims, 0);    // norm of h vector

    resx0 = (norm_c > 1.0) ? norm_c : 1.0;
    resy0 = (norm_b > 1.0) ? norm_b : 1.0;
    resz0 = (norm_h > 1.0) ? norm_h : 1.0;

    // Select initial points.

    double alpha = 0.0;
    double beta = 0.0;

    blas_scal(&alpha, x, -1, 1, 0);
    blas_scal(&alpha, y, -1, 1, 0);

    if (primalstart == NULL || dualstart == NULL) {

        // Factor
        //
        //     [ 0   A'  G' ]
        //     [ A   0   0  ].
        //     [ G   0  -I  ]

        W = ecvxcone_ws->W_init;
        if (W == NULL) {
            ERR("Error: Dummy scaling structure is NULL");
        }
        
        // Try to get solver function
        factor_function(W, NULL, NULL, kkt_ctx, dims);
    }

    if (primalstart == NULL) {

        // minimize    || G * x - h ||^2
        // subject to  A * x = b
        //
        // by solving
        //
        //     [ 0   A'  G' ]   [ x  ]   [ 0 ]
        //     [ A   0   0  ] * [ dy ] = [ b ].
        //     [ G   0  -I  ]   [ -s ]   [ h ]

        double alpha = 0.0;
        blas_scal(&alpha, x, -1, 1, 0);
        xy_copy(b, dy);
        blas_copy(h, s, -1, 1, 1, 0, 0);

        // Try to solve
        solve_function(x, dy, s, kkt_ctx, dims);

        alpha = -1.0;
        blas_scal(&alpha, s, -1, 1, 0);

    } else {
        xy_copy(primalstart->x, x);
        blas_copy(primalstart->s, s, -1, 1, 1, 0, 0);
    }

    // ts = min{ t | s + t*e >= 0 }
    double ts = misc_max_step(s, dims, 0, NULL);
    if (ts >= 0 && primalstart != NULL) {
        ERR("initial s is not positive");
    }

    if (dualstart == NULL) {

        // minimize   || z ||^2
        // subject to G'*z + A'*y + c = 0
        //
        // by solving
        //
        //     [ 0   A'  G' ] [ dx ]   [ -c ]
        //     [ A   0   0  ] [ y  ] = [  0 ].
        //     [ G   0  -I  ] [ z  ]   [  0 ]
        double alpha = -1.0;
        xy_copy(c, dx);
        blas_scal(&alpha, dx, -1, 1, 0);

        alpha = 0.0;
        blas_scal(&alpha, y, -1, 1, 0);
        blas_scal(&alpha, z, -1, 1, 0);
        solve_function(dx, y, z, kkt_ctx, dims);
        // if (result != 0) {
            // ERR("Rank(A) < p or Rank([G; A]) < n");
        // }

    } else {
        if (dualstart->y != NULL) {
            xy_copy(dualstart->y, y);
        }
        blas_copy(dualstart->z, z, -1, 1, 1, 0, 0);
    }

    // tz = min{ t | z + t*e >= 0 }
    double tz = misc_max_step(z, dims, 0, NULL);
    if (tz >= 0 && dualstart != NULL) {
        ERR("initial z is not positive");
    }

    double nrms = misc_snrm2(s, dims, 0);
    double nrmz = misc_snrm2(z, dims, 0);

    double resx;
    double resy;
    double resz;

    double pcost;
    double dcost;
    double relgap = -1.0; // Represents None
    double gap;
    double pres;
    double dres;

    double cx;
    double by;
    double hz;
    double rt;

    int relgap_valid;

    int ind;
    int ind2;

    if (primalstart == NULL && dualstart == NULL) {

        gap = misc_sdot(s, z, dims, 0);
        pcost = blas_dot(c, x, -1, 1, 1, 0, 0).d;
        dcost = -blas_dot(b, y, -1, 1, 1, 0, 0).d - misc_sdot(h, z, dims, 0);
        if (pcost < 0.0) {
            relgap = gap / (-pcost);
        } else if (dcost > 0.0) {
            relgap = gap / dcost;
        } else {
            relgap = -1.0; // Represents None
        }

        if ((ts <= 0 && tz <= 0 && (gap <= ABSTOL || (relgap > 0 && relgap <= RELTOL))) && (KKTREG == -1.0)) {

            // The initial points constructed happen to be feasible and optimal.
            ind = dims->l + sum_dims_q;
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                misc_symm(s, m, ind);
                misc_symm(z, m, ind);
                ind += m * m;
            }

            // rx = A'*y + G'*z + c
            // rx = Matrix_NewFromMatrix(c, c->id);
            beta = 1.0;
            Af_gemv(y, rx, A, 'T', NULL, &beta);
            Gf_gemv(z, rx, G, dims, 'T', NULL, &beta);
            resx = sqrt(blas_dot(rx, rx, -1, 1, 1, 0, 0).d);

            // ry = b - A*x
            alpha = -1.0;
            beta = 1.0;
            // ry = Matrix_NewFromMatrix(b, b->id);
            Af_gemv(x, ry, A, 'N', &alpha, &beta);
            resy = sqrt(blas_dot(ry, ry, -1, 1, 1, 0, 0).d);

            // rz = s + G*x - h
            // rz = Matrix_New(cdim, 1, DOUBLE);
            Gf_gemv(x, rz, G, dims, 'N', NULL, NULL);
            number alpha_n;
            alpha_n.d = -1.0;
            blas_axpy(s, rz, NULL, -1, 1, 1, 0, 0);
            blas_axpy(h, rz, &alpha_n, -1, 1, 1, 0, 0);
            resz = misc_snrm2(rz, dims, 0);

            double norm1 = resy / resy0;
            double norm2 = resz / resz0;
            pres  = (norm1 > norm2) ? norm1 : norm2;
            dres = resx / resx0;
            cx = blas_dot(c, x, -1, 1, 1, 0, 0).d;
            by = blas_dot(b, y, -1, 1, 1, 0, 0).d;
            hz = misc_sdot(h, z, dims, 0);

            if (SHOW_PROGRESS) {
                printf("Optimal solution found.\n");
            }
            
            result->x = x;
            result->y = y;
            result->z = z;
            result->s = s;
            result->status = OPTIMAL;
            result->gap = gap;
            result->relative_gap = relgap;
            result->primal_objective = cx;
            result->dual_objective = -(by + hz);
            result->primal_infeasibility = pres;
            result->primal_slack = -ts;
            result->dual_slack = -tz;
            result->dual_infeasibility = dres;
            result->residual_as_primal_infeasibility_certificate = 0.0;
            result->residual_as_dual_infeasibility_certificate = 0.0;
            result->iterations = 0;

            // Free the KKT context
            KKTCholContext_Free(kkt_ctx);

            return OPTIMAL;
        }

        double threshold_s = (nrms > 1.0) ? nrms : 1.0;
        if (ts >= -1e-8 * threshold_s) {
            double a = 1.0 + ts;
            for (int i = 0; i < dims->l; ++i) {
                MAT_BUFD(s)[i] += a;
            }
            for (int i = 0; i < dims->q_size; ++i) {
                MAT_BUFD(s)[indq[i]] += a;
            }
            ind = dims->l + sum_dims_q;
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                for (int j = 0; j < m; ++j) {
                    MAT_BUFD(s)[ind + j * m + j] += a;
                }
                ind += m * m;
            }
        }

        double threshold_z = (nrmz > 1.0) ? nrmz : 1.0;
        if (tz >= -1e-8 * threshold_z) {
            double a = 1.0 + tz;
            for (int i = 0; i < dims->l; ++i) {
                MAT_BUFD(z)[i] += a;
            }
            for (int i = 0; i < dims->q_size; ++i) {
                MAT_BUFD(z)[indq[i]] += a;
            }
            ind = dims->l + sum_dims_q;
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                for (int j = 0; j < m; ++j) {
                    MAT_BUFD(z)[ind + j * m + j] += a;
                }
                ind += m * m;
            }
        }

    } else if (primalstart == NULL && dualstart != NULL) {

        double threshold_s = (nrms > 1.0) ? nrms : 1.0;
        if (ts >= -1e-8 * threshold_s) {
            double a = 1.0 + ts;
            for (int i = 0; i < dims->l; ++i) {
                MAT_BUFD(s)[i] += a;
            }
            for (int i = 0; i < dims->q_size; ++i) {
                MAT_BUFD(s)[indq[i]] += a;
            }
            ind = dims->l + sum_dims_q;
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                for (int j = 0; j < m; ++j) {
                    MAT_BUFD(s)[ind + j * m + j] += a;
                }
                ind += m * m;
            }
        }

    } else if (primalstart != NULL && dualstart == NULL) {

        double threshold_z = (nrmz > 1.0) ? nrmz : 1.0;
        if (tz >= -1e-8 * threshold_z) {
            double a = 1.0 + tz;
            for (int i = 0; i < dims->l; ++i) {
                MAT_BUFD(z)[i] += a;
            }
            for (int i = 0; i < dims->q_size; ++i) {
                MAT_BUFD(z)[indq[i]] += a;
            }
            ind = dims->l + sum_dims_q;
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                for (int j = 0; j < m; ++j) {
                    MAT_BUFD(z)[ind + j * m + j] += a;
                }
                ind += m * m;
            }
        }
    }

    double tau = 1.0;
    double kappa = 1.0;
    double dg;
    double dgi;
    double tt;
    double tk;
    double wkappa3;
    double mu;
    double sigma;
    double t;
    double dinfres = INFINITY;
    double pinfres = INFINITY;

    int pinfres_valid;
    int dinfres_valid;
    int sol_status;

    number step;

    gap = misc_sdot(s, z, dims, 0);

    for (int iters = 0; iters <= MAXITERS; ++iters) {
        
        alpha = -1.0; 
        beta = 1.0;

        // hrx = -A'*y - G'*z
        Af_gemv(y, hrx, A, 'T', &alpha, NULL);
        Gf_gemv(z, hrx, G, dims, 'T', &alpha, &beta);
        double hresx = sqrt(blas_dot(hrx, hrx, -1, 1, 1, 0, 0).d);
        
        // rx = hrx - c*tau
        // = -A'*y - G'*z - c*tau
        blas_copy(hrx, rx, -1, 1, 1, 0, 0);
        number tau_n;
        tau_n.d = -tau;
        blas_axpy(c, rx, &tau_n, -1, 1, 1, 0, 0);
        resx = sqrt(blas_dot(rx, rx, -1, 1, 1, 0, 0).d) / tau;
        
        // hry = A*x
        Af_gemv(x, hry, A, 'N', NULL, NULL);
        double hresy = sqrt(blas_dot(hry, hry, -1, 1, 1, 0, 0).d);
        
        // ry = hry - b*tau
        // = A*x - b*tau
        blas_copy(hry, ry, -1, 1, 1, 0, 0);
        blas_axpy(b, ry, &tau_n, -1, 1, 1, 0, 0);
        resy = sqrt(blas_dot(ry, ry, -1, 1, 1, 0, 0).d) / tau;
        
        // hrz = s + G*x
        Gf_gemv(x, hrz, G, dims, 'N', NULL, NULL);
        blas_axpy(s, hrz, NULL, -1, 1, 1, 0, 0);
        double hresz = misc_snrm2(hrz, dims, 0);
        
        // rz = hrz - h*tau
        // = s + G*x - h*tau
        alpha = 0.0;
        blas_scal(&alpha, rz, -1, 1, 0);
        blas_axpy(hrz, rz, NULL, -1, 1, 1, 0, 0);
        blas_axpy(h, rz, &tau_n, -1, 1, 1, 0, 0);

        resz = misc_snrm2(rz, dims, 0) / tau;

        // rt = kappa + c'*x + b'*y + h'*z
        cx = blas_dot(c, x, -1, 1, 1, 0, 0).d;
        by = blas_dot(b, y, -1, 1, 1, 0, 0).d;
        hz = misc_sdot(h, z, dims, 0);
        rt = kappa + cx + by + hz;

        // Statistics for stopping criteria.
        pcost = cx / tau;
        dcost = -(by + hz) / tau;
        relgap_valid = 0;
        if (pcost < 0.0) {
            relgap = gap / (-pcost);
            relgap_valid = 1;
        } else if (dcost > 0.0) {
            relgap = gap / dcost;
            relgap_valid = 1;
        } else {
            relgap = -1.0; // Represents None
        }

        double term1 = resy / resy0;
        double term2 = resz / resz0;
        pres = term1 > term2 ? term1 : term2;
        dres = resx / resx0;
        
        pinfres_valid = 0;
        if (hz + by < 0.0) {
            pinfres = hresx / resx0 / (-hz - by);
            pinfres_valid = 1;
        }
        
        dinfres_valid = 0;
        if (cx < 0.0) {
            dinfres = fmax(hresy / resy0, hresz / resz0) / (-cx);
            dinfres_valid = 1;
        }
        
        if (SHOW_PROGRESS) {
            if (iters == 0) {
                printf("% 10s% 12s% 10s% 8s% 7s % 5s\n", "pcost", "dcost", "gap", "pres", "dres", "k/t");
            }
            // printf("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e% 7.0e\n", 
            //        iters, pcost, dcost, gap, pres, dres, kappa/tau);
            printf("%2d: % .10e % .10e % .6e % .6e % .6e % .6e\n", 
                    iters, pcost, dcost, gap, pres, dres, kappa / tau);
        }
        
        if ((pres <= FEASTOL && dres <= FEASTOL && 
             (gap <= ABSTOL || (relgap_valid && relgap <= RELTOL))) || 
            iters == MAXITERS) {
            
            alpha = 1.0 / tau;
            blas_scal(&alpha, x, -1, 1, 0);
            blas_scal(&alpha, y, -1, 1, 0);
            blas_scal(&alpha, s, -1, 1, 0);
            blas_scal(&alpha, z, -1, 1, 0);

            ind = dims->l + sum_dims_q;
            
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                misc_symm(s, m, ind);
                misc_symm(z, m, ind);
                ind += m * m;
            }
            
            ts = misc_max_step(s, dims, 0, NULL);
            tz = misc_max_step(z, dims, 0, NULL);
            
            if (iters == MAXITERS) {
                if (SHOW_PROGRESS) {
                    printf("Terminated (maximum number of iterations reached).\n");
                }
                result->status = UNKNOWN;
                sol_status = UNKNOWN;
            } else {
                if (SHOW_PROGRESS) {
                    printf("Optimal solution found.\n");
                }
                result->status = OPTIMAL;
                sol_status = OPTIMAL;
            }

            result->x = x;
            result->y = y;
            result->z = z;
            result->s = s;
            result->status = sol_status;
            result->gap = gap;
            result->relative_gap = relgap_valid ? relgap : 0.0;
            result->primal_objective = pcost;
            result->dual_objective = dcost;
            result->primal_infeasibility = pres;
            result->dual_infeasibility = dres;
            result->primal_slack = -ts;
            result->dual_slack = -tz;
            result->residual_as_primal_infeasibility_certificate = pinfres_valid ? pinfres : 0.0;
            result->residual_as_dual_infeasibility_certificate = (iters == MAXITERS && dinfres_valid) ? dinfres : 0.0;
            result->iterations = iters;

            if (iters > 0) {
                Scaling_Free(W);
            }
            KKTCholContext_Free(kkt_ctx);
            return sol_status;
        }
        
        else if (pinfres_valid && pinfres <= FEASTOL) {
            alpha = 1.0 / (-hz - by);
            blas_scal(&alpha, y, -1, 1, 0);
            blas_scal(&alpha, z, -1, 1, 0);

            ind = dims->l + sum_dims_q;
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                misc_symm(z, m, ind);
                ind += m * m;
            }
            
            tz = misc_max_step(z, dims, 0, NULL);
            if (SHOW_PROGRESS) {
                printf("Certificate of primal infeasibility found.\n");
            }

            result->x = x;
            result->s = s;
            result->y = y;
            result->z = z;
            result->status = PRIMAL_INFEASIBLE;
            result->gap = 0.0;
            result->relative_gap = 0.0;
            result->primal_objective = 0.0;
            result->dual_objective = 1.0;
            result->primal_infeasibility = 0.0;
            result->dual_infeasibility = 0.0;
            result->primal_slack = 0.0;
            result->dual_slack = -tz;
            result->residual_as_primal_infeasibility_certificate = pinfres;
            result->residual_as_dual_infeasibility_certificate = 0.0;
            result->iterations = iters;

            if (iters > 0) {
                Scaling_Free(W);
            }

            KKTCholContext_Free(kkt_ctx);
            return PRIMAL_INFEASIBLE;
        }
        
        else if (dinfres_valid && dinfres <= FEASTOL) {
            alpha = 1.0 / (-cx);
            blas_scal(&alpha, x, -1, 1, 0);
            blas_scal(&alpha, s, -1, 1, 0);
            
            ind = dims->l + sum_dims_q;
            for (int i = 0; i < dims->s_size; ++i) {
                int m = dims->s[i];
                misc_symm(s, m, ind);
                ind += m * m;
            }
            
            ts = misc_max_step(s, dims, 0, NULL);
            
            if (SHOW_PROGRESS) {
                printf("Certificate of dual infeasibility found.\n");
            }
            
            result->x = x;
            result->y = y;
            result->s = s;
            result->z = z;
            result->status = DUAL_INFEASIBLE;
            result->gap = 0.0;
            result->relative_gap = 0.0;
            result->primal_objective = -1.0;
            result->dual_objective = 0.0;
            result->primal_infeasibility = 0.0;
            result->dual_infeasibility = 0.0;
            result->primal_slack = -ts;
            result->dual_slack = 0.0;
            result->residual_as_primal_infeasibility_certificate = 0.0;
            result->residual_as_dual_infeasibility_certificate = dinfres;
            result->iterations = iters;

            if (iters > 0) {
                Scaling_Free(W);
            }

            KKTCholContext_Free(kkt_ctx);
            return DUAL_INFEASIBLE;
        }
        
        // Compute initial scaling W:
        //
        // W * z = W^{-T} * s = lambda
        // dg * tau = 1/dg * kappa = lambdag.
        
        if (iters == 0) {
            W = misc_compute_scaling(s, z, lmbda, dims, 0);
            // W = ecvxcone_ctx->W_nt;
            // misc_compute_scaling2(W, s, z, lmbda, dims, 0);

            // dg = sqrt( kappa / tau )
            // dgi = sqrt( tau / kappa )
            // lambda_g = sqrt( tau * kappa )
            //
            // lambda_g is stored in the last position of lmbda.
            
            dg = sqrt(kappa / tau);
            dgi = sqrt(tau / kappa);
            MAT_BUFD(lmbda)[cdim_diag] = sqrt(tau * kappa);
        }
        
        // lmbdasq := lmbda o lmbda
        misc_ssqr(lmbdasq, lmbda, dims, 0);
        MAT_BUFD(lmbdasq)[cdim_diag] = MAT_BUFD(lmbda)[cdim_diag] * MAT_BUFD(lmbda)[cdim_diag];

        // f3(x, y, z) solves
        //
        // [ 0  A'  G'   ] [ ux        ]   [ bx ]
        // [ A  0   0    ] [ uy        ] = [ by ].
        // [ G  0  -W'*W ] [ W^{-1}*uz ]   [ bz ]
        //
        // On entry, x, y, z contain bx, by, bz.
        // On exit, they contain ux, uy, uz.
        //
        // Also solve
        //
        // [ 0   A'  G'    ] [ x1        ]          [ c ]
        // [-A   0   0     ]*[ y1        ] = -dgi * [ b ].
        // [-G   0   W'*W  ] [ W^{-1}*z1 ]          [ h ]

        factor_function(W, NULL, NULL, kkt_ctx, dims);

        alpha = -1.0;
        xy_copy(c, x1);
        blas_scal(&alpha, x1, -1, 1, 0);
        xy_copy(b, y1_);
        blas_copy(h, z1, -1, 1, 1, 0, 0);

        solve_function(x1, y1_, z1, kkt_ctx, dims);

        blas_scal(&dgi, x1, -1, 1, 0);
        blas_scal(&dgi, y1_, -1, 1, 0);
        blas_scal(&dgi, z1, -1, 1, 0);
        
        // if (iters == 0 && primalstart && dualstart) {
        //     ERR("Error: Rank(A) < p or Rank([G; A]) < n");
        // } else {
        //     alpha = 1.0 / tau;
        //     blas_scal(&alpha, x, -1, 1, 0);
        //     blas_scal(&alpha, y, -1, 1, 0);
        //     blas_scal(&alpha, s, -1, 1, 0);
        //     blas_scal(&alpha, z, -1, 1, 0);

        //     int ind = dims->l + sum_array(dims->q, dims->q_size);               
        //     for (int i = 0; i < dims->s_size; ++i) {
        //         int m = dims->s[i];
        //         misc_symm(s, m, ind);
        //         misc_symm(z, m, ind);
        //         ind += m * m;
        //     }
            
        //     double ts = misc_max_step(s, dims, 0, NULL);
        //     double tz = misc_max_step(z, dims, 0, NULL);

        //     if (SHOW_PROGRESS) {
        //         printf("Terminated (singular KKT matrix).\n");
        //     }
            
        //     result->x = x;
        //     result->y = y;
        //     result->s = s;
        //     result->z = z;
        //     result->status = UNKNOWN;
        //     result->gap = gap;
        //     result->relative_gap = relgap_valid ? relgap : 0.0;
        //     result->primal_objective = pcost;
        //     result->dual_objective = dcost;
        //     result->primal_infeasibility = pres;
        //     result->dual_infeasibility = dres;
        //     result->primal_slack = -ts;
        //     result->dual_slack = -tz;
        //     result->residual_as_primal_infeasibility_certificate = 
        //         pinfres_valid ? pinfres : 0.0;
        //     result->residual_as_dual_infeasibility_certificate = 
        //         dinfres_valid ? dinfres : 0.0;
        //     result->iterations = iters;

        //     return result;
        // }
        
        
        // th = W^{-T} * h
        if (iters == 0) {
            // th = Matrix_New(cdim, 1, DOUBLE);
        }
        blas_copy(h, th, -1, 1, 1, 0, 0);
        misc_scale(th, W, 'T', 'I');

        mu = pow(blas_nrm2(lmbda, -1, 1, 0), 2) / (1 + cdim_diag);
        sigma = 0.0;

        for (int i = 0; i < 2; ++i) {
            // Solve
            //
            // [ 0         ]   [  0   A'  G'  c ] [ dx        ]
            // [ 0         ]   [ -A   0   0   b ] [ dy        ]
            // [ W'*ds     ] - [ -G   0   0   h ] [ W^{-1}*dz ]
            // [ dg*dkappa ]   [ -c' -b' -h'  0 ] [ dtau/dg   ]
            //
            //                   [ rx   ]
            //                   [ ry   ]
            //     = - (1-sigma) [ rz   ]
            //                   [ rtau ]
            //
            // lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e
            // lmbdag * (dtau + dkappa) = - kappa * tau + sigma*mu
            
            // ds = -lmbdasq if i is 0
            //    = -lmbdasq - dsa o dza + sigma*mu*e if i is 1
            // dkappa = -lambdasq[-1] if i is 0
            //        = -lambdasq[-1] - dkappaa*dtaua + sigma*mu if i is 1.

            blas_copy(lmbdasq, ds, dims->l + sum_dims_q, 1, 1, 0, 0);
            ind = dims->l + sum_dims_q;
            ind2 = ind;
            alpha = 0.0;
            blas_scal(&alpha, ds, -1, 1, ind);
            
            for (int j = 0; j < dims->s_size; ++j) {
                int m = dims->s[j];
                blas_copy(lmbdasq, ds, m, 1, m+1, ind2, ind);
                ind += m * m;
                ind2 += m;
            }
            MAT_BUFD(dkappa)[0] = MAT_BUFD(lmbdasq)[cdim_diag];
            
            if (i == 1) {
                blas_axpy(ws3, ds, NULL, -1, 1, 1, 0, 0);
                for (int j = 0; j < dims->l; ++j) {
                    MAT_BUFD(ds)[j] -= sigma * mu;
                }
                
                // Process indq array
                for (int j = 0; j < dims->q_size; ++j) {
                    MAT_BUFD(ds)[(int)indq[j]] -= sigma * mu;
                }

                ind = dims->l + sum_dims_q;
                ind2 = ind;
                for (int j = 0; j < dims->s_size; ++j) {
                    int m = dims->s[j];
                    for (int k = 0; k < m; k++) {
                        MAT_BUFD(ds)[ind + k * m + k] -= sigma * mu;
                    }
                    ind += m * m;
                }

                MAT_BUFD(dkappa)[0] += wkappa3 - sigma * mu;
            }
            
            // (dx, dy, dz, dtau) = (1-sigma)*(rx, ry, rz, rt)
            xy_copy(rx, dx);
            alpha = 1.0 - sigma;
            blas_scal(&alpha, dx, -1, 1, 0);
            xy_copy(ry, dy);
            blas_scal(&alpha, dy, -1, 1, 0);
            blas_copy(rz, dz, -1, 1, 1, 0, 0);
            blas_scal(&alpha, dz, -1, 1, 0);
            MAT_BUFD(dtau)[0] = (1.0 - sigma) * rt;
            
            f6(dx, dy, dz, dtau, ds, dkappa, c, b, h, G, A, dims, 
                W, kkt_ctx, cdim_diag, dg, dgi, iters, REFINEMENT, DEBUG);
                
            // Save ds o dz and dkappa * dtau for Mehrotra correction
            if (i == 0) {
                blas_copy(ds, ws3, -1, 1, 1, 0, 0);
                misc_sprod(ws3, dz, dims, 0, 'N');
                wkappa3 = MAT_BUFD(dtau)[0] * MAT_BUFD(dkappa)[0];
            }
            
            // Maximum step to boundary.
            //
            // If i is 1, also compute eigenvalue decomposition of the 's'
            // blocks in ds, dz.  The eigenvectors Qs, Qz are stored in
            // dsk, dzk.  The eigenvalues are stored in sigs, sigz.

            misc_scale2(lmbda, ds, dims, 0, 'N');
            misc_scale2(lmbda, dz, dims, 0, 'N');

            if (i == 0) {
                ts = misc_max_step(ds, dims, 0, NULL);
                tz = misc_max_step(dz, dims, 0, NULL);
            } else {
                ts = misc_max_step(ds, dims, 0, sigs);
                tz = misc_max_step(dz, dims, 0, sigz);
            }

            tt = -MAT_BUFD(dtau)[0] / MAT_BUFD(lmbda)[cdim_diag];
            tk = -MAT_BUFD(dkappa)[0] / MAT_BUFD(lmbda)[cdim_diag];
            t = fmax(fmax(fmax(fmax(0.0, ts), tz), tt), tk);
            
            if (t == 0.0) {
                step.d = 1.0;
            } else {
                if (i == 0) {
                    step.d = fmin(1.0, 1.0 / t);
                } else {
                    step.d = fmin(1.0, STEP / t);
                }
            }
            
            if (i == 0) {
                sigma = pow(1.0 - step.d, EXPON);
            }
        }
     
        // Update x, y.
        blas_axpy(dx, x, &step, -1, 1, 1, 0, 0);
        blas_axpy(dy, y, &step, -1, 1, 1, 0, 0);
        
        // Replace 'l' and 'q' blocks of ds and dz with the updated
        // variables in the current scaling.
        // Replace 's' blocks of ds and dz with the factors Ls, Lz in a
        // factorization Ls*Ls', Lz*Lz' of the updated variables in the
        // current scaling.
        
        // ds := e + step*ds for 'l' and 'q' blocks.
        // dz := e + step*dz for 'l' and 'q' blocks.

        blas_scal(&step.d, ds, dims->l + sum_dims_q, 1, 0);
        blas_scal(&step.d, dz, dims->l + sum_dims_q, 1, 0);
        
        for (int i = 0; i < dims->l; ++i) {
            MAT_BUFD(ds)[i] += 1.0;
            MAT_BUFD(dz)[i] += 1.0;
        }
        for (int i = 0; i < dims->q_size; ++i) {
            MAT_BUFD(ds)[indq[i]] += 1.0;
            MAT_BUFD(dz)[indq[i]] += 1.0;
        }
        
        // ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        //
        // This replaces the 'l' and 'q' components of ds and dz with the
        // updated variables in the current scaling.
        // The 's' components of ds and dz are replaced with
        //
        //     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
        //     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
        //

        misc_scale2(lmbda, ds, dims, 0, 'I');
        misc_scale2(lmbda, dz, dims, 0, 'I');

        // sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        // sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas_scal(&step.d, sigs, -1, 1, 0);
        blas_scal(&step.d, sigz, -1, 1, 0);

        for (int j = 0; j < sum_dims_s; ++j) {
            MAT_BUFD(sigs)[j] += 1.0;
            MAT_BUFD(sigz)[j] += 1.0;
        }

        blas_tbsv(lmbda, sigs, 'L', 'N', 'N', sum_dims_s, 0, 1, 1, dims->l + sum_dims_q, 0);
        blas_tbsv(lmbda, sigz, 'L', 'N', 'N', sum_dims_s, 0, 1, 1, dims->l + sum_dims_q, 0);

        // dsk := Ls = dsk * sqrt(sigs).
        // dzk := Lz = dzk * sqrt(sigz).
        ind2 = dims->l + sum_dims_q;
        int ind3 = 0;

        for (int k = 0; k < dims->s_size; k++) {
            int m = dims->s[k];
            for (int i = 0; i < m; ++i) {
                double tmp_s = sqrt(MAT_BUFD(sigs)[ind3 + i]);
                double tmp_z = sqrt(MAT_BUFD(sigz)[ind3 + i]);
                blas_scal(&tmp_s, ds, m, 1, ind2 + m * i);
                blas_scal(&tmp_z, dz, m, 1, ind2 + m * i);
            }
            ind2 += m * m;
            ind3 += m;
        }
    
        // Update lambda and scaling.
        misc_update_scaling(W, lmbda, ds, dz);

        // For kappa, tau block:
        //
        //     dg := sqrt( (kappa + step*dkappa) / (tau + step*dtau) )
        //         = dg * sqrt( (1 - step*tk) / (1 - step*tt) )
        //
        //     lmbda[-1] := sqrt((tau + step*dtau) * (kappa + step*dkappa))
        //                = lmbda[-1] * sqrt(( 1 - step*tt) * (1 - step*tk))

        dg *= sqrt(1.0 - step.d*tk) / sqrt(1.0 - step.d*tt);
        dgi = 1.0 / dg;
        MAT_BUFD(lmbda)[cdim_diag] *= sqrt(1.0 - step.d*tt) * sqrt(1.0 - step.d*tk);

        // Unscale s, z, tau, kappa (unscaled variables are used only to
        // compute feasibility residuals).

        blas_copy(lmbda, s, dims->l + sum_dims_q, 1, 1, 0, 0);
        ind = dims->l + sum_dims_q;
        ind2 = ind;
        for (int i = 0; i < dims->s_size; ++i) {
            int m = dims->s[i];
            double zero = 0.0;
            blas_scal(&zero, s, m*m, 1, ind2);
            blas_copy(lmbda, s, m, 1, m+1, ind, ind2);
            ind += m;
            ind2 += m*m;
        }

        misc_scale(s, W, 'T', 'N');

        blas_copy(lmbda, z, dims->l + sum_dims_q, 1, 1, 0, 0);
        ind = dims->l + sum_dims_q;
        ind2 = ind;
        for (int i = 0; i < dims->s_size; ++i) {
            int m = dims->s[i];
            alpha = 0.0;
            blas_scal(&alpha, z, -1, 1, ind2);
            blas_copy(lmbda, z, m, 1, m+1, ind, ind2);
            ind += m;
            ind2 += m*m;
        }
        misc_scale(z, W, 'N', 'I');

        kappa = MAT_BUFD(lmbda)[cdim_diag] / dgi;
        tau = MAT_BUFD(lmbda)[cdim_diag] * dgi;

        gap = pow(blas_nrm2(lmbda, lmbda->nrows - 1, 1, 0) / tau, 2.0);
    }
}

void Gf_gemv(matrix *x_, matrix *y_, void *G, DIMs *dims, char trans, void* alpha, void* beta) 
{
    if (!x_ || !y_ || !G)  ERR("Error: Gf_gemv requires non-null x, y, and G matrices\n");

    double one_ = 1.0;
    double zero_ = 0.0;
    if (alpha == NULL) {
        alpha = &one_; // Default alpha to 1.0
    }
    if (beta == NULL) {
        beta = &zero_; // Default beta to 0.0
    }
    if (trans == 0) trans = 'N'; // Default transposition to 'N'

    misc_sgemv(G, x_, y_, dims, trans, *(double*)alpha, *(double*)beta, -1, 0, 0, 0);
}

void Af_gemv(matrix *x_, matrix *y_, void *A, char trans, void* alpha, void* beta) 
{
    if (!x_ || !y_ || !A)  ERR("Error: Af_gemv requires non-null x, y, and A matrices\n");

    if (alpha == NULL) {
        double one = 1.0;
        alpha = &one; // Default alpha to 1.0
    }
    if (beta == NULL) {
        double zero = 0.0;
        beta = &zero; // Default beta to 0.0
    }
    if (trans == 0) trans = 'N'; // Default transposition to 'N'

    base_gemv(A, x_, y_, trans, alpha, beta, -1, -1, 1, 1, 0, 0, 0);
}


// res() evaluates residual in 5x5 block KKT system
//
//     [ vx   ]    [ 0         ]   [ 0   A'  G'  c ] [ ux        ]
//     [ vy   ]    [ 0         ]   [-A   0   0   b ] [ uy        ]
//     [ vz   ] += [ W'*us     ] - [-G   0   0   h ] [ W^{-1}*uz ]
//     [ vtau ]    [ dg*ukappa ]   [-c' -b' -h'  0 ] [ utau/dg   ]
//
//           vs += lmbda o (dz + ds)
//       vkappa += lmbdg * (dtau + dkappa).
void res(matrix *ux, matrix *uy, matrix *uz, matrix *utau, matrix *us, matrix *ukappa, 
         matrix *vx, matrix *vy, matrix *vz, matrix *vtau, matrix *vs, matrix *vkappa, 
         matrix *c, matrix *b, matrix *h, void* G, void* A, DIMs *dims, scaling *W, 
         double dg, int cdim_diag)
{

    double alpha = -1.0;
    double beta = 1.0;

    // vx := vx - A'*uy - G'*W^{-1}*uz - c*utau/dg
    Af_gemv(uy, vx, A, 'T', &alpha, &beta);
    blas_copy(uz, wz3, -1, 1, 1, 0, 0);
    misc_scale(wz3, W, 'N', 'I');
    Gf_gemv(wz3, vx, G, dims, 'T', &alpha, &beta);

    number alpha_n;
    alpha_n.d = -MAT_BUFD(utau)[0] / dg;
    blas_axpy(c, vx, &alpha_n, -1, 1, 1, 0, 0);

    alpha = 1.0;
    // vy := vy + A*ux - b*utau/dg
    Af_gemv(ux, vy, A, 'N', &alpha, &beta);
    blas_axpy(b, vy, &alpha_n, -1, 1, 1, 0, 0);

    // vz := vz + G*ux - h*utau/dg + W'*us
    Gf_gemv(ux, vz, G, dims, 'N', &alpha, &beta);
    blas_axpy(h, vz, &alpha_n, -1, 1, 1, 0, 0);
    blas_copy(us, ws3, -1, 1, 1, 0, 0);
    misc_scale(ws3, W, 'T', 'N');
    blas_axpy(ws3, vz, NULL, -1, 1, 1, 0, 0);

    // vtau := vtau + c'*ux + b'*uy + h'*W^{-1}*uz + dg*ukappa
    MAT_BUFD(vtau)[0] += dg*MAT_BUFD(ukappa)[0] + 
        blas_dot(c, ux, -1, 1, 1, 0, 0).d + 
        blas_dot(b, uy, -1, 1, 1, 0, 0).d + 
        misc_sdot(h, wz3, dims, 0);

    // vs := vs + lmbda o (uz + us)
    blas_copy(us, ws3, -1, 1, 1, 0, 0);
    blas_axpy(uz, ws3, NULL, -1, 1, 1, 0, 0);
    misc_sprod(ws3, lmbda, dims, 0, 'D');
    blas_axpy(ws3, vs, NULL, -1, 1, 1, 0, 0);

    // vkappa += vkappa + lmbdag * (utau + ukappa)
    MAT_BUFD(vkappa)[0] += MAT_BUFD(lmbda)[cdim_diag] * (MAT_BUFD(utau)[0] + MAT_BUFD(ukappa)[0]);
               
}

void xy_copy(matrix *x, matrix *y) 
{
    double alpha = 0.0;
    blas_scal(&alpha, y, -1, 1, 0);
    blas_axpy(x, y, NULL, -1, 1, 1, 0, 0);
}


// f6_no_ir function definition
void f6_no_ir(matrix* x_, matrix* y_, matrix* z_, matrix* tau, matrix* s_, matrix* kappa, matrix *c, 
              matrix *b, DIMs *dims, scaling *W, KKTCholContext *kkt_ctx, int cdim_diag, double dgi)
{
    // Solve
    //
    // [  0   A'  G'    0   ] [ ux        ]
    // [ -A   0   0     b   ] [ uy        ]
    // [ -G   0   W'*W  h   ] [ W^{-1}*uz ]
    // [ -c' -b' -h'    k/t ] [ utau/dg   ]
    //
    //       [ bx                    ]
    //       [ by                    ]
    //     = [ bz - W'*(lmbda o\ bs) ]
    //       [ btau - bkappa/tau     ]
    //
    // us = -lmbda o\ bs - uz
    // ukappa = -bkappa/lmbdag - utau.
    
    // First solve
    //
    // [ 0  A' G'   ] [ ux        ]   [  bx                    ]
    // [ A  0  0    ] [ uy        ] = [ -by                    ]
    // [ G  0 -W'*W ] [ W^{-1}*uz ]   [ -bz + W'*(lmbda o\ bs) ]
    
    
    // y := -y = -by
    double alpha = -1.0;
    blas_scal(&alpha, y_, -1, 1, 0);
    
    // s := -lmbda o\ s = -lmbda o\ bs
    misc_sinv(s_, lmbda, dims, 0);
    blas_scal(&alpha, s_, -1, 1, 0);

    // z := -(z + W'*s) = -bz + W'*(lambda o\ bs)
    blas_copy(s_, ws3, -1, 1, 1, 0, 0);
    misc_scale(ws3, W, 'T', 'N');
    blas_axpy(ws3, z_, NULL, -1, 1, 1, 0, 0);
    blas_scal(&alpha, z_, -1, 1, 0);

    solve_function(x_, y_, z_, kkt_ctx, dims);

    // Combine with solution of
    //
    // [ 0   A'  G'    ] [ x1         ]          [ c ]
    // [-A   0   0     ] [ y1         ] = -dgi * [ b ]
    // [-G   0   W'*W  ] [ W^{-1}*dzl ]          [ h ]
    //
    // to satisfy
    //
    // -c'*x - b'*y - h'*W^{-1}*z + dg*tau = btau - bkappa/tau.
    
    // kappa[0] := -kappa[0] / lmbd[-1] = -bkappa / lmbdag
    MAT_BUFD(kappa)[0] = -MAT_BUFD(kappa)[0] / MAT_BUFD(lmbda)[cdim_diag];

    // tau[0] = tau[0] + kappa[0] / dgi = btau[0] - bkappa / tau
    MAT_BUFD(tau)[0] += MAT_BUFD(kappa)[0] / dgi;
    MAT_BUFD(tau)[0] = dgi * (MAT_BUFD(tau)[0] + 
            blas_dot(c, x_, -1, 1, 1, 0, 0).d + blas_dot(b, y_, -1, 1, 1, 0, 0).d + 
            misc_sdot(th, z_, dims, 0)) / (1.0 + misc_sdot(z1, z1, dims, 0));

    number alpha_n;
    alpha_n.d = MAT_BUFD(tau)[0];

    blas_axpy(x1, x_, &alpha_n, -1, 1, 1, 0, 0);
    blas_axpy(y1_, y_, &alpha_n, -1, 1, 1, 0, 0);
    blas_axpy(z1, z_, &alpha_n, -1, 1, 1, 0, 0);

    // s := s - z = - lambda o\ bs - z
    alpha_n.d = -1.0;
    blas_axpy(z_, s_, &alpha_n, -1, 1, 1, 0, 0);

    MAT_BUFD(kappa)[0] -= MAT_BUFD(tau)[0];
}


// f6 function definition
void f6(matrix* x_, matrix* y_, matrix* z_, matrix* tau, matrix* s_, matrix* kappa, 
        matrix *c, matrix *b, matrix *h, void *G, void *A, DIMs *dims, scaling *W, 
        KKTCholContext *kkt_ctx, int cdim_diag, double dg, double dgi, int iters, 
        int REFINEMENT, int DEBUG) 
    {
    if (REFINEMENT || DEBUG) {
        xy_copy(x_, wx);
        xy_copy(y_, wy);
        blas_copy(z_, wz, -1, 1, 1, 0, 0);
        MAT_BUFD(wtau)[0] = MAT_BUFD(tau)[0];
        blas_copy(s_, ws, -1, 1, 1, 0, 0);
        MAT_BUFD(wkappa)[0] = MAT_BUFD(kappa)[0];
    }

    f6_no_ir(x_, y_, z_, tau, s_, kappa, c, b, dims, W, kkt_ctx, cdim_diag, dgi);

    for (int i = 0; i < REFINEMENT; ++i) {
        xy_copy(wx, wx2);
        xy_copy(wy, wy2);
        blas_copy(wz, wz2, -1, 1, 1, 0, 0);
        MAT_BUFD(wtau2)[0] = MAT_BUFD(wtau)[0];
        blas_copy(ws, ws2, -1, 1, 1, 0, 0);
        MAT_BUFD(wkappa2)[0] = MAT_BUFD(wkappa)[0];

        res(x_, y_, z_, tau, s_, kappa, wx2, wy2, wz2, wtau2, ws2, wkappa2, 
            c, b, h, G, A, dims, W, dg, cdim_diag);

        f6_no_ir(wx2, wy2, wz2, wtau2, ws2, wkappa2, c, b, dims, W, kkt_ctx, cdim_diag, dgi);

        blas_axpy(wx2, x_, NULL, -1, 1, 1, 0, 0);
        blas_axpy(wy2, y_, NULL, -1, 1, 1, 0, 0);
        blas_axpy(wz2, z_, NULL, -1, 1, 1, 0, 0);
        MAT_BUFD(tau)[0] += MAT_BUFD(wtau2)[0];
        blas_axpy(ws2, s_, NULL, -1, 1, 1, 0, 0);
        MAT_BUFD(kappa)[0] += MAT_BUFD(wkappa2)[0];
    }
    
    if (DEBUG) {
        res(x_, y_, z_, tau, s_, kappa, wx, wy, wz, wtau, ws, wkappa, 
            c, b, h, G, A, dims, W, dg, cdim_diag);
    
        printf("KKT residuals\n");
        printf("    'x': %e\n", sqrt(blas_dot(wx, wx, -1, 1, 1, 0, 0).d));
        printf("    'y': %e\n", sqrt(blas_dot(wy, wy, -1, 1, 1, 0, 0).d));
        printf("    'z': %e\n", misc_snrm2(wz, dims, 0));
        printf("    'tau': %e\n", fabs(MAT_BUFD(wtau)[0]));
        printf("    's': %e\n", misc_snrm2(ws, dims, 0));
        printf("    'kappa': %e\n", fabs(MAT_BUFD(wkappa)[0]));
    }
}