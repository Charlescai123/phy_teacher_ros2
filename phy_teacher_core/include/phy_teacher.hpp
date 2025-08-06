#ifndef PHY_TEACHER_HPP
#define PHY_TEACHER_HPP

#include <cmath>
#include <ctime>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
  #include "ecvxcone/base.h"
  #include "ecvxcone/blas.h"
  #include "ecvxcone/misc.h"
  #include "ecvxcone/solver.h"
  #include "ecvxcone/lapack.h"
  #include "ecvxcone/cvxopt.h"
  #include "ecvxcone/cpg_solve.h"
  #include "ecvxcone/cpg_workspace.h"
}
#endif

ECVXConeWorkspace* ecvxcone_setup(int n_var, int n_ineq, int n_eq, int nnz_G, int nnz_A, DIMs *dims, ECVXConeSettings *settings);
// extern ECVXConeWorkspace* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, ECVXConeSettings *settings);


// Sampling period
constexpr double Ts = 1.0 / 20.0;

namespace phy_teacher {

// Matrix typedefs
using Matrix3x3 = double[3][3];
using Matrix6x6 = double[6][6];

class PHYTeacher {
public:
  PHYTeacher();  // Constructor
  ~PHYTeacher(); // Destructor

  Matrix6x6 F_kp;
  Matrix6x6 F_kd;

  double tracking_err[10];
  double tracking_err_square[10];

  void update();             // Update matrices and call solver
  void test_and_print();     // Test inputs and print results
  void benchmark(int iterations = 1000, bool verbose = false, double threshold = 0.1);
  
  // Rotation matrices
  void update_Rx(double roll);
  void update_Ry(double pitch);
  void update_Rz(double yaw);
  void update_Rzyx(double roll, double pitch, double yaw);

  // Utility
  static void print_matrix6x6(const Matrix6x6& mat, const char* name);
  static void matrix_multiply(Matrix3x3 A, Matrix3x3 B, Matrix3x3 result) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          result[i][j] = 0;
          for (int k = 0; k < 3; ++k)
              result[i][j] += A[i][k] * B[k][j];
        }
  }

private:
  // matrix objects
  matrix* P_mat_;
  matrix* Q_mat_;
  matrix* R_mat_;
  matrix* aB_;
  matrix* aF_;
  matrix* ipiv_;

  Matrix3x3 Rx_;
  Matrix3x3 Ry_;
  Matrix3x3 Rz_;
  Matrix3x3 Rzyx_;

  // Initialization helpers
  void lmi_init();                 // Initialize LMI matrices
  void test_tracking_err();       // Set up tracking error for testing
  double time_diff(timespec start, timespec end, int count);

  // Matrix operations
  void update_Matrix_A();
  void update_Matrix_B();
  void update_TrackingErrorSquare();
  void post_processing();

};

}  // namespace phy_teacher

#endif  // PHY_TEACHER_HPP
