#ifndef _REGRESSOR_H_
#define _REGRESSOR_H_

#include <Eigen/Eigen/Dense>

class Regressor
{
  private:

    // Declaration
    const int dim;
    int N;
    const double kernel_sigma;
    const double likelihood_sigma;
    Eigen::MatrixXd Data;
    Eigen::MatrixXd Vec;

    // Private member functions
    Eigen::MatrixXd kernel_matrix (Eigen::MatrixXd & X1, Eigen::MatrixXd & X2);

  public:

    // Constructor
    explicit Regressor (const int dim, const int N=1,
                        const double kernel_sigma=1.0, const double likelihood_sigma=1.0);

    // Public member functions
    void train (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
    Eigen::MatrixXd predict (Eigen::MatrixXd & X);
    double evaluate_error (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
};

#endif
