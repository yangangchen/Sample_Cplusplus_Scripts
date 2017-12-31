#ifndef _REGRESSOR_H_
#define _REGRESSOR_H_

#include <Eigen/Eigen/Dense>

class Regressor
{
  private:

    // Declaration
    const int dim;
    const double lambda;
    Eigen::MatrixXd W;

    // Private member functions
    void enlarge_data(Eigen::MatrixXd & X);

  public:

    // Constructor
    explicit Regressor (const int dim, const double lambda=0.0);

    // Public member functions
    Eigen::MatrixXd get_param ();
    void train (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
    Eigen::MatrixXd predict (Eigen::MatrixXd & X);
    double evaluate_error (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
};

#endif
