#ifndef _REGRESSOR_H_
#define _REGRESSOR_H_

#include <Eigen/Eigen/Dense>

class Regressor
{
  private:

    // Declaration
    const int dim;
    const int degree;
    const int tf_dim;
    const double lambda;
    Eigen::MatrixXd W;

    // Private member functions
    int transform_dim (const int dim, const int degree);
    Eigen::MatrixXi build_expn_table_partial (const int input_dim, const int expn_sum);
    Eigen::MatrixXi build_expn_table_full ();
    Eigen::VectorXd transform_data_onecol (Eigen::MatrixXd & X, Eigen::VectorXi expn_onecol);
    Eigen::MatrixXd transform_data (Eigen::MatrixXd & X);

  public:

    // Constructor
    explicit Regressor (const int dim, const int degree=1, const double lambda=0.0);

    // Public member functions
    Eigen::MatrixXd get_param ();
    void train (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
    Eigen::MatrixXd predict (Eigen::MatrixXd & X);
    double evaluate_error (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
};

#endif
