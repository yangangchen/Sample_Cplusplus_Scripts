/*
    class Regressor
    class PolynomialRegressor: public Regressor
    class GaussianProcessRegressor: public Regressor
*/

#ifndef _REGRESSOR_H_
#define _REGRESSOR_H_

#include <Eigen/Eigen/Dense>

class Regressor
{
  protected:

    // Declaration
    const int dim;

  public:

    // Constructor
    explicit Regressor (const int dim);
    // Public member functions
    virtual void train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) = 0;
    virtual Eigen::MatrixXd predict (const Eigen::MatrixXd & X) const = 0;
    virtual double evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const = 0;
};

class PolynomialRegressor: public Regressor
{
  private:

    // Declaration
    const int degree;
    const int tf_dim;
    const double lambda;
    Eigen::MatrixXd W;

    // Private member functions
    int transform_dim (const int dim, const int degree) const;
    Eigen::MatrixXi build_expn_table_partial (const int input_dim, const int expn_sum) const;
    Eigen::MatrixXi build_expn_table_full () const;
    Eigen::VectorXd transform_data_onecol (const Eigen::MatrixXd & X, const Eigen::VectorXi expn_onecol) const;
    Eigen::MatrixXd transform_data (const Eigen::MatrixXd & X) const;

  public:

    // Constructor
    explicit PolynomialRegressor (const int dim, const int degree=1, const double lambda=0.0);
    // Public member functions
    Eigen::MatrixXd get_param () const;
    void train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) override;
    Eigen::MatrixXd predict (const Eigen::MatrixXd & X) const override;
    double evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const override;
};

class GaussianProcessRegressor: public Regressor
{
  private:

    // Declaration
    int N;
    const double kernel_sigma;
    const double likelihood_sigma;
    Eigen::MatrixXd Data;
    Eigen::MatrixXd Vec;

    // Private member functions
    Eigen::MatrixXd kernel_matrix (const Eigen::MatrixXd & X1, const Eigen::MatrixXd & X2) const;

  public:

    // Constructor
    explicit GaussianProcessRegressor (const int dim,
             const double kernel_sigma=1.0, const double likelihood_sigma=1.0);
    // Public member functions
    void train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) override;
    Eigen::MatrixXd predict (const Eigen::MatrixXd & X) const override;
    double evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const override;
};

#endif
