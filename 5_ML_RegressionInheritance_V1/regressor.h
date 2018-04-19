#ifndef _REGRESSOR_H_
#define _REGRESSOR_H_

#include <memory>
#include <Eigen/Eigen/Dense>

class Regressor
{
  protected:

    // Declaration
    int dim;

  public:

    // Constructor
    Regressor (int dim);
    // Destructor
    virtual ~Regressor ();
    // Copy constructor
    Regressor (const Regressor & other);
    // Copy assignment operator
    void swap (Regressor & temp);
    Regressor & operator = (const Regressor & other);
    // Move constructor
    Regressor (Regressor && other);
    // Move assignment operator
    Regressor & operator = (Regressor && other);

    // Public member functions
    virtual void train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) = 0;
    virtual Eigen::MatrixXd predict (const Eigen::MatrixXd & X) const = 0;
    virtual double evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const = 0;
};

class PolynomialRegressor: public Regressor
{
  private:

    // Declaration
    int degree;
    int tf_dim;
    double lambda;
    Eigen::MatrixXd W;

    // Private member functions
    int transform_dim (int dim, int degree) const;
    Eigen::MatrixXi build_expn_table_partial (int input_dim, int expn_sum) const;
    Eigen::MatrixXi build_expn_table_full () const;
    Eigen::VectorXd transform_data_onecol (const Eigen::MatrixXd & X, const Eigen::VectorXi & expn_onecol) const;
    Eigen::MatrixXd transform_data (const Eigen::MatrixXd & X) const;

  public:

    // Constructor
    PolynomialRegressor (int dim, int degree=1, double lambda=0.0);
    // Destructor
    ~PolynomialRegressor ();
    // Copy constructor
    PolynomialRegressor (const PolynomialRegressor & other);
    // Copy assignment operator
    void swap (PolynomialRegressor & temp);
    PolynomialRegressor & operator = (const PolynomialRegressor & other);
    // Move constructor
    PolynomialRegressor (PolynomialRegressor && other);
    // Move assignment operator
    PolynomialRegressor & operator = (PolynomialRegressor && other);

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
    double kernel_sigma;
    double likelihood_sigma;
    const Eigen::MatrixXd * Data; // std::unique_ptr<const Eigen::MatrixXd> Data;
    const Eigen::MatrixXd * Vec; // std::unique_ptr<const Eigen::MatrixXd> Vec;
    // Eigen::MatrixXd * Vec; // std::unique_ptr<Eigen::MatrixXd> Vec;

    // Private member functions
    Eigen::MatrixXd kernel_matrix (const Eigen::MatrixXd & X1, const Eigen::MatrixXd & X2) const;

  public:

    // Constructor
    GaussianProcessRegressor (int dim, int N=1,
             double kernel_sigma=1.0, double likelihood_sigma=1.0);
    // Destructor
    ~GaussianProcessRegressor ();
    // Copy constructor
    GaussianProcessRegressor (const GaussianProcessRegressor & other);
    // Copy assignment operator
    void swap (GaussianProcessRegressor & temp);
    GaussianProcessRegressor & operator = (const GaussianProcessRegressor & other);
    // Move constructor
    GaussianProcessRegressor (GaussianProcessRegressor && other);
    // Move assignment operator
    GaussianProcessRegressor & operator = (GaussianProcessRegressor && other);

    // Public member functions
    void train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) override;
    Eigen::MatrixXd predict (const Eigen::MatrixXd & X) const override;
    double evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const override;

    /*
    // Test whether we can get a deep copy. To do the test, change "const Eigen::MatrixXd * Vec" to "Eigen::MatrixXd * Vec"
    void deepcopy_reset ();
    void deepcopy_get_param ();
    */
};

#endif
