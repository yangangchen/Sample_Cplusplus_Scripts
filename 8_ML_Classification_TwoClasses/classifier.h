/*
    class Classifier
    class LogisticClassifier: public Classifier
    class GaussianNaiveBayesClassifier: public Classifier
*/

#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <Eigen/Eigen/Dense>

class Classifier
{
  protected:

    // Declaration
    const int dim;

  public:

    // Constructor
    explicit Classifier (const int dim);
    // Public member functions
    virtual void train (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) = 0;
    virtual Eigen::MatrixXi predict (const Eigen::MatrixXd & X) const = 0;
    virtual double evaluate_accuracy (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) const = 0;
};

class LogisticClassifier: public Classifier
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
    Eigen::MatrixXd sigmoid (const Eigen::MatrixXd & X) const;

  public:

    // Constructor
    explicit LogisticClassifier (const int dim, const int degree=1, const double lambda=0.0);
    // Public member functions
    Eigen::MatrixXd get_param () const;
    void train (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) override;
    Eigen::MatrixXi predict (const Eigen::MatrixXd & X) const override;
    double evaluate_accuracy (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) const override;
};

class GaussianNaiveBayesClassifier: public Classifier
{
  private:

    // Declaration
    double coeff0;
    double coeff1;
    double pi0;
    double pi1;
    Eigen::MatrixXd mu0;
    Eigen::MatrixXd Sigma0;
    Eigen::MatrixXd Sigma0_inv;
    Eigen::MatrixXd mu1;
    Eigen::MatrixXd Sigma1;
    Eigen::MatrixXd Sigma1_inv;

  public:

    // Constructor
    explicit GaussianNaiveBayesClassifier (const int dim);
    // Public member functions
    void train (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) override;
    Eigen::MatrixXi predict (const Eigen::MatrixXd & X) const override;
    double evaluate_accuracy (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) const override;
};

#endif
