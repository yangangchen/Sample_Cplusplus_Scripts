#ifndef _MODEL_H_
#define _MODEL_H_

#include <Eigen/Eigen/Dense>

class Model
{
  private:

    // Declaration
    Eigen::MatrixXd W;
    const int dim;
    const double lambda;

    // Private member functions
    void enlarge_data(Eigen::MatrixXd & X);

  public:

    // Constructor
    explicit Model (const int dim, const double lambda=0.0);

    // Public member functions
    Eigen::MatrixXd get_param ();
    void train (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
    Eigen::MatrixXd prediction (Eigen::MatrixXd & X);
    double evaluate_error (Eigen::MatrixXd & X, Eigen::MatrixXd & y);
};

#endif
