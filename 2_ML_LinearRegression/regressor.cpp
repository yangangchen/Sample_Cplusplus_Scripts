#include <Eigen/Eigen/Dense>
#include <iostream>
#include "regressor.h"

void Regressor::enlarge_data (Eigen::MatrixXd & X) {
    if (X.cols() == dim) {
        X.conservativeResize(Eigen::NoChange, dim + 1);
        X.col(dim).setOnes();
    }
}

Regressor::Regressor (const int dim, const double lambda):
    dim{dim}, lambda{lambda}, W{Eigen::MatrixXd::Zero(dim + 1, 1)} {
    // std::cout << "The initial coefficient matrix is: \n" << W << std::endl;
}

Eigen::MatrixXd Regressor::get_param () {
    return W;
}

void Regressor::train (Eigen::MatrixXd & X, Eigen::MatrixXd & y) {
    enlarge_data (X);
    Eigen::MatrixXd A = X.transpose() * X + lambda * Eigen::MatrixXd::Identity(dim + 1, dim + 1);
    Eigen::MatrixXd b = X.transpose() * y;
    W = A.partialPivLu().solve(b);
}

Eigen::MatrixXd Regressor::predict (Eigen::MatrixXd & X) {
    enlarge_data (X);
    return X * W;
}

double Regressor::evaluate_error (Eigen::MatrixXd & X, Eigen::MatrixXd & y) {
    return 0.5 * (predict (X) - y).array().square().mean();
}
