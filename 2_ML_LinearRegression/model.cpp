#include <Eigen/Eigen/Dense>
#include <iostream>
#include "model.h"

void Model::enlarge_data (Eigen::MatrixXd & X) {
    if (X.cols() == dim) {
        X.conservativeResize(Eigen::NoChange, dim + 1);
        X.col(dim).setOnes();
    }
}

Model::Model (const int dim, const double lambda):
    W{Eigen::MatrixXd(dim + 1, 1)}, dim{dim}, lambda{lambda} {
    // std::cout << "The initial coefficient matrix is: \n" << W << std::endl;
}

Eigen::MatrixXd Model::get_param () {
    return W;
}

void Model::train (Eigen::MatrixXd & X, Eigen::MatrixXd & y) {
    enlarge_data (X);
    Eigen::MatrixXd A = X.transpose() * X + lambda * Eigen::MatrixXd::Identity(dim + 1, dim + 1);
    Eigen::MatrixXd b = X.transpose() * y;
    W = A.partialPivLu().solve(b);
}

Eigen::MatrixXd Model::prediction (Eigen::MatrixXd & X) {
    enlarge_data (X);
    return X * W;
}

double Model::evaluate_error (Eigen::MatrixXd & X, Eigen::MatrixXd & y) {
    return 0.5 * (prediction (X) - y).array().square().mean();
}
