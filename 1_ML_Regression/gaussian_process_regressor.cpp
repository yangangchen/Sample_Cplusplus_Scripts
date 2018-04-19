/*
    class GaussianProcessRegressor: public Regressor
*/

#include <Eigen/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "regressor.h"

// Private member function: kernel_matrix
Eigen::MatrixXd GaussianProcessRegressor::kernel_matrix (const Eigen::MatrixXd & X1,
                                                         const Eigen::MatrixXd & X2) const {
    int n1 = X1.rows();
    int n2 = X2.rows();
    Eigen::MatrixXd K(n1, n2);
    for (int i1=0; i1<n1; i1++) {
        for (int i2=0; i2<n2; i2++) {
            K(i1, i2) = std::exp(- 0.5 / pow(kernel_sigma, 2.0) * (X1.row(i1) - X2.row(i2)).squaredNorm());
        }
    }
    return K;
}

// Constructor: GaussianProcessRegressor
GaussianProcessRegressor::GaussianProcessRegressor (const int dim, const int N,
                                                    const double kernel_sigma, const double likelihood_sigma):
    Regressor{dim}, N{N}, kernel_sigma{kernel_sigma}, likelihood_sigma{likelihood_sigma},
    Data{Eigen::MatrixXd::Zero(N,dim)}, Vec{Eigen::MatrixXd::Zero(N,1)} {
}

// Public member function: train
void GaussianProcessRegressor::train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) {
    N = X.rows();
    Data = X;
    Eigen::MatrixXd Mat = kernel_matrix (Data, Data) + likelihood_sigma * Eigen::MatrixXd::Identity(N, N);
    Vec = Mat.partialPivLu().solve(y);
}

// Public member function: predict
Eigen::MatrixXd GaussianProcessRegressor::predict (const Eigen::MatrixXd & X) const {
    return kernel_matrix (X, Data) * Vec;
}

// Public member function: evaluate_error
double GaussianProcessRegressor::evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const {
    return 0.5 * (predict (X) - y).array().square().mean();
}
