#include <Eigen/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "regressor.h"

Eigen::MatrixXd Regressor::kernel_matrix (Eigen::MatrixXd & X1, Eigen::MatrixXd & X2) {
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

Regressor::Regressor (const int dim, const int N, const double kernel_sigma, const double likelihood_sigma):
    dim{dim}, N{N}, kernel_sigma{kernel_sigma}, likelihood_sigma{likelihood_sigma},
    Data{Eigen::MatrixXd::Zero(N,dim)}, Vec{Eigen::MatrixXd::Zero(N,1)} {
}

void Regressor::train (Eigen::MatrixXd & X, Eigen::MatrixXd & y) {
    N = X.rows();
    Data = X;
    Eigen::MatrixXd Mat = kernel_matrix (Data, Data) + likelihood_sigma * Eigen::MatrixXd::Identity(N, N);
    Vec = Mat.partialPivLu().solve(y);
}

Eigen::MatrixXd Regressor::predict (Eigen::MatrixXd & X) {
    return kernel_matrix (X, Data) * Vec;
}

double Regressor::evaluate_error (Eigen::MatrixXd & X, Eigen::MatrixXd & y) {
    return 0.5 * (predict (X) - y).array().square().mean();
}
