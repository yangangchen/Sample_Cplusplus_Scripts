/*
    class GaussianNaiveBayesClassifier: public Classifier
*/

#include <Eigen/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "classifier.h"

// Constructor: GaussianNaiveBayesClassifier
// mu0, Sigma0, mu1, Sigma1 are constructed by the default constructor (empty matrices)
GaussianNaiveBayesClassifier::GaussianNaiveBayesClassifier (const int dim):
    Classifier{dim}, coeff0{0}, coeff1{0}, pi0{0.5}, pi1{0.5} {
}

// Public member function: train
void GaussianNaiveBayesClassifier::train (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) {
    int N = X.rows();
    int N0 = 0, N1 = 0;
    mu0 = Eigen::MatrixXd::Zero(1, dim);
    mu1 = Eigen::MatrixXd::Zero(1, dim);
    for (int r=0; r<N; r++) {
        if (y(r,0) == 0) {
            N0++;
            mu0 += X.row(r);
        }
        else {
            N1++;
            mu1 += X.row(r);
        }
    }
    pi0 = N0 / N;
    pi1 = N1 / N;
    mu0 /= N0;
    mu1 /= N1;

    Sigma0 = Eigen::MatrixXd::Zero(dim, dim);
    Sigma1 = Eigen::MatrixXd::Zero(dim, dim);
    for (int r=0; r<N; r++) {
        if (y(r,0) == 0) {
            Eigen::MatrixXd x0 = X.row(r) - mu0;
            Sigma0 += x0.transpose() * x0;
        }
        else {
            Eigen::MatrixXd x1 = X.row(r) - mu1;
            Sigma1 += x1.transpose() * x1;
        }
    }
    Sigma0 /= N0;
    Sigma1 /= N1;

    Sigma0_inv = Sigma0.inverse();
    Sigma1_inv = Sigma1.inverse();

    double pi = 3.14159265358979;
    coeff0 = 1 / (std::sqrt(std::pow(2*pi, dim) * Sigma0.determinant()));
    coeff1 = 1 / (std::sqrt(std::pow(2*pi, dim) * Sigma1.determinant()));
}

// Public member function: predict
Eigen::MatrixXi GaussianNaiveBayesClassifier::predict (const Eigen::MatrixXd & X) const {
    int N = X.rows();
    Eigen::MatrixXi y = Eigen::MatrixXi::Zero(N, 1);
    for (int r=0; r<N; r++) {
        Eigen::MatrixXd x0 = X.row(r) - mu0;
        double p0 = coeff0 * std::exp((-0.5 * x0 * Sigma0_inv * x0.transpose())(0, 0));
        Eigen::MatrixXd x1 = X.row(r) - mu1;
        double p1 = coeff1 * std::exp((-0.5 * x1 * Sigma1_inv * x1.transpose())(0, 0));
        // Uncomment these two lines if we want to use Maximum A Posteriori rather than Maximum Likelihood
        // p0 = pi0 * p0;
        // p1 = pi1 * p1;
        if (p1 > p0)
            y(r,0) = 1;
        else
            y(r,0) = 0;
    }
    return y;
}

// Public member function: evaluate_accuracy
double GaussianNaiveBayesClassifier::evaluate_accuracy (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) const {
    return ((predict (X) - y).array() == 0).cast<double>().mean();
}
