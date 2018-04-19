/*
    Main script
*/

#include "dataset.h"
#include "regressor.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <Eigen/Eigen/Dense>

int main () {
    Dataset<Eigen::MatrixXd, Eigen::MatrixXd> dataset("Dataset/fData.csv", "Dataset/fLabels.csv");
    dataset.train_test_split(0.8);

    PolynomialRegressor model1(dataset.dim, 3, 0.58);  // (dataset.dim, degree, lambda)
    GaussianProcessRegressor model2(dataset.dim, 4.0, 1.0);  // (dataset.dim, kernel_sigma, likelihood_sigma)

    int K = 10;
    double model1_cv_error = dataset.cross_validation(&model1, K);
    double model1_test_error = dataset.train_test_evaluation(&model1);

    double model2_cv_error = dataset.cross_validation(&model2, K);
    double model2_test_error = dataset.train_test_evaluation(&model2);

    std::ofstream myfile;
    myfile.open("results.txt");
    myfile << "The " << K << "-fold cross-validation error for Polynomial Regressor is: " << model1_cv_error << "\n"
           << "The test error for Polynomial Regressor is: " << model1_test_error << "\n"
           << "The " << K << "-fold cross-validation error for Gaussian Process Regressor is: " << model2_cv_error << "\n"
           << "The test error for Gaussian Process Regressor is: " << model2_test_error << "\n";
    myfile.close();
    return 0;
}
