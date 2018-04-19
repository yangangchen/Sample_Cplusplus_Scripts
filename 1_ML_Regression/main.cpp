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
    Dataset<Eigen::MatrixXd> dataset("dataset/fData.csv", "dataset/fLabels.csv");
    dataset.train_test_split(0.72);

    PolynomialRegressor model1(2, 3, 0.58);
    GaussianProcessRegressor model2(2, 1, 4.0, 1.0);

    int K = 10;
    double error1 = dataset.cross_validation(&model1, K);
    double error2 = dataset.cross_validation(&model2, K);

    std::ofstream myfile;
    myfile.open("results.txt");
    myfile << "The " << K << "-fold cross-validation error for Polynomial Regressor is: " << error1 << "\n"
           << "The " << K << "-fold cross-validation error for Gaussian Process Regressor is: " << error2 << "\n";
    myfile.close();
    return 0;
}
