#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <Eigen/Eigen/Dense>
#include "regressor.h"

template <typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

Eigen::MatrixXd load_train_once (int total_file_num, int test_file_num, std::string s="Data") {
    int flag = 1;
    Eigen::MatrixXd X;
    for (int train_file_num=1; train_file_num<=total_file_num; train_file_num++) {
        if (train_file_num != test_file_num) {
            if (flag) {
                X = load_csv<Eigen::MatrixXd>("regression-dataset/f" + s + std::to_string(train_file_num) + ".csv");
                flag = 0;
            }
            else {
                Eigen::MatrixXd X0 = load_csv<Eigen::MatrixXd>("regression-dataset/f" + s + std::to_string(train_file_num) + ".csv");
                int n = X.rows();
                X.conservativeResize(X.rows() + X0.rows(), Eigen::NoChange);
                X.block(n,0,X0.rows(),X0.cols()) = X0;
            }
        }
    }
    return X;
}

Eigen::MatrixXd load_test_once (int test_file_num, std::string s="Data") {
    Eigen::MatrixXd X = load_csv<Eigen::MatrixXd>("regression-dataset/f" + s + std::to_string(test_file_num) + ".csv");
    return X;
}

double cross_validation (Regressor* M) {
    int total_file_num = 10;
    Eigen::VectorXd cv_errors(total_file_num);

    for (int test_file_num=1; test_file_num<=total_file_num; test_file_num++) {
        Eigen::MatrixXd X_train = load_train_once (total_file_num, test_file_num, "Data");
        Eigen::MatrixXd y_train = load_train_once (total_file_num, test_file_num, "Labels");
        Eigen::MatrixXd X_test = load_test_once (test_file_num, "Data");
        Eigen::MatrixXd y_test = load_test_once (test_file_num, "Labels");

        M -> train(X_train, y_train);
        cv_errors(test_file_num - 1) = M -> evaluate_error(X_test, y_test);
    }
    return cv_errors.mean();
}

int main () {
    PolynomialRegressor M1(2, 3, 0.58);
    double error1 = cross_validation(&M1);
    PolynomialRegressor M11 = M1;
    double error11 = cross_validation(&M11);
    PolynomialRegressor M12(2);
    M12 = M1;
    double error12 = cross_validation(&M12);

    GaussianProcessRegressor M2(2, 1, 4.0, 1.0);
    double error2 = cross_validation(&M2);
    GaussianProcessRegressor M21 = M2;
    GaussianProcessRegressor M22(2);
    M22 = M2;
    // M2.deepcopy_reset(); M2.deepcopy_get_param(); M21.deepcopy_get_param(); M22.deepcopy_get_param();
    double error21 = cross_validation(&M21);
    double error22 = cross_validation(&M22);

    std::ofstream myfile;
    myfile.open("results.txt");
    myfile << "The error for Polynomial Regressor is: " << error1 << " " << error11 << " " << error12 << "\n"
           << "The error for Gaussian Process Regressor is: " << error2 << " " << error21 << " " << error22 << "\n";
    myfile.close();
    return 0;
}
