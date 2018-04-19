#include <iostream>
#include <fstream>
#include <vector>
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

double cross_validation (double kernel_sigma) {
    int total_file_num = 10;
    Eigen::VectorXd cv_errors(total_file_num);
    Regressor M(2, 1, kernel_sigma);

    for (int test_file_num=1; test_file_num<=total_file_num; test_file_num++) {
        Eigen::MatrixXd X_train = load_train_once (total_file_num, test_file_num, "Data");
        Eigen::MatrixXd y_train = load_train_once (total_file_num, test_file_num, "Labels");
        Eigen::MatrixXd X_test = load_test_once (test_file_num, "Data");
        Eigen::MatrixXd y_test = load_test_once (test_file_num, "Labels");

        M.train(X_train, y_train);
        // std::cout << "The coefficient matrix is: \n" << M.get_param() << std::endl;
        cv_errors(test_file_num - 1) = M.evaluate_error(X_test, y_test);
    }
    return cv_errors.mean();
}

Eigen::VectorXd model_selection () {
    Eigen::MatrixXd results_sigmas_errors(2, 10);
    for (int k=0; k<10; k++) {
        double kernel_sigma = 0.5 * (k + 1);
        std::cout << kernel_sigma << std::endl;
        results_sigmas_errors(0, k) = kernel_sigma;
        results_sigmas_errors(1, k) = cross_validation(kernel_sigma);
    }
    // std::cout << results_sigmas_errors << std::endl;
    Eigen::VectorXd::Index j;
    results_sigmas_errors.row(1).minCoeff(&j);
    Eigen::VectorXd optimal_sigma_error = results_sigmas_errors.col(j);
    // std::cout << optimal_sigma_error << std::endl;
    return optimal_sigma_error;
}

int main () {
    Eigen::VectorXd optimal_sigma_error = model_selection ();
    std::ofstream myfile;
    myfile.open("results.txt");
    myfile << "The optimal kernel_sigma is: " << optimal_sigma_error(0) << "\n"
           << "The optimal error is: " << optimal_sigma_error(1) << "\n";
    myfile.close();
    return 0;
}
