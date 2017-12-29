#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Eigen/Dense>
#include "model.h"

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

double cross_validation (double lambda) {
    int total_file_num = 10;
    Eigen::VectorXd cv_errors(total_file_num);
    Model M(2, lambda);  // int dim = X_train.cols();

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
    Eigen::MatrixXd results_lambdas_errors(2, 41);
    for (int k=0; k<=40; k++) {
        double lambda = 0.1 * k;
        results_lambdas_errors(0, k) = lambda;
        results_lambdas_errors(1, k) = cross_validation(lambda);
    }
    // std::cout << results_lambdas_errors << std::endl;
    Eigen::VectorXd::Index j;
    results_lambdas_errors.row(1).minCoeff(&j);
    Eigen::VectorXd optimal_lambda_error = results_lambdas_errors.col(j);
    // std::cout << optimal_lambda_error << std::endl;
    return optimal_lambda_error;
}

int main () {
    Eigen::VectorXd optimal_lambda_error = model_selection ();
    std::ofstream myfile;
    myfile.open("results.txt");
    myfile << "The optimal lambda is: " << optimal_lambda_error(0) << "\n"
           << "The optimal error is: " << optimal_lambda_error(1) << "\n";
    myfile.close();
    return 0;
}
