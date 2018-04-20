/*
    class Dataset
*/

#include <Eigen/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "dataset.h"
#include "classifier.h"

// Constructor: Dataset
// X_train, y_train, X_test, y_test, etc are constructed by the default constructor (empty matrices)
template<typename T1, typename T2>
Dataset<T1,T2>::Dataset (const std::string & X_path, const std::string & y_path):
    X{load_csv<T1>(X_path)}, y{load_csv<T2>(y_path)}, dim{X.cols()}, N{X.rows()}, N_train{0}, N_test{0} {
}

// Public member function: load_csv
// Note: T::Map (values.data(), rows, columns) uses column major order!
template<typename T1, typename T2>
template<typename T>
T Dataset<T1,T2>::load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<typename T::Scalar> values;
    int rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    int columns = values.size()/rows;
    return Eigen::Map<
               Eigen::Matrix<typename T::Scalar,
                             T::RowsAtCompileTime, T::ColsAtCompileTime, Eigen::RowMajor>
           >(values.data(), rows, columns);
}

// Public member function: shuffle
template<typename T1, typename T2>
void Dataset<T1,T2>::shuffle () {
    std::srand(std::time(NULL));
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(N, 0, N);
    std::random_shuffle(indices.data(), indices.data() + N);
    X = indices.asPermutation() * X;
    y = indices.asPermutation() * y;
}

// Public member function: train_test_split
template<typename T1, typename T2>
void Dataset<T1,T2>::train_test_split (const double percent) {
    shuffle();
    N_train = N * percent;
    N_test = N - N_train;
    X_train = X.topRows(N_train);
    y_train = y.topRows(N_train);
    X_test = X.bottomRows(N_test);
    y_test = y.bottomRows(N_test);
    // std::cout << X_train.rows() << " " << X_train.cols() << std::endl;
    // std::cout << X_test.rows() << " " << X_test.cols() << std::endl;
    // std::cout << y_train.rows() << " " << y_train.cols() << std::endl;
    // std::cout << y_test.rows() << " " << y_test.cols() << std::endl;
    // std::cout << "y: \n" << y.transpose() << std::endl;
    // std::cout << "y_train: \n" << y_train.transpose() << std::endl;
    // std::cout << "y_test: \n" << y_test.transpose() << std::endl;
}

// Public member function: kfoldcv_split
template<typename T1, typename T2>
void Dataset<T1,T2>::Kfoldcv_split (const int K, const int k) {
    long int n = N_train / K;
    if (k >= 0 && k < K - 1) {
        X_Kfoldcv_cv = X_train.block(n*k, 0, n, dim);
        X_Kfoldcv_train = T1::Zero(N_train-n, dim);
        X_Kfoldcv_train << X_train.topRows(n*k),
                           X_train.bottomRows(N_train-n*(k+1));
        y_Kfoldcv_cv = y_train.block(n*k, 0, n, 1);
        y_Kfoldcv_train = T2::Zero(N_train-n, 1);
        y_Kfoldcv_train << y_train.topRows(n*k),
                           y_train.bottomRows(N_train-n*(k+1));
    }
    else if (k == K - 1) {
        X_Kfoldcv_cv = X_train.bottomRows(N_train-n*k);
        X_Kfoldcv_train = X_train.topRows(n*k);
        y_Kfoldcv_cv = y_train.bottomRows(N_train-n*k);
        y_Kfoldcv_train = y_train.topRows(n*k);
    }
    else {
        std::cout << "k must be an integer between 0 and K-1!" << std::endl;
    }
    // std::cout << X_Kfoldcv_train.rows() << " " << X_Kfoldcv_train.cols() << std::endl;
    // std::cout << X_Kfoldcv_cv.rows() << " " << X_Kfoldcv_cv.cols() << std::endl;
    // std::cout << y_Kfoldcv_train.rows() << " " << y_Kfoldcv_train.cols() << std::endl;
    // std::cout << y_Kfoldcv_cv.rows() << " " << y_Kfoldcv_cv.cols() << std::endl;
    // std::cout << "y_train: \n" << y_Kfoldcv_train.transpose() << std::endl;
    // std::cout << "y_Kfoldcv_train: \n" << y_Kfoldcv_train.transpose() << std::endl;
    // std::cout << "y_Kfoldcv_cv: \n" << y_Kfoldcv_cv.transpose() << std::endl;
}

// Public member function: cross_validation
// Must cast the types T1 & T2 to the type Eigen::MatrixXd & Eigen::MatrixXi using X.template cast<double>() & y.template cast<int>()
template<typename T1, typename T2>
double Dataset<T1,T2>::cross_validation (Classifier* model, const int K) {
    Eigen::VectorXd cv_errors(K);
    for (int k=0; k<K; k++) {
        Kfoldcv_split(K,k);
        model -> train(X_Kfoldcv_train.template cast<double>(),
                       y_Kfoldcv_train.template cast<int>());
        cv_errors(k) = model -> evaluate_accuracy(X_Kfoldcv_cv.template cast<double>(),
                                                  y_Kfoldcv_cv.template cast<int>());
    }
    return cv_errors.mean();
}

// Public member function: model_train_test
template<typename T1, typename T2>
double Dataset<T1,T2>::train_test_evaluation (Classifier* model) const {
    model -> train(X_train.template cast<double>(),
                   y_train.template cast<int>());
    return model -> evaluate_accuracy(X_test.template cast<double>(),
                                      y_test.template cast<int>());
}

// Explicit instantiations
// See https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
template class Dataset<Eigen::MatrixXd, Eigen::MatrixXd>;
template class Dataset<Eigen::MatrixXd, Eigen::MatrixXi>;
