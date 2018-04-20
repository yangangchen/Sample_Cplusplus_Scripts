/*
    class LogisticClassifier: public Classifier
*/

#include <Eigen/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "classifier.h"

// Private member function: transform_dim
int LogisticClassifier::transform_dim (const int input_dim, const int input_degree) const {
    int output_dim = 1;
    for (int s=std::max(input_dim,input_degree)+1; s<=input_dim+input_degree; s++) {
        output_dim *= s;
    }
    for (int s=1; s<=std::min(input_dim,input_degree); s++) {
        output_dim /= s;
    }
    return output_dim;
}

// Private member function: build_expn_table_partial
Eigen::MatrixXi LogisticClassifier::build_expn_table_partial (const int input_dim,
                                                              const int expn_sum) const {
    if (expn_sum == 0) {
        return Eigen::MatrixXi::Zero(input_dim, 1);
    }
    else if (input_dim == 1) {
        return expn_sum * Eigen::MatrixXi::Ones(1, 1);
    }
    else {
        int num_cols = transform_dim(input_dim,expn_sum) - transform_dim(input_dim,expn_sum-1);
        Eigen::MatrixXi expn_table_partial(input_dim, num_cols);
        int col_pointer = 0;
        for (int first_expn=0; first_expn<=expn_sum; first_expn++) {
            int remaining_expn_sum = expn_sum - first_expn;
            // Remaining table: input_dim - 1, remaining_expn_sum. Recursive function calls
            Eigen::MatrixXi expn_table_partial_attach = build_expn_table_partial(input_dim - 1, remaining_expn_sum);
            int num_cols_sub = expn_table_partial_attach.cols();
            expn_table_partial.block(0,col_pointer,1,num_cols_sub) = first_expn * Eigen::MatrixXi::Ones(1, num_cols_sub);
            expn_table_partial.block(1,col_pointer,input_dim-1,num_cols_sub) = expn_table_partial_attach;
            col_pointer += num_cols_sub;
        }
        return expn_table_partial;
    }
}

// Private member function: build_expn_table_full
Eigen::MatrixXi LogisticClassifier::build_expn_table_full () const {
    Eigen::MatrixXi expn_table (dim, tf_dim);
    int col_pointer = 0;
    for (int expn_sum=0; expn_sum<=degree; expn_sum++) {
        Eigen::MatrixXi expn_table_partial = build_expn_table_partial (dim, expn_sum);
        // std::cout << "Check expn_table_partial: \n" << expn_table_partial << std::endl;
        int num_cols = expn_table_partial.cols();
        expn_table.block(0,col_pointer,dim,num_cols) = expn_table_partial;
        col_pointer += num_cols;
    }
    // std::cout << "Check expn_table: \n" << expn_table << std::endl;
    return expn_table;
}

// Private member function: transform_data_onecol
Eigen::VectorXd LogisticClassifier::transform_data_onecol (const Eigen::MatrixXd & X,
                                                           const Eigen::VectorXi expn_onecol) const {
    Eigen::VectorXd transformed_X_onecol = Eigen::VectorXd::Ones(X.rows());
    for (int c=0; c<dim; c++) {
        Eigen::ArrayXd z = X.col(c).array().pow((double)expn_onecol(c));
        transformed_X_onecol = transformed_X_onecol.array() * z;
    }
    // std::cout << "Check: \n" << transformed_X_onecol.transpose() << std::endl;
    return transformed_X_onecol;
}

// Private member function: transform_data
Eigen::MatrixXd LogisticClassifier::transform_data (const Eigen::MatrixXd & X) const {
    Eigen::MatrixXi expn_table = build_expn_table_full ();
    Eigen::MatrixXd transformed_X(X.rows(), tf_dim);
    for (int c=0; c<tf_dim; c++) {
        transformed_X.col(c) = transform_data_onecol (X, expn_table.col(c));
    }
    // std::cout << "Check transformed_X: \n" << transformed_X << std::endl;
    return transformed_X;
}

// Private member function: sigmoid
Eigen::MatrixXd LogisticClassifier::sigmoid (const Eigen::MatrixXd & X) const {
    return (1 + (-X).array().exp()).inverse().matrix();
}

// Constructor: LogisticClassifier
LogisticClassifier::LogisticClassifier (const int dim, const int degree, const double lambda):
    Classifier{dim}, degree{degree},
    tf_dim{transform_dim(dim, degree)}, lambda{lambda},
    W{Eigen::MatrixXd::Zero(tf_dim, 1)} {
}

// Public member function: get_param
Eigen::MatrixXd LogisticClassifier::get_param () const {
    return W;
}

// Public member function: train
// Newton's method
void LogisticClassifier::train (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) {
    Eigen::MatrixXd phiX = transform_data (X);
    for (int iter=0; iter<=100; iter++) {
        std::cout << "." << std::flush;
        Eigen::MatrixXd fX = sigmoid(phiX * W);
        Eigen::MatrixXd gradL = phiX.transpose() * (fX - y.cast<double>()) + lambda * W;
        Eigen::MatrixXd R = (fX.array() * (1 - fX.array())).matrix().asDiagonal();
        Eigen::MatrixXd hessianL = phiX.transpose() * R * phiX + lambda * Eigen::MatrixXd::Identity(tf_dim, tf_dim);
        W -= hessianL.partialPivLu().solve(gradL);
        double epsilon = gradL.cwiseAbs().rowwise().sum().maxCoeff();
        if (epsilon < 1e-6) {
            std::cout << "\n"
            << "Logistic classifier converges in " << iter
            << " steps to the residual " << epsilon << std::endl;
            break;
        }
    }
}

// Public member function: predict
Eigen::MatrixXi LogisticClassifier::predict (const Eigen::MatrixXd & X) const {
    Eigen::MatrixXd phiX = transform_data (X);
    Eigen::MatrixXd fX = sigmoid(phiX * W);
    return (fX.array() > 0.5).matrix().cast<int>();
}

// Public member function: evaluate_accuracy
double LogisticClassifier::evaluate_accuracy (const Eigen::MatrixXd & X, const Eigen::MatrixXi & y) const {
    return ((predict (X) - y).array() == 0).cast<double>().mean();
}
