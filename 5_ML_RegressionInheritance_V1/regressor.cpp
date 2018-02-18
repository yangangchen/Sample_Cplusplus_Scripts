#include <Eigen/Eigen/Dense>
#include <iostream>
#include <algorithm>
#include "regressor.h"

/*
                   class Regressor
*/
Regressor::Regressor (int dim):
    dim{dim} {
}

Regressor::~Regressor () {
}

Regressor::Regressor (const Regressor & other):
    dim{other.dim} {
}

// Copy-and-swap-idiom
void Regressor::swap(Regressor & temp)
{
    using std::swap;
    swap(dim, temp.dim);
}

// Cannot use copy-and-swap-idiom on the abstract class!
/*
Regressor & Regressor::operator = (const Regressor & other)
{
    Regressor temp = other;
    swap(temp);
    return *this;
}
*/

Regressor & Regressor::operator = (const Regressor & other)
{
    dim = other.dim;
    return *this;
}

Regressor::Regressor (Regressor && other):
    dim{std::move(other.dim)} {
}

Regressor & Regressor::operator = (Regressor && other)
{
    using std::swap;
    swap(dim, other.dim);
    return *this;
}

/*
             class GeneralizedLinearRegressor: public Regressor
*/
int GeneralizedLinearRegressor::transform_dim (int input_dim, int input_degree) const {
    int output_dim = 1;
    for (int s=std::max(input_dim,input_degree)+1; s<=input_dim+input_degree; s++) {
        output_dim *= s;
    }
    for (int s=1; s<=std::min(input_dim,input_degree); s++) {
        output_dim /= s;
    }
    return output_dim;
}

Eigen::MatrixXi GeneralizedLinearRegressor::build_expn_table_partial (int input_dim,
                                                                      int expn_sum) const {
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

Eigen::MatrixXi GeneralizedLinearRegressor::build_expn_table_full () const {
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

Eigen::VectorXd GeneralizedLinearRegressor::transform_data_onecol (const Eigen::MatrixXd & X,
                                                                   const Eigen::VectorXi & expn_onecol) const {
    Eigen::VectorXd transformed_X_onecol = Eigen::VectorXd::Ones(X.rows());
    for (int c=0; c<dim; c++) {
        Eigen::ArrayXd z = X.col(c).array().pow((double)expn_onecol(c));
        transformed_X_onecol = transformed_X_onecol.array() * z;
    }
    // std::cout << "Check: \n" << transformed_X_onecol.transpose() << std::endl;
    return transformed_X_onecol;
}

Eigen::MatrixXd GeneralizedLinearRegressor::transform_data (const Eigen::MatrixXd & X) const {
    Eigen::MatrixXi expn_table = build_expn_table_full ();
    Eigen::MatrixXd transformed_X(X.rows(), tf_dim);
    for (int c=0; c<tf_dim; c++) {
        transformed_X.col(c) = transform_data_onecol (X, expn_table.col(c));
    }
    // std::cout << "Check transformed_X: \n" << transformed_X << std::endl;
    return transformed_X;
}

GeneralizedLinearRegressor::GeneralizedLinearRegressor (int dim, int degree, double lambda):
    Regressor{dim}, degree{degree},
    tf_dim{transform_dim(dim, degree)}, lambda{lambda},
    W{Eigen::MatrixXd::Zero(tf_dim, 1)} {
}

GeneralizedLinearRegressor::~GeneralizedLinearRegressor () {
}

GeneralizedLinearRegressor::GeneralizedLinearRegressor (const GeneralizedLinearRegressor & other):
    Regressor{other}, degree{other.degree},
    tf_dim{other.tf_dim}, lambda{other.lambda}, W{other.W} {
}

// Copy-and-swap-idiom
void GeneralizedLinearRegressor::swap(GeneralizedLinearRegressor & temp)
{
    using std::swap;
    Regressor::swap(temp);
    swap(degree, temp.degree);
    swap(tf_dim, temp.tf_dim);
    swap(lambda, temp.lambda);
    swap(W, temp.W);
}

GeneralizedLinearRegressor & GeneralizedLinearRegressor::operator = (const GeneralizedLinearRegressor & other)
{
    GeneralizedLinearRegressor temp = other;
    swap(temp);
    return *this;
}

GeneralizedLinearRegressor::GeneralizedLinearRegressor (GeneralizedLinearRegressor && other):
    Regressor{std::move(other)}, degree{std::move(other.degree)},
    tf_dim{std::move(other.tf_dim)}, lambda{std::move(other.lambda)}, W{std::move(other.W)} {
}

GeneralizedLinearRegressor & GeneralizedLinearRegressor::operator = (GeneralizedLinearRegressor && other)
{
    using std::swap;
    Regressor::swap(other);
    swap(degree, other.degree);
    swap(tf_dim, other.tf_dim);
    swap(lambda, other.lambda);
    swap(W, other.W);
    return *this;
}

Eigen::MatrixXd GeneralizedLinearRegressor::get_param () const {
    return W;
}

void GeneralizedLinearRegressor::train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) {
    Eigen::MatrixXd phiX = transform_data (X);
    Eigen::MatrixXd A = phiX.transpose() * phiX + lambda * Eigen::MatrixXd::Identity(tf_dim, tf_dim);
    Eigen::MatrixXd b = phiX.transpose() * y;
    W = A.partialPivLu().solve(b);
}

Eigen::MatrixXd GeneralizedLinearRegressor::predict (const Eigen::MatrixXd & X) const {
    Eigen::MatrixXd phiX = transform_data (X);
    return phiX * W;
}

double GeneralizedLinearRegressor::evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const {
    return 0.5 * (predict (X) - y).array().square().mean();
}

/*
             class GaussianProcessRegressor: public Regressor
*/
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

GaussianProcessRegressor::GaussianProcessRegressor (int dim, int N,
                                                    double kernel_sigma, double likelihood_sigma):
    Regressor{dim}, N{N}, kernel_sigma{kernel_sigma}, likelihood_sigma{likelihood_sigma},
    Data{nullptr}, Vec{nullptr} {
}

GaussianProcessRegressor::~GaussianProcessRegressor () {
    delete Data;
    delete Vec;
}

GaussianProcessRegressor::GaussianProcessRegressor (const GaussianProcessRegressor & other):
    Regressor{other}, N{other.N}, kernel_sigma{other.kernel_sigma}, likelihood_sigma{other.likelihood_sigma},
    Data{other.Data ? new Eigen::MatrixXd{*other.Data} : nullptr},
    Vec{other.Vec ? new Eigen::MatrixXd{*other.Vec} : nullptr} {
}

// Copy-and-swap-idiom
void GaussianProcessRegressor::swap(GaussianProcessRegressor & temp)
{
    using std::swap;
    Regressor::swap(temp);
    swap(N, temp.N);
    swap(kernel_sigma, temp.kernel_sigma);
    swap(likelihood_sigma, temp.likelihood_sigma);
    swap(Data, temp.Data);
    swap(Vec, temp.Vec);
}

GaussianProcessRegressor & GaussianProcessRegressor::operator = (const GaussianProcessRegressor & other)
{
    GaussianProcessRegressor temp = other;
    swap(temp);
    return *this;
}

GaussianProcessRegressor::GaussianProcessRegressor (GaussianProcessRegressor && other):
    Regressor{std::move(other)}, N{std::move(other.N)}, kernel_sigma{std::move(other.kernel_sigma)},
    likelihood_sigma{std::move(other.likelihood_sigma)},
    Data{std::move(other.Data)}, Vec{std::move(other.Vec)} {
}

GaussianProcessRegressor & GaussianProcessRegressor::operator = (GaussianProcessRegressor && other)
{
    using std::swap;
    Regressor::swap(other);
    swap(N, other.N);
    swap(kernel_sigma, other.kernel_sigma);
    swap(likelihood_sigma, other.likelihood_sigma);
    swap(Data, other.Data);
    swap(Vec, other.Vec);
    return *this;
}

void GaussianProcessRegressor::train (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) {
    N = X.rows();
    Data = new Eigen::MatrixXd{X};
    Eigen::MatrixXd Mat = kernel_matrix (*Data, *Data) + likelihood_sigma * Eigen::MatrixXd::Identity(N, N);
    Vec = new Eigen::MatrixXd{Mat.partialPivLu().solve(y)};
}

Eigen::MatrixXd GaussianProcessRegressor::predict (const Eigen::MatrixXd & X) const {
    return kernel_matrix (X, *Data) * (*Vec);
}

double GaussianProcessRegressor::evaluate_error (const Eigen::MatrixXd & X, const Eigen::MatrixXd & y) const {
    return 0.5 * (predict (X) - y).array().square().mean();
}

/*
// Test whether we can get a deep copy.
void GaussianProcessRegressor::deepcopy_reset () {
    *Vec = Eigen::MatrixXd::Zero(N, 1);
}

void GaussianProcessRegressor::deepcopy_get_param () {
    std::cout << Vec->transpose() << "\n" << std::endl;
}
*/
