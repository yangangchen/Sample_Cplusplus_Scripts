/*
    Main script
*/

#include "dataset.h"
#include "classifier.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <Eigen/Eigen/Dense>

int main () {
    Dataset<Eigen::MatrixXd, Eigen::MatrixXi> dataset("Dataset/fData.csv", "Dataset/fLabels.csv");
    dataset.train_test_split(0.8);

    LogisticClassifier model1(dataset.dim, 1, 0.0);  // (dataset.dim, degree, lambda)
    GaussianNaiveBayesClassifier model2(dataset.dim);  // (dataset.dim)
    
    int K = 10;
    double model1_cv_accuracy = dataset.cross_validation(&model1, K);
    double model1_test_accuracy = dataset.train_test_evaluation(&model1);

    double model2_cv_accuracy = dataset.cross_validation(&model2, K);
    double model2_test_accuracy = dataset.train_test_evaluation(&model2);

    std::ofstream myfile;
    myfile.open("results.txt");
    myfile << "The " << K << "-fold cross-validation accuracy for Logistic Classifier is: "
           << model1_cv_accuracy * 100 << "%\n"
           << "The test accuracy for Logistic Classifier is: "
           << model1_test_accuracy * 100 << "%\n"
           << "The " << K << "-fold cross-validation accuracy for Gaussian Naive Bayes Classifier is: "
           << model2_cv_accuracy * 100 << "%\n"
           << "The test accuracy for Gaussian Naive Bayes Classifier is: "
           << model2_test_accuracy * 100 << "%\n";
    myfile.close();
    return 0;
}
