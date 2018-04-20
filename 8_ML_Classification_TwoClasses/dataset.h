/*
    class Dataset
*/

#ifndef _DATASET_H_
#define _DATASET_H_

#include "classifier.h"

template <typename T1, typename T2>
class Dataset
{
  public:

    // Declaration
    T1 X;
    T2 y;
    T1 X_train;
    T2 y_train;
    T1 X_test;
    T2 y_test;
    T1 X_Kfoldcv_train;
    T2 y_Kfoldcv_train;
    T1 X_Kfoldcv_cv;
    T2 y_Kfoldcv_cv;
    const long int dim;
    const long int N;
    long int N_train;
    long int N_test;

    // Constructor
    Dataset (const std::string & X_path, const std::string & y_path);

    // Public member functions
    template <typename T> T load_csv (const std::string & path);
    void shuffle ();
    void train_test_split (const double percent);
    void Kfoldcv_split (const int K, const int k);
    double cross_validation (Classifier* model, const int K);
    double train_test_evaluation (Classifier* model) const;
};

#endif
