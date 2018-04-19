/*
    class Dataset
*/

#ifndef _DATASET_H_
#define _DATASET_H_

#include "regressor.h"

template <typename T>
class Dataset
{
  public:

    // Declaration
    T X;
    T y;
    T X_train;
    T y_train;
    T X_test;
    T y_test;
    T X_Kfoldcv_train;
    T y_Kfoldcv_train;
    T X_Kfoldcv_cv;
    T y_Kfoldcv_cv;
    const int dim;
    const int N;
    int N_train;
    int N_test;

    // Constructor
    Dataset (const std::string & X_path, const std::string & y_path);

    // Public member functions
    T load_csv (const std::string & path);
    void shuffle ();
    void train_test_split (const double percent);
    void Kfoldcv_split (const int K, const int k);
    double cross_validation (Regressor* model, const int K);
    double train_test_evaluation (Regressor* model);
};

#endif
