# Sample C++ Scripts

Copyright (C) 2017  Yangang Chen

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

**Author: Yangang Chen. My sample C++ scripts.**

**Contents:**

1_EclipseTest: Test script on Eclipse IDE.

2_ML_LinearRegression: Linear regression with the selection of regularization parameter lambda.

3_ML_PolynomialRegression: Polynomial regression with the selection of regularization parameter lambda.

4_ML_GaussianProcessRegression: Gaussian process regression with the selection of kernel parameter sigma.

5_ML_RegressionInheritance_V1: Combining the major nonliner regressors (polynomial regression and Gaussian process regression) into a single class of inheritance. Added features include inheritance, polymorphism in main.cpp (main, cross_validation), the rule of five (copy constructor, copy assignment operator, move constructor, move assignment operator, destructor), dynamic memory allocation, etc.

6_ML_RegressionInheritance_V2: Combining the major nonliner regressors (polynomial regression and Gaussian process regression) into a single class of inheritance. Only keep the necessary features, including inheritance, polymorphism in main.cpp (main, cross_validation), etc.

7_ML_Regression: Combining the major nonliner regressors (polynomial regression and Gaussian process regression) into a single class of inheritance. Introduce a class "Dataset" to handle the dataset and the k-fold cross validation.

**Requirements:**

To run the code, you need to install the C++ library **Eigen**:
http://eigen.tuxfamily.org

Then execute:
```
$ make
$ ./myprogram
```
