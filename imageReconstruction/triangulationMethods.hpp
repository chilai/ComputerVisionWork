/*Implementation of triangulation methods */

#ifndef FMATRIX
#define FMATRIX

#include <iostream>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <Eigen/Dense>
#include "preprocessing.hpp"

// Function prototypes


Eigen::Matrix<double,3,1> DirectLinearTriangulation(Eigen::Matrix<double,4,1> X, Eigen::Matrix<double,3,4> & P, 
                                                    Eigen::Matrix<double,3,4> & Pprime);




#endif
