/*Implementation of algorithms 
 * _for computing approximate derivatives
 * _ for optimization
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <Eigen/Dense>
#include "preprocessing.hpp"
#include <math.h>
#include <algorithm>

//Function prototypes
Eigen::MatrixXd LeastSquareGivenConstraint(Eigen::MatrixXd A, Eigen::MatrixXd G, int r);



//**********************************************************************
// Function implementations

Eigen::MatrixXd LeastSquareGivenConstraint(Eigen::MatrixXd A, Eigen::MatrixXd G, int r)
{
	//Function finds a vector x that minimizes ||Ax|| subject to the conditions
	//  || x || = 1 and x = Gu for some vector u
	// The matrix G is assumed to have rank r
	
	//Compute singular value decomposition of the matrix G
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(G, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd Uprime = U.leftCols(r);
    
    //Minimizing the function in the space spanned by the column space of Uprime
    Eigen::MatrixXd Aprime = A*Uprime;
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(Aprime, Eigen::ComputeFullV);
    Eigen::MatrixXd V2 = svd2.matrixV();
           
    Eigen::MatrixXd xprime = V2.rightCols(1);
    
       
    //Finding a solution in terms of the original coordinates
    
    return Uprime*xprime;
}
    
    
		
    
	
	



