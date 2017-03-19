/*Implementation of algorithms 
 * _for computing approximate derivatives
 * _ for optimization
 */

#ifndef OPT_APP
#define OPT_APP


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

Eigen::MatrixXcd  FindRootsUsingCompanionMatrix(Eigen::MatrixXd coefficients);



//**********************************************************************
// Function implementations

Eigen::MatrixXd LeastSquareGivenConstraint(Eigen::MatrixXd A, Eigen::MatrixXd G, int r)
{
	//Function finds a vector x that minimizes ||Ax|| subject to the conditions
	//  || x || = 1 and x = Gu for some vector u
	// The matrix G is assumed to have rank r
	
	//Compute singular value decomposition of the matrix G
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(G, Eigen::ComputeFullU | Eigen::ComputeFullV);
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

Eigen::MatrixXcd FindRootsUsingCompanionMatrix(Eigen::MatrixXd coefficients)
{
	// Function computes the roots of a polynomial by finding the 
	// eigenvalues of the companion matrix
	
	const int & n = coefficients.rows();
		
	Eigen::MatrixXd  A = Eigen::MatrixXd::Constant(n,n,0);
	A.bottomLeftCorner(n-1,n-1) = Eigen::MatrixXd::Identity(n-1,n-1);
	A.row(0) = -coefficients.transpose();
	
	Eigen::ComplexEigenSolver<Eigen::MatrixXd> eigensolver(A);
	if (eigensolver.info() != Eigen::Success) abort();
	
	Eigen::MatrixXcd roots = eigensolver.eigenvalues();
	
	
	return roots;
	
		
}
	
	
#endif	



