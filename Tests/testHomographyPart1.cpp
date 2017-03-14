#include <iostream>
#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include "homography.hpp"
#include "preprocessing.hpp"


// Testing the functions from header file homography.hpp


//Testing the Jacobian implementation 


int main()
{
	std::cout << "Testing Normalize function!" << std::endl;
	Eigen::MatrixXd X(4,4);
	X << 1, 10, 4, 7,
	     2, 8, 7, -15,
	     3, 6 , 6, 17,
	     9, 5, 13, 20;
	     
	
	std::cout << "Printing out the matrix X" << std::endl;
	std::cout << X << "\n\n";
	
	Eigen::MatrixXd Xnorm(4,4);
	Eigen::Matrix<double, 3, 3> T;
	Eigen::Matrix<double, 3, 3> Tprime;
	
	
	
	Normalize(X, Xnorm, T, Tprime);
	
	
	
	std::cout << "Printing out the matrix Xnorm" << std::endl;
	std::cout << Xnorm << "\n\n";
	
	std::cout << "Printing out the similarity T" << std::endl;
	std::cout << T << "\n\n" ;
	
	std::cout << "Printing out the similarity Tprime" << std::endl;
	std::cout << Tprime << "\n\n";
	
		
	
	std::cout << "Testing Algebraic error implementation" << std::endl;
	Eigen::Matrix<double, 4, 1> x;
	x << 4,-1, 2, 1;
	Eigen::Matrix<double, 9, 1> h;
	h << 10, 2, 3, 7, 8, 2, 12, 65, 1;
	
	std::cout << "x" << std::endl;
	std::cout << x << "\n\n";
	
	std::cout << "h" << std::endl;
	std::cout << h << "\n\n";
	
	
	
	
	std::cout << "Algebraic error " << std:: endl;
	std::cout <<  AlgebraicError(h, x)<< "\n\n";
                                          
	Eigen::Matrix<double,2,1> Cost = AlgebraicError(h,x);
	
	Eigen::Matrix<double,2,4> J = AlgebraicErrorJacobian(h,x);
	
	Eigen::Matrix<double,2,4> Jnum;
	
	double epsilon = 10e-6;
	
	Eigen::Matrix<double,4,1> xleft;
	Eigen::Matrix<double,4,1> xright;
	
	
	for (int j = 0; j < 4; j++)
	{
		xleft= x;
		xright = x;
		xleft(j) = xleft(j) - epsilon;
		xright(j) = xright(j) + epsilon;
		
		Jnum.col(j)  = (AlgebraicError(h,xright) - AlgebraicError(h, xleft))/(2*epsilon);
		
    }
    
    double total_error = 0;
	for (int j = 0; j < 4; j++)
	{
		for ( int i = 0; i < 2; i ++)
		{
			total_error += fabs(J(i,j)- Jnum(i,j));
	
	     }
	}
	
	std::cout << "The numerical Jacobian Jnum " << std::endl;
	std::cout << Jnum << "\n\n";
	
	std::cout << "The algebraic Jacobian J " << std::endl;
	std::cout << J << "\n\n";
	
	std::cout << "The total error: " << total_error << std::endl;
	
	std::cout << "The Samspon error: " << std::endl;
	std::cout << HSampsonError(h, x) << "\n\n";
	
	std::cout << "The Jacobian of the Sampson error: " << std::endl;
	
	Eigen::Matrix<double, 4, 9> Grad  = HSampsonErrorJacobian(h, x);
    
    std::cout << Grad << "\n\n"; 
    
    
    Eigen::Matrix<double,4,9> Gradnum;
	
	epsilon = 10e-10;
	
	Eigen::Matrix<double,9,1> hleft;
	Eigen::Matrix<double,9,1> hright;
	
	
	for (int j = 0; j < 9; j++)
	{
		hleft= h;
		hright = h;
		hleft(j) = hleft(j) - epsilon;
		hright(j) = hright(j) + epsilon;
		
		Gradnum.col(j)  = (HSampsonError(hright,x) -HSampsonError(hleft, x))/(2*epsilon);
		
    }
    
    total_error = 0;
    
    for (int j = 0; j < 9; j++)
	{
		for ( int i = 0; i < 4; i ++)
		{
			total_error += fabs(Grad(i,j)- Gradnum(i,j));
	
	     }
	}
	
	std::cout << "The total error between the Jacobian of the Sampson error and its approximation: " << total_error << "\n\n";
	
	std::cout << "The numerical Jacobian of the Sampson error"  << std::endl;
	
	std::cout << Gradnum << "\n\n"; 
	
	
	std::cout << "Testing the HApproximateMSResidualError function: " << std::endl;
	std::cout << HApproximateMSResidualError(h,X) << "\n\n";
	
	std::cout << "Testing the HTotalSampsonErrorJacobian: " << std::endl;
	std::cout << HTotalSampsonErrorJacobian(h,X) << "\n\n";
	
	Eigen::MatrixXd totalError  = HTotalSampsonError(h,X);
	double a = (totalError.transpose()*totalError)(0)/(4*X.rows());
	
	std::cout << "Testing the total cost function again: " << std::endl;
	std::cout << a << "\n\n";
	std::cout << totalError << "\n\n";
	
	
	
	Eigen::MatrixXd X2 = GetFromFile("homography_noisy.txt");
	std::cout << X2.rows() << std::endl;

    std::cout << "\n\n";
    
    Eigen::MatrixXd X2norm(X2.rows(),4);
	Eigen::Matrix<double, 3, 3> T2;
	Eigen::Matrix<double, 3, 3> Tprime2;
	
	Normalize(X2, X2norm, T2, Tprime2);
    	
	Eigen::Matrix<double,3,3> H2norm = DLT(X2norm);
	
	Eigen::Matrix<double,3,3> H2 = NormalizedDLT(X2);
	
	Eigen::Matrix<double,3,3> H3 = (Tprime2.inverse())* H2norm* T2;
	
	std::cout << "Testing DLT and NormalizedDLT" << std::endl;
	
	std::cout << H2/H2(2,2) << "\n\n";
	
	std::cout << H3/H3(2,2) << "\n\n";
	
			
	int numIterations = 30;
	
	std::cout << "Running the HMinimizeApproximateMSResidualError" << std::endl;
	
	Eigen::Matrix <double,3,3> H4 = HMinimizeApproximateMSResidualError(H2norm, X2norm, numIterations);
	
	Eigen::Matrix<double,3,3> H5 = (Tprime2.inverse())* H4* T2;
	
	H5 = H5/H5(2,2);
	
	std::cout << "The answer: " << std::endl;
	std::cout << H5 << "\n\n";
	
	std::cout << "The approximate residual error for the unormalized data: " << std::endl;
	Eigen::Map<Eigen::MatrixXd> h5(H5.transpose().data(), 9,1);
	
	double MSResidualError  = HApproximateMSResidualError(h5,X2);
	
	std::cout << MSResidualError << "\n\n";
	
	double sigma = EstimatedNoiseStandardDeviation(MSResidualError, X2.rows());
	
    std::cout << "The estimated standard deviation:" << sigma << "\n\n";
	
	return 0;
}
		
