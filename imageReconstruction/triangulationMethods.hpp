/*Implementation of triangulation methods */

#ifndef TRIANGULATE
#define TRIANGULATE

#include <iostream>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <Eigen/Dense>
#include "preprocessing.hpp"
#include "fundamentalMatrix.hpp"

// Function prototypes

Eigen::MatrixXd DirectLinearTriangulation(const Eigen::MatrixXd & X, const Eigen::Matrix<double,3,3> & F);
                                                    

Eigen::MatrixXd NormalizedDirectLinearTriangulation(const Eigen::MatrixXd & X, const  Eigen::Matrix<double,3,3> & F);
                                                    
                                                    
Eigen::MatrixXd OptimalTriangulation(const Eigen::MatrixXd & X, const Eigen::Matrix<double,3,3> & F);
                                     

//*********************************************************************//
Eigen::MatrixXd DirectLinearTriangulation(const Eigen::MatrixXd & X, const Eigen::Matrix<double,3,3> & F)                                                
{
	//Function computes  world points for a set of correspondences by using 
	//the direct linear triangulation method
	
	int s  = X.rows();
	
	Eigen::MatrixXd X_world(s,4); // matrix where each row is a reconstructed 
	                                  // world point for a correspondence
	
	//Construct the canonical camera pairs
	Eigen::Matrix<double,3,4> P = Eigen::MatrixXd::Constant(3,4,0);
	Eigen::Matrix<double,3,4> Pprime = Eigen::MatrixXd::Constant(3,4,0);
	
	ComputeCanonicalCameraPairs(F,P,Pprime);
	
	
	Eigen::Matrix<double,4,4> A; 
	for (int i = 0; i < s; i++)
	{
	   A.row(0) = X(i,0)*P.row(2) - P.row(0);
	   A.row(1) = X(i,1)*P.row(2) - P.row(1);
	   A.row(2) = X(i,2)*Pprime.row(2) - Pprime.row(0);
	   A.row(3) = X(i,3)*Pprime.row(2) - Pprime.row(1);
	   
	 
	  
	  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
      X_world.row(i) = svd.matrixV().rightCols(1).transpose();
      
     }
     	
	return X_world;
}

          //***************************************************//

Eigen::MatrixXd NormalizedDirectLinearTriangulation(const Eigen::MatrixXd & X, const Eigen::Matrix<double,3,3> & F) 
{
	Eigen::MatrixXd Xnorm(X.rows(),4);
    Eigen::Matrix <double, 3, 3> T;
    Eigen::Matrix <double, 3, 3> Tprime;
    
    //Normalizing and storing generated similarity transformations
    Normalize(X, Xnorm, T, Tprime);
    
       
    Eigen::MatrixXd X_world = DirectLinearTriangulation(Xnorm, Tprime.inverse().transpose() * F * T.inverse());
    
    return X_world;
    
}
                                                    


          //***************************************************//
        
       
 Eigen::MatrixXd OptimalTriangulation(const Eigen::MatrixXd & X, const Eigen::Matrix<double,3,3> & F)                                 
{
	// Optimal method to compute a 3d world point from a correspondence.
	// This is done by finding epipolar lines that minimize  a square distance
	// See "Multiple view geometry" Chapter 12 section 5
	
	int s = X.rows();
	
	//Transformations used to translate image points to origin
	Eigen::Matrix<double,3,3> T = Eigen::MatrixXd::Identity(3,3);
	Eigen::Matrix<double,3,3> Tprime = Eigen::MatrixXd::Identity(3,3);
	
	//Rotation matrices used to rotate epipolar points on the x-axis
	Eigen::Matrix<double,3,3> R;
	Eigen::Matrix<double,3,3> Rprime;
	R(2,2) = 1;
	Rprime(2,2)= 1;
	
	//Fundamental matrix for the new coordinate systems
	Eigen::Matrix<double,3,3> Fnew;
	
	// Array to store the epipoles
	Eigen::Matrix<double,3,1> * epipoles;
	epipoles  = new Eigen::Matrix<double,3,1> [2];
	
	Eigen::Matrix<double,3,1>  first_epipole;
	Eigen::Matrix<double,3,1>  second_epipole;
	
	
	// variables and vectors needed to compute the sixth order coefficients
	// that provide the optimal solution
	double a,b,c,d,p,q;  
	Eigen::Matrix<double,7,1> v1;
	Eigen::Matrix<double,7,1> v2;
	Eigen::Matrix<double,7,1> v;
	Eigen::Matrix<double,6,1> coefficients;
	
	// Variables to store the cost function 
	double cost;  
	double best_cost;
	
    double t;  // parameter for the epipolar line
    double t_best;	
    
    // Vectors to store homogeneous coordinates of the optimal epipolar lines
    // and homogeneous coordinates of estimated closest points
    Eigen::Matrix<double,3,1> L;
    Eigen::Matrix<double,3,1> Lprime;
    
    Eigen::Matrix<double,3,1> x_estimated;
    Eigen::Matrix<double,3,1> xprime_estimated;
    
    // Matrix to store non-homogeneous coordinates of estimated closest points 
    // on the epipolar lines
    Eigen::MatrixXd X_estimated(s,4);
    

	
	
	for (int i = 0; i < s; i++)
	{
		// Updating translation maps
		T(0,2) = - X(i,0);
		T(1,2) = - X(i,1);
		Tprime(0,2)= -X(i,2);
		Tprime(1,2)= -X(i,3);
		
		// Updating the fundamental matrix for the new coordinate system
		Fnew = Tprime.inverse().transpose() * F * T.inverse();
		
		// Computing the epipoles of the new fundamental matrix
		//Retrieve epipoles
	    ComputeEpipoles(Fnew, epipoles);
	    
	    first_epipole  = epipoles[0];
	    first_epipole = first_epipole/sqrt(pow(first_epipole(0),2) + pow(first_epipole(1),2));  //rescaling
	    
	    second_epipole  = epipoles[1];
	    second_epipole = second_epipole/sqrt(pow(second_epipole(0),2) + pow(second_epipole(1),2));  //rescaling
	    
	    
	    
	    // Update rotation matrices
	    R.block<2,2>(0,0) << first_epipole(0), first_epipole(1),
	                              -first_epipole(1), first_epipole(0);
	                              
	    
	    Rprime.block<2,2>(0,0) << second_epipole(0), second_epipole(1),
	                              -second_epipole(1), second_epipole(0);                         
	    
	    
	    //Computing the fundamental matrix for the new coordinate systems
	    Fnew = Rprime * Fnew  * R.transpose();
	    
	    //Computing the coefficients of the polynomial
	    p = pow(first_epipole(2),2);
	    q = pow(second_epipole(2),2);
	    a =  F(1,1);
	    b = F(1,2);
	    c = F(2,1);
	    d = F(2,2);
	    
	    v1 << 0,
	          pow((pow(a,2) + q*pow(c,2)),2),
	          4*(pow(a,2) + q*pow(c,2))*(a*b + q*c*d),
	          4*(a*b + q*c*d) + 2*(pow(a,2) + q*pow(c,2))*(pow(b,2) + q*pow(d,2)),
	          4*(a*b + q*c*d)*(pow(b,2) + q*pow(d,2)),
	          pow((pow(b,2) + q*pow(d,2)),2),
	          0;
	          
	   v2 << pow(p,2)*a*c,
	         (a*d + b*c)*pow(p,2),
	         pow(p,2)*b*d + 2*p*a*c,
	         2*p*(a*d + b*c),
	         2*p*b*d + a*c,
	         a*d + b*c,
	         b*d;
	         
	  v = v1 -(a*d-b*c)*v2;
	  
	  v = v/v(0);
	  
	  coefficients = v.block<6,1>(1,0);
	  
	  best_cost = 1.0/p + c*c/(a*a + q*c*c);
	  
	  // Computing the roots
	  Eigen::MatrixXcd roots = FindRootsUsingCompanionMatrix(coefficients);
	  
	  // Finding the optimal value for the cost function (reprojection error)
	  for (int k = 0; k < 6; k++)
	  {
		  if (roots(k).imag() < 10e-12)
		  {
			  t = roots(k).real();
			  
			  cost  = pow(t,2)/(1 + p*pow(t,2)) + pow((c*t+d),2)/( pow((a*t+b),2) + q*pow((c*t+d),2) );
			  
			  if (cost < best_cost)
			  {
				  t_best = t;
				  best_cost = cost;
			   }
		  }
	 }
	 
	 // Computing the optimal epipolar lines
	  L << first_epipole(2)*t_best,
	      1,
	      -t_best;
	      
	 Lprime << -second_epipole(2)*(c*t+d),
	            a*t + b,
	            c*t + d;
	 
	 // Computing the estimated points (the points on the epipolar lines 
	 // closest to the noisy points)
	 // The noisy point corresponds to the origin of the coordinate system 
	 x_estimated << -L(0)*L(2)/(L(0)*L(0) + L(1)*L(1)),
	                -L(1)*L(2)/(L(0)*L(0) + L(1)*L(1)),
	                1;
	                
    xprime_estimated << -Lprime(0)*Lprime(2)/(Lprime(0)*Lprime(0) + Lprime(1)*Lprime(1)),
	                    -Lprime(1)*Lprime(2)/(Lprime(0)*Lprime(0) + Lprime(1)*Lprime(1)),
	                    1;
     
    // Transfroming back to original coordinates of the the image planes
    x_estimated  = T.inverse()*R.transpose()*x_estimated;
    xprime_estimated = Tprime.inverse()*Rprime.transpose()*xprime_estimated;
      
   // storing the estimated points 
   X_estimated.row(i) << x_estimated(0),
                        x_estimated(1),
                        xprime_estimated(0),
                        xprime_estimated(1);
                        
                           
   }
	 
	delete [] epipoles;	
		
	// Computing the estimated world points
	
	Eigen::MatrixXd X_world = DirectLinearTriangulation(X_estimated, F);
	
	return X_world;
	
}


    //*************************************************//
	                       
	                    
	            
	            	  
		  
		  
		  	  
				  
			  
	  
	  	   
		
 
         

	



#endif
