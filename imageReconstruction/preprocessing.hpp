/*Implementation of preprocessing algorithms for retrieving data and 
 *  normalizing data */

#ifndef PREPROCESSING
#define PREPROCESSING

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stdio.h>
#include "opencv2/flann/flann.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <cstdlib>
#include <Eigen/Dense>
#include <string>
#include <fstream>


// Function prototypes



                            
Eigen::MatrixXd GenCorresFromOpenCV(std::vector<cv::KeyPoint> kPoints1, 
                                    std::vector<cv::KeyPoint> kPoints2,
                                     std::vector<cv::DMatch> matches);  
                                     
Eigen::MatrixXd GetFromFile(std::string fileName);

void Normalize(Eigen::MatrixXd & v);



//**********************************************************************
Eigen::MatrixXd GenCorresFromOpenCV(std::vector<cv::KeyPoint> kPoints1, 
                                    std::vector<cv::KeyPoint> kPoints2,
                                     std::vector<cv::DMatch> matches)
{
	//Function generates a matrix of corrspondences from OpenCV data
    // Input:
    //      _ kPoints1, kPoints2: vectors of openCV keypoints from the first and second image
    //     _ matches: vector of openCV matches generated from these keypoints
    // Output:
    //        X:  (N x 4) matrix containing the correspondences
                              
	int s  = matches.size(); //number of descriptors
	Eigen::MatrixXd X(s, 4); 
	
	// Accessing each descriptor
	for (int i = 0; i < s; i++)
	{
		// Access ith match
		cv::KeyPoint kp2 = kPoints1[matches[i].queryIdx];
		X(i,2) = kp2.pt.x;
		X(i,3) = kp2.pt.y;
		
		cv::KeyPoint kp1 = kPoints2[matches[i].trainIdx];
		X(i,0) = kp1.pt.x;
		X(i,1) = kp1.pt.y;
	}
    return X;
}


Eigen::MatrixXd GetFromFile(std::string fileName)
{
	std::ifstream myfile;
	
	myfile.open(fileName.c_str(), std::ios::in);
	
	//First line of data should specify the size.
	int N;
	myfile >> N;
	
	
	Eigen::MatrixXd X(N,4);
	for (int i = 0; i < N; i++)
	{
		myfile >> X(i,0); 
		myfile >> X(i,1);
		myfile >> X(i,2);
		myfile >> X(i,3);
		
	}
	
	myfile.close();
	
	return X;
}
	 
void Normalize(Eigen::MatrixXd X, Eigen::MatrixXd & Xnorm, Eigen::Matrix<double,3,3> & T, Eigen::Matrix<double, 3,3> & Tprime)
{
	//function that normalizes the columns of a matrix of correspondences
    // Input:
    //       _ X: (N x 4) matrix containing the correspondences
    //       _ Xnorm: (N x 4) matrix to store the correspondences
    //       _ T: (3x3) matrix to represent the generated similarity transformation for first image
    //       _ Tprime: (3x3) matrix to represent the similarity transformation for the second image

	
	if ((X.cols() != 4) ||(Xnorm.cols() != 4) || (X.rows() != Xnorm.rows()))
	{
		std::cout << "Check dimensions!" << std::endl;
		return;
	}
	T = Eigen::MatrixXd::Identity(3, 3);
	Tprime = Eigen::MatrixXd::Identity(3, 3); 
			
	double mean;
	double variance = 0;
	double variance_p = 0;
	for (int j = 0; j < 4; j++)
	{    
		mean = X.col(j).mean();
	    Xnorm.col(j) =  (X.col(j) - Eigen::MatrixXd::Constant(X.rows(),1,mean));  // centering 
	   
	    if (j < 2)
	    {
	       T(j,2) = -mean;
	       variance += Xnorm.col(j).dot(Xnorm.col(j))/X.rows();
	    }
	    else
	    {
		   Tprime(j-2, 2) = -mean;
		   variance_p += Xnorm.col(j).dot(Xnorm.col(j))/X.rows();
		}
	 }
	 
	 double scale = sqrt(variance/2.0);
	 
	 	 
	 double scale_p = sqrt(variance_p/2.0);
	 
	 
	 Xnorm.col(0) = Xnorm.col(0)/scale;
	 Xnorm.col(1) = Xnorm.col(1)/scale;
	 Xnorm.col(2) = Xnorm.col(2)/scale_p;
	 Xnorm.col(3) = Xnorm.col(3)/scale_p;
	 
	 T = T/scale;
	 Tprime = Tprime/scale_p;
	 T(2,2) = 1;
	 Tprime(2,2) = 1;
	 
	return; 
}

//From DFK and Erik Nelson with some edits by Dapo
void symmetricMatches(const std::vector<cv::DMatch>& feature_matches_lhs,std::vector<cv::DMatch>& feature_matches_rhs)
{
	std::map<int, int> feature_indices;
	//feature_indices.reserve(feature_matches_lhs.size());

  	// Add all LHS matches to the map.
  	for (int i = 0; i < feature_matches_lhs.size(); i++) {

		const cv::DMatch& feature_match = feature_matches_lhs.at(i);
		feature_indices.insert(std::make_pair(feature_match.queryIdx,
                                          feature_match.trainIdx));
  	}

  	// For each match in the RHS set, search for the same match in the LHS set.
  	// If the match is not symmetric, remove it from the RHS set.
  	std::vector<cv::DMatch>::iterator rhs_iter = feature_matches_rhs.begin();
  	while (rhs_iter != feature_matches_rhs.end()) {
    	const std::map<int,int>::iterator& lhs_matched_iter = feature_indices.find(rhs_iter->trainIdx);

    	// If a symmetric match is found, keep it in the RHS set.
    	if (lhs_matched_iter != feature_indices.end()) {
      		if (lhs_matched_iter->second == rhs_iter->queryIdx) {
       	 		++rhs_iter;
       			 continue;
  			}
    	}

    	// Remove the non-symmetric match and continue on.
    	feature_matches_rhs.erase(rhs_iter);
  	}
}

	
	 

#endif
