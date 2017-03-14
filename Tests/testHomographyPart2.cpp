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
#include "homography.hpp"
#include "preprocessing.hpp"
#include "fundamentalMatrix.hpp"
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// Function takes as input the path to two images and computes a homography 
// matrix.  

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
		std:: cout << "Usage: Image_1_location Image_2_location"<<std::endl;
		return -1;
    }
    
    //Read images
    cv::Mat image1;
    image1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    
    if (!image1.data)
    {
		std::cout << "Could not read image1" << std::endl;
	}
    
    cv::Mat image2;
    image2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    
    if (!image2.data)
    {
		std::cout << "Could not read image2" << std::endl;
	}
	
	
	// Creating a keypoint vector container for each image
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	
	    
    //Create a SURF object detector with a smart pointer
    int minHessian = 4000;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
	
	//Arrays to store descriptors
    cv::Mat descriptor1;
	cv::Mat descriptor2;
	
	//  Get features and descriptors
	if (image1.cols > 0 && image1.rows > 0)
	{
		keypoints1.clear();
		detector->detect(image1, keypoints1);
		detector->compute(image1, keypoints1, descriptor1);
	}
	
	if (image2.cols > 0 && image2.rows > 0)
	{
		keypoints2.clear();
		detector->detect(image2, keypoints2);
		detector->compute(image2, keypoints2, descriptor2);
	}
	
	// Draw keypoints
    cv::Mat image_keypoints1;
    cv::Mat image_keypoints2;
    
    cv::drawKeypoints( image1, keypoints1, image_keypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    cv::drawKeypoints( image2, keypoints2, image_keypoints2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    
    
    cv::imshow("Keypoints 1", image_keypoints1 );
    cv::imshow("Keypoints 2", image_keypoints2 );

    
    
	if(descriptor1.type() != CV_32F){
		
		descriptor1.convertTo(descriptor1, CV_32F);
	}
	
	if(descriptor2.type() != CV_32F){
		
		descriptor2.convertTo(descriptor2, CV_32F);
	}
      
	
	// Matching the interest points
 	std::vector<cv::DMatch> matches2;
	std::vector<cv::DMatch> matches;

	cv::FlannBasedMatcher matcher;
	matcher.match(descriptor1, descriptor2, matches);
	matcher.match(descriptor2, descriptor1, matches2);

	symmetricMatches(matches2,matches);
	
	
	//Draw matches
	cv::Mat img_matches;
	cv::drawMatches( image1, keypoints1, image2, keypoints2, matches, img_matches);
	
	//Show detected matches
	cv::namedWindow("Matches",CV_WINDOW_NORMAL);
	cv::resizeWindow("Matches", 600,600);
	cv::imshow("Matches", img_matches);
	
	//Generate matrix of correspondences
	Eigen::MatrixXd X = GenCorresFromOpenCV(keypoints1, keypoints2, matches);
	
	std::cout << "The number of correspondences." << std::endl;
	std::cout << X.rows()  << std::endl;
	
	double p = 0.99;
	
	double sigma = 1.0;
	
	int numIterations  = 50;
	
	Eigen::Matrix<double,3,3> H = RobustRANSACSampson(X, numIterations, sigma);
	
	Eigen::ComplexEigenSolver<Eigen::Matrix3d> eigensolver2(H);
	if (eigensolver2.info() != Eigen::Success) abort();
	cout << "The eigenvalues of H are:\n" << eigensolver2.eigenvalues() << endl;
	cout << "determinant is: "<< H.determinant() << endl;
	
	std::cout << H << std::endl;
	
	return 0;
}
	
