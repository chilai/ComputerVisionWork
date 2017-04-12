#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include "types2.h"
#include "ceres/ceres.h"

namespace ceres {
namespace examples {

	typedef std::map<int, Pose3d, std::less<int>,
		             Eigen::aligned_allocator<std::pair<const int, Pose3d> > >
		MapOfPoses;

	typedef std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d> >
		VectorOfConstraints;
	}
}


class Backend{

public:
	// Map to store poses (see types.h)
	ceres::examples::MapOfPoses* _poses;
	// Vector to store constraints (see types.h)
	ceres::examples::VectorOfConstraints* _constraints;
	// Default constructor
	Backend(ceres::examples::MapOfPoses* poses,
		ceres::examples::VectorOfConstraints* constraints, ceres::Problem* problem ):_poses(poses),
		_constraints(constraints),_problem(problem){};
	// solves the problem, given current poses and constraints
	bool Solve(Eigen::Quaterniond& q_ab, Eigen::Vector3d& p_ab);
	// Reads a file in the g2o filename format that describes a pose graph
	// problem. The g2o format consists of two entries, vertices and constraints.
	bool ReadG2oFile(const std::string& filename); 
	// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
	bool OutputPoses(const std::string& filename);
	
private:
	// Creates a problem from the poses and constraints
	void BuildOptimizationProblem(const ceres::examples::VectorOfConstraints& constraints, ceres::examples::MapOfPoses* poses, ceres::Problem* problem,
									Eigen::Quaterniond& q_ab, Eigen::Vector3d& p_ab);
	// Solves the problem 
	bool SolveOptimizationProblem(ceres::Problem* problem);
	// Problem object 
	ceres::Problem* _problem;

};



//Checks transformed points against threshold
static bool checkThreshold(const pcl::PointXYZ& p1, pcl::PointXYZ& p2, float inlier_threshold){
	return (100*100*((p1.x-p2.x)*(p1.x-p2.x)+
			(p1.y-p2.y)*(p1.y-p2.y)+
			(p1.z-p2.z)*(p1.z-p2.z))<
			(inlier_threshold*inlier_threshold));
}

Eigen::MatrixXd GetFromFile(std::string fileName)
{
	std::ifstream myfile;
	
	myfile.open(fileName.c_str(), std::ios::in);
	
	//First line of data should specify the size.
	int N;
	myfile >> N;
	
	
	Eigen::MatrixXd X(N,6);
	for (int i = 0; i < N; i++)
	{
		myfile >> X(i,0); 
		myfile >> X(i,1);
		myfile >> X(i,2);
		myfile >> X(i,3);
		myfile >> X(i,4);
		myfile >> X(i,5);
		
	}
	
	myfile.close();
	
	return X;
}

pcl::PointCloud<pcl::PointXYZ>* Eigen2pcl(Eigen::MatrixXd points){

	pcl::PointCloud<pcl::PointXYZ>* cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>() );
	cloud_ptr->is_dense = false;

	// For each row in matrix create a point and store in pointcloud
	
	int r = points.rows();
	int c = points.cols();

	if(c == 3){
		for(int i =0; i < r; i++){

			pcl::PointXYZ point;
			point.x = points(i,0);
			point.y = points(i,1);
			point.z = points(i,2);

			cloud_ptr->points.push_back(point);
		

		}
	}


	return cloud_ptr;

}

bool Eigen2ceres(Eigen::MatrixXd points1, Eigen::MatrixXd points2,ceres::examples::MapOfPoses* pose_map,ceres::examples::VectorOfConstraints* constraints_list, Eigen::Matrix3d rotation_guess){


	// for each pair of correspoinding points add a pose and constraint
	
	int r1 = points1.rows();
	int c1 = points1.cols();

	int c2 = points2.cols();

	if(c1 == 3 && c2 == 3){

		pose_map->clear();
		constraints_list->clear();

		int pose_idx = 0;
		for(int i =0; i < r1; i++){


					
					Eigen::Vector3d ceres_x1 = Eigen::Vector3d (points1(i,0),points1(i,1),points1(i,2));
					Eigen::Vector3d ceres_x2 = Eigen::Vector3d (points2(i,0),points2(i,1),points2(i,2));
					Eigen::Matrix3d ceres_R  = rotation_guess;
					Eigen::Vector3d ceres_t = ceres_x1 - ceres_R*ceres_x2;
					

					ceres::examples::Pose3d pose_xb;
					pose_xb.p = ceres_x1;
					pose_xb.q = Eigen::Quaterniond(0.0,0.0,0.0,1.0);


					ceres::examples::Pose3d pose_xe;
					pose_xe.p = ceres_x2;
					pose_xe.q = Eigen::Quaterniond(0.0,0.0,0.0,1.0);

			
					
					(*pose_map)[pose_idx] = pose_xb;
					(*pose_map)[pose_idx + 1] = pose_xe;
					

					ceres::examples::Constraint3d constraint_be;
					constraint_be.id_begin = pose_idx;
					constraint_be.id_end = pose_idx + 1;
					
					ceres::examples::Pose3d pose_be;
					pose_be.p = ceres_t;
					pose_be.q = Eigen::Quaterniond(ceres_R);

					constraint_be.t_be = pose_be;
					Eigen::Matrix<double,6,6> information = Eigen::Matrix<double, 6, 6>::Zero();
					Eigen::Matrix3d position_information = 1.0*Eigen::Matrix3d::Identity();
					Eigen::Matrix3d orientation_information = 1.0*Eigen::Matrix3d::Identity();

					information.block<3,3>(0,0) = position_information;
					information.block<3,3>(3,3) = orientation_information;

					constraint_be.information = information;

					(*constraints_list).push_back(constraint_be);

					pose_idx += 2;
		

		}

		return true;
	}

	return false;





}

Eigen::Matrix<double,4,4> EstimateMotionICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_ptr, bool* converged, double* fitness_score){


	// returns rigid body transformation that takes robot from view at cloud1 to view at cloud2.
	// same as the change of basis from frame of reference of cloud2 to that of cloud1
  	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud2_ptr);
  	icp.setInputTarget(cloud1_ptr);
	pcl::PointCloud<pcl::PointXYZ> Final;
 	icp.align(Final);

	*converged = icp.hasConverged();
	*fitness_score = icp.getFitnessScore();




	Eigen::Matrix<double,4,4> trafo = icp.getFinalTransformation().cast<double>();
	return  trafo;

}

Eigen::Matrix<double,4,4> EstimateMotion3Dto3D(Eigen::MatrixXd Xprev, Eigen::MatrixXd Xnew)
{
	//Set of points has shape 3 x N
	// where N is the number of points
	
	//Computing the mean vector
	Eigen::Matrix<double,3,1>  Xprev_mean;
	Eigen::Matrix<double,3,1>  Xnew_mean;
	
	for (int i = 0; i < 3; i++)
	{
		Xprev_mean(i) = Xprev.row(i).mean();
		Xnew_mean(i)  = Xnew.row(i).mean();
	}
	
    int s = Xprev.cols();
    
    Eigen::MatrixXd Xprev_norm(3,s);
    Eigen::MatrixXd Xnew_norm(3,s);
    
    for (int j = 0; j < s; j++)
    {
		Xprev_norm.col(j) = Xprev.col(j) - Xprev_mean;
		Xnew_norm.col(j) = Xnew.col(j) - Xnew_mean;
	}
    
	Eigen::Matrix<double,3,3> A = Xprev_norm* Xnew_norm.transpose();
	
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    
    Eigen::Matrix<double,3,3> R = V*U.transpose();
    
    Eigen::Matrix<double,3,1> t = Xnew_mean - R*Xprev_mean;
    
    Eigen::Matrix<double,4,4> T = Eigen::MatrixXd::Constant(4,4,1);
	T.block<1,3>(3,0) = Eigen::Vector3d(0,0,0);
    
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    
    return T.inverse();
 }


void transform_estimation_ransac(
								pcl::PointCloud<pcl::PointXYZ>::Ptr pc1, 
								pcl::PointCloud<pcl::PointXYZ>::Ptr pc2,
								ceres::examples::MapOfPoses& pose_map,
								ceres::examples::VectorOfConstraints& constraints_list,
								int num_samples,
								float inlier_threshold,
								int num_iterations,
								int& num_inliers_out,
								Eigen::Matrix4d& transformation){


Eigen::Matrix4d best_transform = Eigen::Matrix4d::Identity();
int best_inliers = 0;
int list_sz = constraints_list.size();

if(num_samples < list_sz){


	//Run RANSAC the specified number of times 
    for ( int r = 0; r< num_iterations; r++){

		// sampled num_sampled points from constraints

		/* initialize random seed: */
		srand (time(NULL));

		std::set<int> rand_points_set;
		int rand_num;
		do {
			rand_num = rand() % list_sz;
			rand_points_set.insert(rand_num);
		}while(rand_points_set.size() < num_samples);

		//std::cout << "sample idx: " << std::endl;
		//for(std::set<int>::iterator it=rand_points_set.begin(); it!=rand_points_set.end(); ++it)
			//std::cout << *it << " , ";

		//std::cout << " " << std::endl;

		// create new vector of constriants
		ceres::examples::VectorOfConstraints sampled_constraints_list;
		

		for(std::set<int>::iterator it=rand_points_set.begin(); it!=rand_points_set.end(); ++it){

			sampled_constraints_list.push_back(constraints_list[*it]);
		


		}

		// find corresponding poses and create a new pose_map

		ceres::examples::MapOfPoses sampled_pose_map;

		for (ceres::examples::VectorOfConstraints::const_iterator constraints_iter = sampled_constraints_list.begin();
			constraints_iter != sampled_constraints_list.end(); ++constraints_iter) {

			const ceres::examples::Constraint3d& constraint = *constraints_iter;

			sampled_pose_map.insert(*(pose_map.find(constraint.id_begin)));
			sampled_pose_map.insert(*(pose_map.find(constraint.id_end)));

		}
		

		// solve optimization problem

		Eigen::Quaterniond final_q;
		Eigen::Vector3d final_p;

		ceres::Problem problem;
		Backend _backend(&sampled_pose_map, &sampled_constraints_list,&problem);
		_backend.Solve(final_q, final_p);
		
		Eigen::Matrix4d transform = Eigen::MatrixXd::Constant(4,4,1);
		transform.block<1,3>(3,0) = Eigen::Vector3d(0,0,0);
		transform.block<3,1>(0,3) = final_p;
		transform.block<3,3>(0,0) = final_q.toRotationMatrix();

		//std::cout << "q: " << final_q.x() <<", " << final_q.y() << ", " << final_q.z() << " ," << final_q.w() << std::endl;
		//std::cout << "p: " << result << std::endl;



		//apply transform to cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ> ());
		pcl::transformPointCloud(*pc2,*temp_cloud_ptr,transform);

		// get inliers
		int inliers = 0;
		for (int i = 0; i < pc1->size(); ++i)
		{
			if(checkThreshold(pc1->points[i],temp_cloud_ptr->points[i],inlier_threshold)){
				inliers++;
			}
		}

		//store best inliers and best transform
		//save best result
		if(inliers > best_inliers){

			num_inliers_out = inliers;
			best_transform = transform;
		} 


	}

	transformation = best_transform;

}else{

	Eigen::Quaterniond final_q;
	Eigen::Vector3d final_p;

	ceres::Problem problem;
	Backend _backend(&pose_map, &constraints_list,&problem);
	_backend.Solve(final_q, final_p);
	transformation = Eigen::MatrixXd::Constant(4,4,1);
	transformation.block<1,3>(3,0) = Eigen::Vector3d(0,0,0);
	transformation.block<3,1>(0,3) = final_p;
	transformation.block<3,3>(0,0) = final_q.toRotationMatrix();

	//std::cout << "q: " << final_q.x() <<", " << final_q.y() << ", " << final_q.z() << " ," << final_q.w() << std::endl;
	//std::cout << "p: " << result << std::endl;


	}

	//apply transform to cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::transformPointCloud(*pc2,*temp_cloud_ptr,transformation);

	// get inliers
	num_inliers_out = 0;
	for (int i = 0; i < pc1->size(); ++i)
	{
		if(checkThreshold(pc1->points[i],temp_cloud_ptr->points[i],inlier_threshold)){
			num_inliers_out++;
		}
	}

}

int main(int argc, char** argv){


 if(argc != 2)
    {
		std:: cout << "Usage: motionEstimation inputFile"<<std::endl;
		return -1;
    }

// read from input file 
Eigen::MatrixXd Data = GetFromFile(std::string(argv[1]));

// Split Matrix into two
Eigen::MatrixXd X_1_tmp = (Data.leftCols(3));
Eigen::MatrixXd X_2_tmp = (Data.rightCols(3));

Eigen::MatrixXd X_1 = X_1_tmp.transpose();
Eigen::MatrixXd X_2 = X_2_tmp.transpose();

//std::cout << X_1 << std::endl;

pcl::PointCloud<pcl::PointXYZ>* cloud1_ptr = Eigen2pcl(X_1_tmp);
pcl::PointCloud<pcl::PointXYZ>* cloud2_ptr = Eigen2pcl(X_2_tmp);


// Compute Transformation
Eigen::Matrix<double, 4,4> Trafo = EstimateMotion3Dto3D(X_1,X_2);

bool converged = false;
double fitness_score = 1000;
Eigen::Matrix<double, 4,4> Trafo_icp = EstimateMotionICP(cloud1_ptr->makeShared(),cloud2_ptr->makeShared(), &converged, &fitness_score);


ceres::examples::MapOfPoses pose_map;
ceres::examples::VectorOfConstraints constraints_list;


Eigen::Matrix3d R;
R << 0.9996, -0.0300, 0,
	0.0300, 0.9996, 0,
	0.0, 0.0, 1;
Eigen2ceres(X_1_tmp,X_2_tmp,&pose_map,&constraints_list, R);
Eigen::Quaterniond final_q;
Eigen::Vector3d final_p;
ceres::Problem problem;
Backend _backend(&pose_map, &constraints_list,&problem);
_backend.Solve(final_q, final_p);
	
Eigen::Matrix<double, 4,4> Trafo_ceres = Eigen::MatrixXd::Constant(4,4,1);
Trafo_ceres.block<1,3>(3,0) = Eigen::Vector3d(0,0,0);
Trafo_ceres.block<3,1>(0,3) = final_p;
Trafo_ceres.block<3,3>(0,0) = final_q.toRotationMatrix();


int num_samples = 30;
float inlier_threshold = 15;
int num_iterations = 50;
int num_inliers_out = 0;
Eigen::Matrix<double, 4, 4> Trafo_ceres_ransac = Eigen::MatrixXd::Constant(4,4,1);
transform_estimation_ransac(
							cloud1_ptr->makeShared(), 
							cloud2_ptr->makeShared(),
							pose_map,
							constraints_list,
							num_samples,
							inlier_threshold,
							num_iterations,
							num_inliers_out,
							Trafo_ceres_ransac);



// Print Transformation
std::cout << "Transformation Estimation 3D-to-3D: " << std::endl;
std::cout << Trafo << std::endl;

std::cout << "Transformation Estimation ICP: " << std::endl;
std::cout << Trafo_icp << std::endl;
std::cout << "converged?: "<< converged << std::endl;
std::cout << "fitness score: "<< fitness_score << std::endl;


std::cout << "Transformation Estimation ceres: " << std::endl;
std::cout << Trafo_ceres << std::endl;


std::cout << "Transformation Estimation ceres ransac: " << std::endl;
std::cout << Trafo_ceres_ransac << std::endl;
std::cout << "num inliers : " << num_inliers_out << std::endl;



return 0;

}


