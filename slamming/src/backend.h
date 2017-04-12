// Backend for SLAM problems.
#ifndef BACKEND_H
#define BACKEND_H

#include "ceres/ceres.h"
#include "landmark_3d_error_term.h"
#include "types.h"

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
#endif // BACKEND_H
