// Backend.cpp implements a simple backend for 3D SLAM.

#include "landmark_3d_error_term.h"
#include "glog/logging.h"
#include "backend.h"
#include "types.h"
#include "read_g20.h"

using namespace ceres::examples;
using namespace ceres;

bool Backend::Solve(Eigen::Quaterniond& q_ab, Eigen::Vector3d& p_ab){

//Build Problem
BuildOptimizationProblem(*_constraints, _poses, _problem, q_ab, p_ab); 

//Solve Problem
return SolveOptimizationProblem(_problem);


}


void Backend::BuildOptimizationProblem(const ceres::examples::VectorOfConstraints& constraints, ceres::examples::MapOfPoses* poses, ceres::Problem* problem,
										Eigen::Quaterniond& q_ab, Eigen::Vector3d& p_ab){

	CHECK(poses != NULL);
	CHECK(problem != NULL);

	if (constraints.empty()) {
		LOG(INFO) << "No constraints, no problem to optimize.";
	return;
	}


	ceres::LossFunction* loss_function = new ceres::HuberLoss(0.5);
	ceres::LocalParameterization* quaternion_local_parameterization = new EigenQuaternionParameterization;

	
	for (ceres::examples::VectorOfConstraints::const_iterator constraints_iter = constraints.begin(); constraints_iter != constraints.end(); ++constraints_iter ){

		const ceres::examples::Constraint3d& constraint = *constraints_iter;

		MapOfPoses::iterator pose_begin_iter = poses->find(constraint.id_begin);
		CHECK(pose_begin_iter != poses->end()) << "Pose with ID: " << constraint.id_begin << " not found.";
		MapOfPoses::iterator pose_end_iter = poses->find(constraint.id_end);
		CHECK(pose_begin_iter != poses->end()) << "Pose with ID: " << constraint.id_end << " not found.";

		const Eigen::Matrix<double, 6,6> sqrt_information = 
			constraint.information.llt().matrixL();
		// Ceres will take ownership of the pointer
		
		ceres::CostFunction* cost_function = 
			Landmark3dErrorTerm::Create(pose_begin_iter->second,pose_end_iter->second,constraint.t_be,sqrt_information);

		q_ab = constraint.t_be.q;
		p_ab = constraint.t_be.p;
		
		problem->AddResidualBlock(cost_function, loss_function,
      							p_ab.data(),
                          		q_ab.coeffs().data());

	}

	problem->SetParameterization(q_ab.coeffs().data(),
                             	quaternion_local_parameterization);

	// The pose graph optimization problem has six DOFs that are not fully
	// constrained. This is typically referred to as gauge freedom. You can apply
	// a rigid body transformation to all the nodes and the optimization problem
	// will still have the exact same cost. The Levenberg-Marquardt algorithm has
	// internal damping which mitigates this issue, but it is better to properly
	// constrain the gauge freedom. This can be done by setting one of the poses
	// as constant so the optimizer cannot change it.
	//MapOfPoses::iterator pose_start_iter = poses->begin();
	//CHECK(pose_start_iter != poses->end()) << "There are no poses.";
	//problem->SetParameterBlockConstant(pose_start_iter->second.p.data());
	//problem->SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());


}

bool Backend::SolveOptimizationProblem(ceres::Problem* problem){
	CHECK(problem != NULL);
	
	ceres::Solver::Options options;
	options.max_num_iterations = 200;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

	ceres::Solver::Summary summary;
	ceres::Solve(options, problem, &summary);

	std::cout << summary.FullReport() << '\n';
	
	return summary.IsSolutionUsable();

}

// Reads a file in the g2o filename format that describes a pose graph
// problem. The g2o format consists of two entries, vertices and constraints.
bool Backend::ReadG2oFile(const std::string& filename){

	return ceres::examples::ReadG2oFile(filename, _poses, _constraints);

}
 
// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool Backend::OutputPoses(const std::string& filename){

	std::fstream outfile;
	outfile.open(filename.c_str(), std::istream::out);
	if (!outfile) {
		LOG(ERROR) << "Error opening the file: " << filename;
		return false;
	}
	for (std::map<int, ceres::examples::Pose3d, std::less<int>,
		Eigen::aligned_allocator<std::pair<const int, ceres::examples::Pose3d> > >::const_iterator poses_iter = _poses->begin();
   		poses_iter != _poses->end(); ++poses_iter) {
    		const std::map<int, ceres::examples::Pose3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, ceres::examples::Pose3d> > >::value_type& pair = *poses_iter;
			outfile << pair.first << " " << pair.second.p.transpose() << " "
        	<< pair.second.q.x() << " " << pair.second.q.y() << " "
        	<< pair.second.q.z() << " " << pair.second.q.w() << '\n';
	}
	return true;

}

