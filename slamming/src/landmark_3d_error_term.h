// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: vitus@google.com (Michael Vitus)

#ifndef EXAMPLES_CERES_LANDMARK_3D_ERROR_TERM_H_
#define EXAMPLES_CERES_LANDMARK_3D_ERROR_TERM_H_

#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"

#include "types.h"

namespace ceres {
namespace examples {

// Computes the error term for two measuremnts of the same landmark from different
// coordinate frames. Let the hat variables be the measurement. We have two coordinates x_a_hat
// and x_b_hat. Through sensor measurements we can measure the transformation of
// frame B w.r.t frame A denoted as t_ab_hat. We can compute an error metric
// between the current estimate of the transformation and the measurement.
//
// In this formulation, we have chosen to represent the rigid transformation as
// a Hamiltonian quaternion, q, and position, p. The quaternion ordering is
// [x, y, z, w].

// The estimated measurement is:
//      t_ab = [ p_ab ]  
//             [ q_ab ]    
//
// Now we can compute an error metric between the estimated and
// measurement transformation. For the orientation error, we will use the
// standard multiplicative error resulting in:
//
//   error = [ x_a_hat - (R(q_ab)*x_b_hat + p_ab)]                 ]
//           [ 2.0 * Vec(q_ab * \hat{q}_ab^{-1}) ]
//
// where ^{-1} denotes the inverse and R(q) is the rotation matrix for the
// quaternion
// where Vec(*) returns the vector (imaginary) part of the quaternion. Since
// the measurement has an uncertainty associated with how accurate it is, we
// will weight the errors by the square root of the measurement information
// matrix:
//
//   residuals = I^{1/2) * error
// where I is the information matrix which is the inverse of the covariance.
class Landmark3dErrorTerm {
 public:
  Landmark3dErrorTerm(const Pose3d& x_a_hat, const Pose3d& x_b_hat,
					   					const Pose3d& t_ab_measured,
                     	const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information),x_a_hat_(x_a_hat), x_b_hat_(x_b_hat) {}

  template <typename T>
  bool operator()(const T* const p_ab_ptr, const T* const q_ab_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_ab(p_ab_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_ab(q_ab_ptr);

    // Compute the displacement between the two frames in the A frame.
    // Eigen::Quaternion<T> q_ab_inverse = q_ab.conjugate();

    Eigen::Matrix<T, 3, 1> x_b_in_a = q_ab*x_b_hat_.p.template cast<T>() + p_ab;

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        t_ab_measured_.q.template cast<T>() * q_ab.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        x_a_hat_.p.template cast<T>() - x_b_in_a;
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(
			const Pose3d& x_a_hat,
			const Pose3d& x_b_hat,
      const Pose3d& t_ab_measured,
      const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<Landmark3dErrorTerm, 6, 3, 4>(
        new Landmark3dErrorTerm(x_a_hat,x_b_hat,t_ab_measured, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
	// The measurement for the landmark in frame A.
  const Pose3d x_a_hat_;
	// The measurement for the landmark in frame B.
  const Pose3d x_b_hat_;
  // The measurement for change of basis from B to A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

}  // namespace examples
}  // namespace ceres

#endif  // EXAMPLES_CERES_LANDMARK_3D_ERROR_TERM_H_
