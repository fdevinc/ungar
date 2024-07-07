/******************************************************************************
 *
 * @file ungar/example/rbd/robot.example.cpp
 * @author Flavio De Vincenti (flavio.devincenti@inf.ethz.ch)
 *
 * @section LICENSE
 * -----------------------------------------------------------------------
 *
 * Copyright 2023 Flavio De Vincenti
 *
 * -----------------------------------------------------------------------
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS
 * IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 ******************************************************************************/

#include "ungar/rbd/robot.hpp"
#include "ungar/autodiff/function.hpp"
#include "ungar/mvariable_lazy_map.hpp"

#include "ungar/rbd/quantities/centroidal_momentum.hpp"
#include "ungar/rbd/quantities/com_acceleration.hpp"
#include "ungar/rbd/quantities/composite_rigid_body_inertia.hpp"
#include "ungar/rbd/quantities/generalized_accelerations.hpp"

using namespace Ungar;

namespace ANYmalB {
namespace Variables {

/***************                         ANYMAL B                         ***************/
/***************                Define helper m-variables.                ***************/
inline constexpr auto NUM_LEGS = 4_idx;

inline constexpr auto LF = 0_idx;
inline constexpr auto LH = 1_idx;
inline constexpr auto RF = 2_idx;
inline constexpr auto RH = 3_idx;

inline constexpr auto LF_FOOT_FRAME_ID = 12UL;
inline constexpr auto LH_FOOT_FRAME_ID = 22UL;
inline constexpr auto RF_FOOT_FRAME_ID = 32UL;
inline constexpr auto RH_FOOT_FRAME_ID = 42UL;

UNGAR_LEAF_MVARIABLE(position, 3);
UNGAR_LEAF_MVARIABLE(orientation, Q);
UNGAR_BRANCH_MVARIABLE(base_pose, position, orientation);

UNGAR_LEAF_MVARIABLE(hip_aa, 1);
UNGAR_LEAF_MVARIABLE(hip_fe, 1);
UNGAR_LEAF_MVARIABLE(knee_fe, 1);
UNGAR_BRANCH_MVARIABLE(leg_joint_coords, hip_aa, hip_fe, knee_fe);
UNGAR_MVARIABLE_ARRAY(joint_coords, leg_joint_coords, NUM_LEGS);
UNGAR_BRANCH_MVARIABLE(q, base_pose, joint_coords);

UNGAR_LEAF_MVARIABLE(b_linear_velocity, 3);
UNGAR_LEAF_MVARIABLE(b_angular_velocity, 3);
UNGAR_BRANCH_MVARIABLE(base_twist, b_linear_velocity, b_angular_velocity);

UNGAR_BRANCH_MVARIABLE(leg_joint_vels, hip_aa, hip_fe, knee_fe);
UNGAR_MVARIABLE_ARRAY(joint_vels, leg_joint_vels, NUM_LEGS);
UNGAR_BRANCH_MVARIABLE(v, base_twist, joint_vels);

UNGAR_BRANCH_MVARIABLE(qv, q, v);

}  // namespace Variables
}  // namespace ANYmalB

int main() {
    namespace vs = ANYmalB::Variables;
    namespace qs = RBD::Quantities;

    /***************                      Create robot.                       ***************/
    // Create the quadruped robot ANYmal B with Ungar.
    const std::string urdfFilename{UNGAR_DATA_FOLDER
                                   "/robots/anymal_b_description/robots/anymal.urdf"};
    Robot<real_t> robot{urdfFilename};

    // Log information about ANYmal B.
    UNGAR_LOG(info, "Model name: {}", robot.Model().name);
    UNGAR_LOG(info, "Dimension of configuration vector representation: {}", robot.Model().nq);
    UNGAR_LOG(info, "Dimension of velocity vector space: {}", robot.Model().nv);
    UNGAR_LOG(info, "Number of joints: {}", robot.Model().njoints);
    UNGAR_LOG(info, Utils::DASH_LINE_SEPARATOR);

    /***************                      Use algorithms.                     ***************/
    // Create data required by the RBD algorithms.
    VectorXr q   = robot.RandomConfiguration();
    VectorXr v   = VectorXr::Random(vs::v.Size());
    VectorXr a   = VectorXr::Random(vs::v.Size());
    VectorXr tau = VectorXr::Random(vs::v.Size());

    /// @note The limits of the free flyer joint default to infinite. The following
    ///       lines prevent infinite numbers that would break the computations.
    {
        // Create an m-lazy variable map for the configuration vector.
        auto q_ = MakeMVariableLazyMap(q, vs::q);
        q_.Get(vs::position).setRandom();
    }

    UNGAR_LOG(info, "Computing the center of mass acceleration...");
    auto res = robot.Compute(qs::com_acceleration);
    res.At(q, v, a);
    UNGAR_LOG(info, "Done. Center of mass acceleration:\n{}", robot.Get(qs::com_acceleration));
    UNGAR_LOG(info, Utils::DASH_LINE_SEPARATOR);

    UNGAR_LOG(info, "Computing the composite rigid body inertia...");
    robot.Compute(qs::composite_rigid_body_inertia).At(q, v);
    UNGAR_LOG(info,
              "Done. Composite rigid body inertia:\n{}",
              robot.Get(qs::composite_rigid_body_inertia));
    UNGAR_LOG(info, Utils::DASH_LINE_SEPARATOR);

    UNGAR_LOG(info, "Computing the generalized accelerations...");
    robot.Compute(qs::generalized_accelerations).At(q, v, tau);
    UNGAR_LOG(
        info, "Done. Generalized accelerations:\n{}", robot.Get(qs::generalized_accelerations));
    UNGAR_LOG(info, Utils::DASH_LINE_SEPARATOR);

    /***************                  Generate derivatives.                   ***************/
    // Implement autodiff function that computes the centroidal momentum of the robot.
    UNGAR_LOG(
        info,
        "Generating the autodiff function to compute the centroidal momentum of the robot...");
    auto functionImpl = [&](const VectorXad& qvUnderlying, VectorXad& y) -> void {
        Robot<ad_scalar_t> robotAD{urdfFilename};
        const auto qv_ = MakeMVariableLazyMap(qvUnderlying, vs::qv);
        robotAD.Compute(qs::centroidal_momentum).At(qv_.Get(vs::q), qv_.Get(vs::v));
        y = robotAD.Get(qs::centroidal_momentum).toVector_impl();
    };
    Autodiff::Function::Blueprint blueprint{
        functionImpl, vs::qv.Size(), 0_idx, "robot_example", EnabledDerivatives::JACOBIAN};
    Autodiff::Function function = MakeFunction(blueprint, true);

    // Compare results.
    robot.Compute(qs::centroidal_momentum).At(q, v);

    VectorXr qv{q.size() + v.size()};
    qv << q, v;
    const bool ok = Utils::CompareMatrices(function(qv),
                                           "Autodiff function"sv,
                                           robot.Get(qs::centroidal_momentum).toVector_impl(),
                                           "Ground truth"sv);
    UNGAR_ASSERT(ok);

    UNGAR_LOG(info, "Autodiff function:\n{}", function(qv).transpose());
    UNGAR_LOG(
        info, "Ground truth:\n{}", robot.Get(qs::centroidal_momentum).toVector_impl().transpose());
    UNGAR_LOG(info,
              "Autodiff Jacobian (6 rightmost columns):\n{}",
              function.Jacobian(qv).toDense().rightCols(6_idx));

    return 0;
}
