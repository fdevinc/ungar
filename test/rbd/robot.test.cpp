/******************************************************************************
 *
 * @file ungar/test/robot.test.cpp
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

#include <gtest/gtest.h>

#include "ungar/autodiff/function.hpp"
#include "ungar/mvariable_lazy_map.hpp"
#include "ungar/rbd/robot.hpp"

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

UNGAR_LEAF_MVARIABLE(b_generalized_force, 3);
UNGAR_LEAF_MVARIABLE(b_generalized_torque, 3);
UNGAR_BRANCH_MVARIABLE(base_wrench, b_generalized_force, b_generalized_torque);

UNGAR_BRANCH_MVARIABLE(leg_joint_torques, hip_aa, hip_fe, knee_fe);
UNGAR_MVARIABLE_ARRAY(joint_torques, leg_joint_torques, NUM_LEGS);
UNGAR_BRANCH_MVARIABLE(tau, base_wrench, joint_torques);

UNGAR_BRANCH_MVARIABLE(qvtau, q, v, tau);

}  // namespace Variables
}  // namespace ANYmalB

namespace Ungar {
namespace Test {
namespace {

TEST(RobotTest, Constructor) {
    namespace vs = ANYmalB::Variables;

    // Create the quadruped robot ANYmal B with Ungar.
    const std::string urdfFilename{UNGAR_DATA_FOLDER
                                   "/robots/anymal_b_description/robots/anymal.urdf"};
    Robot<real_t> robot{urdfFilename};

    // Create the robot with Pinocchio.
    pinocchio::Model model;
    pinocchio::JointModelFreeFlyer rootJoint;
    pinocchio::urdf::buildModel(urdfFilename, rootJoint, model);

    ASSERT_TRUE((model.name == robot.Model().name) &&
                (model.nq == robot.Model().nq && model.nq == vs::q.Size()) &&
                (model.nv == robot.Model().nv && model.nv == vs::v.Size()) &&
                (model.njoints == robot.Model().njoints));
}

TEST(RobotTest, Autodiff) {
    namespace vs = ANYmalB::Variables;
    namespace qs = RBD::Quantities;

    // Create ANYmal B with Ungar.
    const std::string urdfFilename{UNGAR_DATA_FOLDER
                                   "/robots/anymal_b_description/robots/anymal.urdf"};
    Robot<real_t> robot{urdfFilename};

    // Create robot with Pinocchio.
    pinocchio::Model model;
    pinocchio::JointModelFreeFlyer rootJoint;
    pinocchio::urdf::buildModel(urdfFilename, rootJoint, model);

    // Create autodiff function for the ABA algorithm.
    auto functionImpl = [&](const VectorXad& qvtauUnderlying, VectorXad& y) -> void {
        Robot<ad_scalar_t> robotAD{urdfFilename};
        const auto qvtau_ = MakeMVariableLazyMap(qvtauUnderlying, vs::qvtau);

        const ad_scalar_t standardGravity{9.80665};
        robotAD.Compute(qs::generalized_accelerations)
            .At(qvtau_.Get(vs::q), qvtau_.Get(vs::v), qvtau_.Get(vs::tau));
        y = robotAD.Get(qs::generalized_accelerations) / standardGravity;
    };
    Autodiff::Function::Blueprint blueprint{
        functionImpl, vs::qvtau.Size(), 0_idx, "robot_test", EnabledDerivatives::JACOBIAN};
    Autodiff::Function function = MakeFunction(blueprint, true);

    for (auto func = [&](const VectorXr& qvtauUnderlying) -> VectorXr {
             Robot<real_t> robot{urdfFilename};
             const auto qvtau_ = MakeMVariableLazyMap(qvtauUnderlying, vs::qvtau);
             const real_t standardGravity{9.80665};
             robot.Compute(qs::generalized_accelerations)
                 .At(qvtau_.Get(vs::q), qvtau_.Get(vs::v), qvtau_.Get(vs::tau));
             return robot.Get(qs::generalized_accelerations) / standardGravity;
         };
         const auto k : enumerate(1024_idx)) {
        // Create data required by the RBD algorithms.
        VectorXr q   = robot.RandomConfiguration();
        VectorXr v   = VectorXr::Random(vs::v.Size());
        VectorXr tau = VectorXr::Random(vs::v.Size());
        {
            auto q_ = MakeMVariableLazyMap(q, vs::q);
            q_.Get(vs::position).setRandom();
        }

        VectorXr qvtau{q.size() + v.size() + tau.size()};
        qvtau << q, v, tau;

        // Test generated code.
        EXPECT_TRUE(function.TestFunction(qvtau, func));
        EXPECT_TRUE(function.TestJacobian(qvtau));
    }
}

}  // namespace
}  // namespace Test
}  // namespace Ungar

int main() {
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
