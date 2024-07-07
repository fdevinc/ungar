/******************************************************************************
 *
 * @file ungar/example/rbd/quantity.example.cpp
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

#include "pinocchio/algorithm/frames.hpp"
#include "ungar/mvariable_lazy_map.hpp"

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

inline constexpr auto FOOT_FRAME_IDS = std::array{12UL, 22UL, 32UL, 42UL};

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

namespace Quantities {

UNGAR_MAKE_QUANTITY(foot_pose);

}
}  // namespace ANYmalB

namespace Ungar {
namespace RBD {

template <typename _Scalar>
struct Evaluator<ANYmalB::Quantities::foot_pose, _Scalar> {
    void At(const auto& q) {
        namespace vs = ANYmalB::Variables;

        pinocchio::forwardKinematics(model, data, q);
        for (const auto frameID : vs::FOOT_FRAME_IDS) {
            pinocchio::updateFramePlacement(model, data, frameID);
        }
    }

    const pinocchio::ModelTpl<_Scalar>& model;
    pinocchio::DataTpl<_Scalar>& data;
};

template <typename _Scalar>
struct Getter<ANYmalB::Quantities::foot_pose, _Scalar> {
    const auto& Get(const index_t foot) const {
        namespace vs = ANYmalB::Variables;

        return data.oMf[vs::FOOT_FRAME_IDS[static_cast<size_t>(foot)]];
    }

    auto& Get(const index_t foot) {
        namespace vs = ANYmalB::Variables;

        return data.oMf[vs::FOOT_FRAME_IDS[static_cast<size_t>(foot)]];
    }

    const pinocchio::ModelTpl<_Scalar>& model;
    pinocchio::DataTpl<_Scalar>& data;
};

}  // namespace RBD
}  // namespace Ungar

int main() {
    namespace vs = ANYmalB::Variables;
    namespace qs = ANYmalB::Quantities;

    /***************                      Create robot.                       ***************/
    // Create the quadruped robot ANYmal B with Ungar.
    const std::string urdfFilename{UNGAR_DATA_FOLDER
                                   "/robots/anymal_b_description/robots/anymal.urdf"};
    Robot<real_t> robot{urdfFilename};

    /***************                 Testing custom quantity.                 ***************/
    // Create data required by the RBD algorithms.
    VectorXr q = robot.RandomConfiguration();

    /// @note The limits of the free flyer joint default to infinite. The following
    ///       lines prevent infinite numbers that would break the computations.
    {
        // Create an m-lazy variable map for the configuration vector.
        auto q_ = MakeMVariableLazyMap(q, vs::q);
        q_.Get(vs::position).setRandom();
    }

    UNGAR_LOG(info, "Computing the pose of feet...");
    robot.Compute(qs::foot_pose).At(q);
    UNGAR_LOG(info, "Done.");
    for (const auto foot : {vs::LF, vs::LH, vs::RF, vs::RH}) {
        UNGAR_LOG(info, "Pose of the {}-th foot:\n{}", foot, robot.Get(qs::foot_pose, foot));
    }
    UNGAR_LOG(info, Utils::DASH_LINE_SEPARATOR);

    return 0;
}
