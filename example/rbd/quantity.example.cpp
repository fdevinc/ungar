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

inline auto FOOT_FRAME_NAMES = std::array{"LF_FOOT"s, "LH_FOOT"s, "RF_FOOT"s, "RH_FOOT"s};
inline constexpr auto FOOT_FRAME_INDICES = std::array{12_idx, 22_idx, 32_idx, 42_idx};

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

/***************                 Define custom quantity.                  ***************/
/**
 * @brief To compute and get a quantity using \c Ungar::Robot, the following steps must be
 *        followed:
 *          1) define the quantity;
 *          2) define a corresponding evaluator;
 *          3) define a corresponding getter.
 */

/*======================================================================================*/
/*~~~~~~~~~~~~~|                     PART I: QUANTITY                     |~~~~~~~~~~~~~*/
/*======================================================================================*/
/// @brief Ungar provides a convenient macro to simplify the definition of quantities.
namespace Quantities {

UNGAR_MAKE_QUANTITY(frame_pose);  // Quantity corresponding to the pose of a frame.

}
}  // namespace ANYmalB

/*======================================================================================*/
/*~~~~~~~~~~~~~|                    PART II: EVALUATOR                    |~~~~~~~~~~~~~*/
/*======================================================================================*/
/// @brief Evaluators wrap Pinocchio's functionalities and provide the user with a
///        minimal interface.
namespace Ungar {
namespace RBD {

/**
 * @brief Evaluators must implement a (possibly overloaded) function \c At computing
 *        the desired quantity and returning \c void.
 */
template <typename _Scalar>
struct Evaluator<ANYmalB::Quantities::frame_pose, _Scalar> {
    void At(const auto& q) {
        // Compute the pose of each joint frame.
        pinocchio::forwardKinematics(model, data, q);
    }

    const pinocchio::ModelTpl<_Scalar>& model;
    pinocchio::DataTpl<_Scalar>& data;
};

}  // namespace RBD
}  // namespace Ungar

/*======================================================================================*/
/*~~~~~~~~~~~~~|                     PART III: GETTER                     |~~~~~~~~~~~~~*/
/*======================================================================================*/
/// @brief Getters allow the user to easily retrieve desired quantities.
namespace Ungar {
namespace RBD {

/**
 * @brief Getters must implement a (possibly overloaded) function \c Get returning
 *        the desired quantity.
 */
template <typename _Scalar>
struct Getter<ANYmalB::Quantities::frame_pose, _Scalar> {
    const auto& Get(const index_t frameIndex) const {
        UNGAR_ASSERT(static_cast<index_t>(frameIndex) < model.nframes);

        const size_t frameID = static_cast<size_t>(frameIndex);

        // Update the pose of the desired frame before returning it.
        pinocchio::updateFramePlacement(model, data, frameID);
        return data.oMf[frameID];
    }

    const auto& Get(const std::string& frameName) const {
        UNGAR_ASSERT(model.existFrame(frameName));

        const size_t frameID = model.getFrameId(frameName);

        // Update the pose of the desired frame before returning it.
        pinocchio::updateFramePlacement(model, data, frameID);
        return data.oMf[frameID];
    }

    auto& Get(const index_t frameIndex) {
        UNGAR_ASSERT(static_cast<index_t>(frameIndex) < model.nframes);

        const size_t frameID = static_cast<size_t>(frameIndex);

        // Update the pose of the desired frame before returning it.
        pinocchio::updateFramePlacement(model, data, frameID);
        return data.oMf[frameID];
    }

    auto& Get(const std::string& frameName) {
        UNGAR_ASSERT(model.existFrame(frameName));

        const size_t frameID = model.getFrameId(frameName);

        // Update the pose of the desired frame before returning it.
        pinocchio::updateFramePlacement(model, data, frameID);
        return data.oMf[frameID];
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

    UNGAR_LOG(info, "Computing the pose of the feet...");
    robot.Compute(qs::frame_pose).At(q);
    UNGAR_LOG(info, "Done.");
    for (const auto i : enumerate(vs::NUM_LEGS) | cast_to<size_t>) {
        const auto [footFrameIndex, footFrameName] =
            std::pair{vs::FOOT_FRAME_INDICES[i], vs::FOOT_FRAME_NAMES[i]};
        [[maybe_unused]] const pinocchio::SE3 footFramePoseByIndex =
            robot.Get(qs::frame_pose, footFrameIndex);
        [[maybe_unused]] const pinocchio::SE3 footFramePoseByName =
            robot.Get(qs::frame_pose, footFrameName);

        UNGAR_LOG(info, Utils::DASH_LINE_SEPARATOR);
        UNGAR_LOG(info, "Pose of the foot ('{}'):\n{}", footFrameName, footFramePoseByIndex);
        UNGAR_ASSERT(footFramePoseByIndex.isEqual(footFramePoseByName));
    }

    return 0;
}
