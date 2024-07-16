/******************************************************************************
 *
 * @file ungar/rbd/robot.hpp
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

#ifndef _UNGAR__RBD__ROBOT_HPP_
#define _UNGAR__RBD__ROBOT_HPP_

#include "ungar/rbd/quantity.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "ungar/utils/utils.hpp"

namespace Ungar {

template <Concepts::Scalar _Scalar = real_t>
class Robot {
  public:
    template <typename _RootJoint = pinocchio::JointModelFreeFlyer>
    Robot(const std::string& urdfFilename, const _RootJoint& rootJoint = {}) : _model{}, _data{} {
        pinocchio::Model model;
        pinocchio::urdf::buildModel(urdfFilename, rootJoint, model);

        _model = model.cast<_Scalar>();
        _data  = std::make_unique<pinocchio::DataTpl<_Scalar>>(_model);
        UNGAR_ASSERT(_model.check(*_data));
    }

    Robot(const Robot& other)
        : _model{other._model}, _data{std::make_unique<pinocchio::DataTpl<_Scalar>>(_model)} {
        UNGAR_ASSERT(_model.check(*_data));
    }

    constexpr auto Compute(auto quantity) {
        return RBD::Evaluator<quantity, _Scalar>{_model, *_data};
    }

    decltype(auto) Get(auto quantity) const {
        const auto getter = RBD::Getter<quantity, _Scalar>{_model, *_data};
        return getter.Get();
    }

    decltype(auto) Get(auto quantity, auto&&... args) const {
        const auto getter = RBD::Getter<quantity, _Scalar>{_model, *_data};
        return getter.Get(std::forward<decltype(args)>(args)...);
    }

    decltype(auto) Get(auto quantity) {
        auto getter = RBD::Getter<quantity, _Scalar>{_model, *_data};
        return getter.Get();
    }

    decltype(auto) Get(auto quantity, auto&&... args) {
        auto getter = RBD::Getter<quantity, _Scalar>{_model, *_data};
        return getter.Get(std::forward<decltype(args)>(args)...);
    }

    const pinocchio::ModelTpl<_Scalar>& Model() const {
        return _model;
    }

    pinocchio::ModelTpl<_Scalar>& Model() {
        return _model;
    }

    const pinocchio::DataTpl<_Scalar>& Data() const {
        return *_data;
    }

    pinocchio::DataTpl<_Scalar>& Data() {
        return *_data;
    }

    VectorX<_Scalar> RandomConfiguration() const {
        return pinocchio::randomConfiguration(_model);
    }

  protected:
    pinocchio::ModelTpl<_Scalar> _model;
    std::unique_ptr<pinocchio::DataTpl<_Scalar>> _data;
};

}  // namespace Ungar

#endif /* _UNGAR__RBD__ROBOT_HPP_ */
