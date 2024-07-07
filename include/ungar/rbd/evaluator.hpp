/******************************************************************************
 *
 * @file ungar/rbd/evaluator.hpp
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

#ifndef _UNGAR__RBD__EVALUATOR_HPP_
#define _UNGAR__RBD__EVALUATOR_HPP_

#include "ungar/data_types.hpp"
#include "ungar/utils/macros/for_each.hpp"

namespace Ungar {

namespace RBD {

template <auto _QUANTITY, typename _Scalar>
struct Evaluator {
    static_assert(dependent_false<decltype(_QUANTITY)>,
                  "The evaluator for the given quantity was not included or does not exist.");
};

#define _UNGAR_MAKE_EVALUATOR_HELPER_1(arg) , const auto& arg
#define _UNGAR_MAKE_EVALUATOR_HELPER_2(arg) , arg
#define UNGAR_MAKE_EVALUATOR(quantity, algorithm, firstArg, ...)                       \
    template <typename _Scalar>                                                        \
    struct Evaluator<::Ungar::RBD::Quantities::quantity, _Scalar> {                    \
        void At(const auto& firstArg UNGAR_FOR_EACH(_UNGAR_MAKE_EVALUATOR_HELPER_1,    \
                                                    __VA_ARGS__)) {                    \
            ::pinocchio::algorithm(                                                    \
                model,                                                                 \
                data,                                                                  \
                firstArg UNGAR_FOR_EACH(_UNGAR_MAKE_EVALUATOR_HELPER_2, __VA_ARGS__)); \
        }                                                                              \
                                                                                       \
        const ::pinocchio ::ModelTpl<_Scalar>& model;                                  \
        ::pinocchio ::DataTpl<_Scalar>& data;                                          \
    }

}  // namespace RBD
}  // namespace Ungar

#endif /* _UNGAR__RBD__EVALUATOR_HPP_ */
