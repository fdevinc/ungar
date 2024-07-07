/******************************************************************************
 *
 * @file ungar/rbd/quantities.hpp
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

#ifndef _UNGAR__RBD__QUANTITY_HPP_
#define _UNGAR__RBD__QUANTITY_HPP_

#include "pinocchio/codegen/cppadcg.hpp"

#include "ungar/rbd/evaluator.hpp"
#include "ungar/rbd/getter.hpp"

namespace Ungar {
namespace RBD {
namespace Quantities {

/**
 * @brief Define a quantity computable using rigid body dynamics algorithms.
 */
#define UNGAR_MAKE_QUANTITY(name)      \
    inline constexpr struct name##_t { \
    } name

}  // namespace Quantities
}  // namespace RBD
}  // namespace Ungar

#endif /* _UNGAR__RBD__QUANTITY_HPP_ */
