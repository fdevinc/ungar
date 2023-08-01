/******************************************************************************
 *
 * @file ungar/autodiff/vector_composer.hpp
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

#ifndef _UNGAR__AUTODIFF__VECTOR_COMPOSER_HPP_
#define _UNGAR__AUTODIFF__VECTOR_COMPOSER_HPP_

#include "ungar/autodiff/data_types.hpp"

namespace Ungar {
namespace Autodiff {

class VectorComposer {
  public:
    VectorComposer() : _impl{} {
    }

    template <typename _Vector>  // clang-format off
    requires std::same_as<typename _Vector::Scalar, ad_scalar_t>
    VectorComposer& operator<<(
        const Eigen::MatrixBase<_Vector>& vector) {  // clang-format on
        _impl.emplace_back(vector);
        return *this;
    }

    VectorComposer& operator<<(const ad_scalar_t& scalar) {
        _impl.emplace_back((VectorXad{1_idx} << scalar).finished());
        return *this;
    }

    void Clear() {
        _impl.clear();
    }

    index_t Size() const {
        index_t size = 0_idx;
        for (const VectorXad& el : _impl) {
            size += el.size();
        }
        return size;
    }

    VectorXad Compose() const {
        VectorXad composedOutput{Size()};
        for (index_t i = 0; const VectorXad& el : _impl) {
            composedOutput.segment(i, el.size()) = el;
            i += el.size();
        }
        return composedOutput;
    }

  private:
    std::vector<VectorXad> _impl;
};

}  // namespace Autodiff
}  // namespace Ungar

#endif /* _UNGAR__AUTODIFF__VECTOR_COMPOSER_HPP_ */
