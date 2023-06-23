/******************************************************************************
 *
 * @file ungar/variable_lazy_map.hpp
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

#ifndef _UNGAR__VARIABLE_LAZY_MAP_HPP_
#define _UNGAR__VARIABLE_LAZY_MAP_HPP_

#include "ungar/variable.hpp"

namespace Ungar {

template <typename _Scalar, Concepts::Variable _Variable, Concepts::HanaBool _EnableMutableMembers>
class VariableLazyMap {
  private:
    template <typename _Underlying>
    static constexpr auto UnderlyingSize(hana::basic_type<_Underlying>) {
        if constexpr (Concepts::DenseMatrixExpression<_Underlying>) {
            return std::remove_cvref_t<_Underlying>::RowsAtCompileTime;
        } else {
            return static_cast<index_t>(std::ranges::size(_Underlying{}));
        }
    }

  public:
    using ScalarType = _Scalar;

    constexpr VariableLazyMap() : _data{nullptr}, _variable{} {
        Unreachable();
    }

    VariableLazyMap(const VectorX<_Scalar>& underlying, const _Variable& var)
        : _data{underlying.data()}, _variable{var} {
        UNGAR_ASSERT(underlying.size() == var.Size() && !var.Index());
    }

    VariableLazyMap(VectorX<_Scalar>& underlying, const _Variable& var)
        : _data{underlying.data()}, _variable{var} {
        UNGAR_ASSERT(underlying.size() == var.Size() && !var.Index());
    }

    template <typename _Underlying>
    constexpr VariableLazyMap(const _Underlying& underlying, const _Variable& var) requires(
        UnderlyingSize(hana::type_c<_Underlying>) == static_cast<size_t>(_Variable::Size()))
        : _data{std::ranges::data(underlying)}, _variable{var} {
        UNGAR_ASSERT(!var.Index());
    }

    template <typename _Underlying>
    constexpr VariableLazyMap(_Underlying& underlying, const _Variable& var) requires(
        UnderlyingSize(hana::type_c<_Underlying>) == static_cast<size_t>(_Variable::Size()))
        : _data{std::ranges::data(underlying)}, _variable{var} {
        UNGAR_ASSERT(!var.Index());
    }

    decltype(auto) Get() const {
        return Get1(_variable);
    }

    decltype(auto) Get() requires _EnableMutableMembers::value {
        return Get1(_variable);
    }

    decltype(auto) Get(auto&&... args) const {
        return Get1(_variable(std::forward<decltype(args)>(args)...));
    }

    decltype(auto) Get(auto&&... args) requires _EnableMutableMembers::value {
        return Get1(_variable(std::forward<decltype(args)>(args)...));
    }

    decltype(auto) Get1(const Concepts::Variable auto& var) const {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::Size() == 1_idx) {
            return GetImpl(var).get();
        } else {
            return GetImpl(var);
        }
    }

    decltype(auto) Get1(const Concepts::Variable auto& var) {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::Size() == 1_idx) {
            return GetImpl(var).get();
        } else {
            return GetImpl(var);
        }
    }

  protected:
    auto GetImpl(const Concepts::Variable auto& var) const {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::IsQuaternion()) {
            return Eigen::Map<const Quaternion<_Scalar>>{_data + var.Index()};
        } else {
            constexpr auto size = VariableType::Size();
            if constexpr (size == 1_idx) {
                return std::cref(*(_data + var.Index()));
            } else if constexpr (size <= 32_idx) {
                return Eigen::Map<const Vector<_Scalar, static_cast<int>(size)>>{_data +
                                                                                 var.Index()};
            } else {
                return Eigen::Map<const VectorX<_Scalar>>{_data + var.Index(), size};
            }
        }
    }

    auto GetImpl(const Concepts::Variable auto& var) requires _EnableMutableMembers::value {
        using VariableType = std::remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::IsQuaternion()) {
            return Eigen::Map<Quaternion<_Scalar>>{const_cast<_Scalar*>(_data) + var.Index()};
        } else {
            constexpr auto size = VariableType::Size();
            if constexpr (size == 1_idx) {
                return std::ref(*(const_cast<_Scalar*>(_data) + var.Index()));
            } else if constexpr (size <= 32_idx) {
                return Eigen::Map<Vector<_Scalar, static_cast<int>(size)>>{
                    const_cast<_Scalar*>(_data) + var.Index()};
            } else {
                return Eigen::Map<VectorX<_Scalar>>{const_cast<_Scalar*>(_data) + var.Index(),
                                                    size};
            }
        }
    }

  private:
    template <typename _Sc, Concepts::Variable _Var>
    friend class VariableMap;

    const _Scalar* _data;
    _Variable _variable;
};

VariableLazyMap(const auto& underlying, const Concepts::Variable auto& var)
    -> VariableLazyMap<std::remove_cvref_t<decltype(*std::ranges::data(underlying))>,
                       std::remove_cvref_t<decltype(var)>,
                       hana::bool_<false>>;

VariableLazyMap(auto& underlying, const Concepts::Variable auto& var)
    -> VariableLazyMap<std::remove_reference_t<decltype(*std::ranges::data(underlying))>,
                       std::remove_cvref_t<decltype(var)>,
                       hana::bool_<true>>;

inline static auto MakeVariableLazyMap(const auto& underlying, const Concepts::Variable auto& var) {
    return VariableLazyMap{underlying, var};
}

inline static auto MakeVariableLazyMap(auto& underlying, const Concepts::Variable auto& var) {
    return VariableLazyMap{underlying, var};
}

}  // namespace Ungar

#endif /* _UNGAR__VARIABLE_LAZY_MAP_HPP_ */
