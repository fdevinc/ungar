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

#include <type_traits>
#include "ungar/data_types.hpp"
#include "ungar/variable.hpp"

namespace Ungar {

template <typename _Scalar, typename _Variable, bool _ENABLE_MUTABLE_MEMBERS>
class VariableLazyMap {
  private:
    template <typename _Underlying>
    static constexpr auto UnderlyingSize(hana::basic_type<_Underlying>) {
        if constexpr (is_dense_vector_expression_v<_Underlying>) {
            return remove_cvref_t<_Underlying>::RowsAtCompileTime;
        } else {
            return static_cast<index_t>(nano::ranges::size(_Underlying{}));
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

    template <
        typename _Underlying,
        typename _V,
        std::enable_if_t<UnderlyingSize(hana::type_c<_Underlying>) == _V::Size(), bool> = true>
    constexpr VariableLazyMap(const _Underlying& underlying, const _V& var)
        : _data{nano::ranges::data(underlying)}, _variable{var} {
        UNGAR_ASSERT(!var.Index());
    }

    template <
        typename _Underlying,
        typename _V,
        std::enable_if_t<UnderlyingSize(hana::type_c<_Underlying>) == _V::Size(), bool> = true>
    constexpr VariableLazyMap(_Underlying& underlying, const _V& var)
        : _data{nano::ranges::data(underlying)}, _variable{var} {
        UNGAR_ASSERT(!var.Index());
    }

    decltype(auto) Get() const {
        return Get1(_variable);
    }

    template <typename _Dummy = _Scalar,
              std::enable_if_t<dependent_bool_value<_ENABLE_MUTABLE_MEMBERS, _Dummy>, bool> = true>
    decltype(auto) Get() {
        return Get1(_variable);
    }

    template <typename... _Args>
    decltype(auto) Get(_Args&&... args) const {
        return Get1(_variable(std::forward<decltype(args)>(args)...));
    }

    template <
        typename... _Args,
        std::enable_if_t<dependent_bool_value<_ENABLE_MUTABLE_MEMBERS, _Args...>, bool> = true>
    decltype(auto) Get(_Args&&... args) {
        return Get1(_variable(std::forward<decltype(args)>(args)...));
    }

    template <typename _V>
    decltype(auto) Get1(const _V& var) const {
        using VariableType = remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::Size() == 1_idx) {
            return GetImpl(var).get();
        } else {
            return GetImpl(var);
        }
    }

    template <typename _V,
              std::enable_if_t<dependent_bool_value<_ENABLE_MUTABLE_MEMBERS, _V>, bool> = true>
    decltype(auto) Get1(const _V& var) {
        using VariableType = remove_cvref_t<decltype(var)>;
        if constexpr (VariableType::Size() == 1_idx) {
            return GetImpl(var).get();
        } else {
            return GetImpl(var);
        }
    }

  protected:
    template <typename _V>
    auto GetImpl(const _V& var) const {
        using VariableType = remove_cvref_t<decltype(var)>;
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

    template <typename _V,
              std::enable_if_t<dependent_bool_value<_ENABLE_MUTABLE_MEMBERS, _V>, bool> = true>
    auto GetImpl(const _V& var) {
        using VariableType = remove_cvref_t<decltype(var)>;
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
    template <typename _Sc, typename _Var>
    friend class VariableMap;

    const _Scalar* _data;
    _Variable _variable;
};

template <typename _Underlying, typename _Variable>
VariableLazyMap(const _Underlying& underlying, const _Variable& var)
    -> VariableLazyMap<remove_cvref_t<decltype(*nano::ranges::data(underlying))>,
                       remove_cvref_t<decltype(var)>,
                       false>;

template <typename _Underlying, typename _Variable>
VariableLazyMap(_Underlying& underlying, const _Variable& var)
    -> VariableLazyMap<std::remove_reference_t<decltype(*nano::ranges::data(underlying))>,
                       remove_cvref_t<decltype(var)>,
                       true>;

template <typename _Underlying, typename _Variable>
inline static auto MakeVariableLazyMap(const _Underlying& underlying, const _Variable& var) {
    return VariableLazyMap{underlying, var};
}

template <typename _Underlying, typename _Variable>
inline static auto MakeVariableLazyMap(_Underlying& underlying, const _Variable& var) {
    return VariableLazyMap{underlying, var};
}

}  // namespace Ungar

#endif /* _UNGAR__VARIABLE_LAZY_MAP_HPP_ */
