/******************************************************************************
 *
 * @file ungar/variable_map.hpp
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

#ifndef _UNGAR__VARIABLE_MAP_HPP_
#define _UNGAR__VARIABLE_MAP_HPP_

#include "ungar/utils/integral_map.hpp"
#include "ungar/variable_lazy_map.hpp"

namespace Ungar {

template <typename _Scalar, Concepts::Variable _Variable>
class VariableMap : private VectorX<_Scalar>,
                    private VariableLazyMap<_Scalar, _Variable, hana::true_> {
  private:
    using VectorType   = VectorX<_Scalar>;
    using LazyMapType  = VariableLazyMap<_Scalar, _Variable, hana::true_>;
    using VariableType = _Variable;

    template <Concepts::Variable _Var>
    static constexpr auto SizeCImpl(const _Var& var) {
        if constexpr (_Var::IsQuaternion()) {
            return Q_c;
        } else {
            return var.SizeC();
        }
    }

    static auto __attribute__((optimize(0)))
    InitImpl(LazyMapType lazyMap, const VariableType& var, Concepts::HanaBool auto initConstMap) {
        add_const_if_t<LazyMapType, initConstMap>& lm{lazyMap};

        auto unique = var.Unique([](const auto& v) { return SizeCImpl(v); });
        auto map    = hana::unpack(unique, [](auto... sizes) {
            return hana::make_map(hana::make_pair(
                sizes,
                integral_map<decltype(lm.GetImpl(Variable{hana::string_c<>, sizes})),
                             static_cast<size_t>(VariableType::Size())>{})...);
        });
        var.ForEach([&](const auto& v) {
            static_assert(decltype(hana::contains(map, SizeCImpl(v)))::value);
            map[SizeCImpl(v)].emplace(v.Index(), lm.GetImpl(v));
        });
        return map;
    }

    LazyMapType& AsLazyMap() {
        return static_cast<LazyMapType&>(*this);
    }

    const LazyMapType& AsLazyMap() const {
        return static_cast<const LazyMapType&>(*this);
    }

  public:
    using ScalarType = _Scalar;

    VariableMap(hana::basic_type<_Scalar>, const VariableType& var)
        : VectorType{var.Size()},
          LazyMapType{Get(), var},
          _impl{InitImpl(AsLazyMap(), var, hana::false_c)},
          _constImpl{InitImpl(AsLazyMap(), var, hana::true_c)} {
        UNGAR_ASSERT(!var.Index());
    }

    decltype(auto) Get() const {
        return static_cast<const VectorType&>(*this);
    }

    decltype(auto) Get() {
        return static_cast<VectorType&>(*this);
    }

    decltype(auto) Get(auto&&... args) const {
        return Get1(AsLazyMap()._variable(std::forward<decltype(args)>(args)...));
    }

    decltype(auto) Get(auto&&... args) {
        return Get1(AsLazyMap()._variable(std::forward<decltype(args)>(args)...));
    }

    template <Concepts::Variable _Var>
    const auto& Get1(const _Var& var) const {
        if constexpr (decltype(hana::contains(_constImpl, SizeCImpl(var)))::value) {
            UNGAR_ASSERT(_constImpl[SizeCImpl(var)].contains(var.Index()));
            if constexpr (_Var::Size() == 1_idx) {
                return _constImpl[SizeCImpl(var)][var.Index()].get();
            } else {
                return _constImpl[SizeCImpl(var)][var.Index()];
            }
        } else {
            Unreachable();
        }
    }

    template <Concepts::Variable _Var>
    auto& Get1(const _Var& var) {
        if constexpr (decltype(hana::contains(_impl, SizeCImpl(var)))::value) {
            UNGAR_ASSERT(_impl[SizeCImpl(var)].contains(var.Index()));
            if constexpr (_Var::Size() == 1_idx) {
                return _impl[SizeCImpl(var)][var.Index()].get();
            } else {
                return _impl[SizeCImpl(var)][var.Index()];
            }
        } else {
            Unreachable();
        }
    }

  private:
    decltype(InitImpl(LazyMapType{}, VariableType{}, hana::false_c)) _impl;
    decltype(InitImpl(LazyMapType{}, VariableType{}, hana::true_c)) _constImpl;
};

template <typename _Scalar, Concepts::Variable _Variable>
VariableMap(const _Variable& var, hana::basic_type<_Scalar>) -> VariableMap<_Scalar, _Variable>;

template <typename _Scalar>
inline static auto MakeVariableMap(const Concepts::Variable auto& var) {
    return VariableMap{hana::type_c<_Scalar>, var};
}

}  // namespace Ungar

#endif /* _UNGAR__VARIABLE_MAP_HPP_ */
