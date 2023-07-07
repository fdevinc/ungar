/******************************************************************************
 *
 * @file ungar/variable.hpp
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

#ifndef _UNGAR__VARIABLE_HPP_
#define _UNGAR__VARIABLE_HPP_

#include "ungar/utils/utils.hpp"

namespace Ungar {

template <typename _N, typename _M, index_t _S>
struct Variable;

template <class _T>
struct is_variable : std::false_type {};
template <typename _N, typename _M, index_t _S>
struct is_variable<Variable<_N, _M, _S>> : std::true_type {};
template <class _T>
constexpr bool is_variable_v = is_variable<_T>::value;

template <typename _Count, typename _Variable>
struct VariableProductExpr {
    constexpr VariableProductExpr(_Count cnt_, _Variable var_) : cnt{cnt_}, var{var_} {
    }

    _Count cnt;
    _Variable var;
};

template <typename _Count, typename _Variable>
VariableProductExpr(_Count cnt_, _Variable var_) -> VariableProductExpr<_Count, _Variable>;

template <class _T>
struct is_variable_product_expr : std::false_type {};
template <typename _Count, typename _Variable>
struct is_variable_product_expr<VariableProductExpr<_Count, _Variable>> : std::true_type {};
template <class _T>
constexpr bool is_variable_product_expr_v = is_variable_product_expr<_T>::value;

template <typename _Name, typename _Map = decltype(hana::make_map()), index_t _Size = 0_idx>
class Variable {
  public:
    constexpr Variable() : _index{0_idx} {
    }

    constexpr Variable(_Name /* name */) : _index{0_idx} {
    }

    template <typename _IntegralConstant>
    constexpr Variable(_Name /* name */, _IntegralConstant /* size */) : _index{0_idx} {
    }

    static constexpr auto Name() {
        return _Name{};
    }

    static constexpr auto Size() {
        return SizeC().value;
    }

    constexpr auto Index() const {
        return _index;
    }

    static constexpr bool IsLeaf() {
        static_assert(hana::is_empty(hana::keys(DefaultMap())) || !_Size);
        return !!_Size;
    }

    static constexpr bool IsBranch() {
        return !_Size;
    }

    static constexpr bool IsScalar() {
        return _Size == 1_idx;
    }

    static constexpr bool IsVector() {
        return (!IsScalar()) && (!IsQuaternion());
    }

    static constexpr bool IsQuaternion() {
        return _Size == Q;
    }

    template <typename _T>
    constexpr auto operator<<=(_T el) const {
        static_assert(IsBranch());

        if constexpr (is_variable_v<_T>) {
            auto& var = el;
            auto map  = hana::insert(
                _map, hana::make_pair(var.Name(), var.CloneWithIndexOffset(Index() + SizeC())));
            return Make(Name(), map, Index(), 0_c);
        } else if constexpr (is_hana_tuple_v<_T>) {
            auto& tuple = el;
            auto lambda = [&](auto var, auto el) { return var <<= el; };
            return hana::fold(tuple, *this, lambda);
        } else if constexpr (is_variable_product_expr_v<_T>) {
            auto& prodExpr = el;
            auto array     = hana::unpack(hana::make_range(0_c, prodExpr.cnt), [&](auto... is) {
                return std::array{prodExpr.var.CloneWithIndexOffset(Index() + SizeC() +
                                                                    prodExpr.var.SizeC() * is)...};
            });
            auto map       = hana::insert(_map, hana::make_pair(prodExpr.var.Name(), array));
            return Make(Name(), map, Index(), 0_c);
        } else {
            static_assert(dependent_false<_T>);
            Unreachable();
        }
    }

    template <typename _Variable, typename... _Args>
    constexpr decltype(auto) operator()(_Variable var, _Args... args) const {
        auto tuple            = hana::make_tuple(hana::type_c<_Args>...);
        constexpr size_t SIZE = hana::length(hana::take_while(tuple, [](auto t) {
            using Type              = typename decltype(t)::type;
            constexpr bool INTEGRAL = std::is_integral_v<Type> || is_hana_integral_constant_v<Type>;
            return hana::bool_c<INTEGRAL>;
        }));

        if constexpr (SIZE == 0UL) {
            return GetImpl0(var.Name(), args...);
        } else if constexpr (SIZE == 1UL) {
            return GetImpl1(var.Name(), args...);
        } else if constexpr (SIZE == 2UL) {
            return GetImpl2(var.Name(), args...);
        } else if constexpr (SIZE == 3UL) {
            return GetImpl3(var.Name(), args...);
        }
    }

    template <typename _N, typename... _Args>
    constexpr decltype(auto) At(_N, _Args... args) const {
        auto tuple            = hana::make_tuple(hana::type_c<_Args>...);
        constexpr size_t SIZE = hana::length(hana::take_while(tuple, [](auto t) {
            using Type              = typename decltype(t)::type;
            constexpr bool INTEGRAL = std::is_integral_v<Type> || is_hana_integral_constant_v<Type>;
            return hana::bool_c<INTEGRAL>;
        }));
        static_assert(SIZE == sizeof...(_Args),
                      "All the arguments to the 'At' accessor must be indices.");

        if constexpr (SIZE == 0UL) {
            return GetImpl0(_N{}, args...);
        } else if constexpr (SIZE == 1UL) {
            return GetImpl1(_N{}, args...);
        } else if constexpr (SIZE == 2UL) {
            return GetImpl2(_N{}, args...);
        } else if constexpr (SIZE == 3UL) {
            return GetImpl3(_N{}, args...);
        }
    }

    template <typename _State, typename _F>
    auto Fold(_State&& state, _F&& f) const {
        return hana::fold(
            hana::values(_map),
            std::forward<_F>(f)(std::forward<_State>(state), *this),
            [&](auto&& s1, const auto& value) {
                if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                    return hana::fold(
                        value, std::forward<decltype(s1)>(s1), [&](auto&& s2, const auto& var) {
                            return var.Fold(std::forward<decltype(s2)>(s2), std::forward<_F>(f));
                        });
                } else {
                    return value.Fold(std::forward<decltype(s1)>(s1), std::forward<_F>(f));
                }
            });
    }

    template <typename _Variable>
    constexpr auto operator==(_Variable var) const {
        return boost::hana::bool_c<std::is_same_v<Variable, decltype(var)>>;
    }

    template <typename _Variable>
    constexpr auto operator!=(_Variable var) const {
        return boost::hana::bool_c<!std::is_same_v<Variable, decltype(var)>>;
    }

    template <typename _F>
    constexpr void ForEach(_F&& f) const {
        std::forward<_F>(f)(*this);
        hana::for_each(hana::values(_map), [&](const auto& value) {
            if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                for (const auto& var : value) {
                    var.ForEach(std::forward<_F>(f));
                }
            } else {
                value.ForEach(std::forward<_F>(f));
            }
        });
    }

  private:
    template <typename _N, typename _M, index_t _S>
    friend class Variable;
    template <typename _Scalar, typename _Variable>
    friend class VariableMap;

    static constexpr auto DefaultMap() {
        return Variable{}._map;
    }

    template <typename _N, typename _M, typename _S>
    static constexpr auto Make(_N name, _M map, const index_t index, _S size) {
        return Variable<_N, _M, _S::value>{name, map, index};
    }

    constexpr Variable(_Name /* name */, _Map map, const index_t index) : _map{map}, _index{index} {
    }

    constexpr auto Clone() const {
        return Variable{*this};
    }

    constexpr auto CloneWithIndexOffset(const index_t offset) const {
        Variable var = Clone();
        var.ForEach([offset](auto& v) { v._index += offset; });
        return var;
    }

    template <typename _Dummy>
    constexpr auto Clone(_Dummy&& /* dummy */) const {
        return Variable{*this};
    }

    static constexpr auto SizeC() {
        if constexpr (IsLeaf()) {
            if constexpr (IsQuaternion()) {
                return 4_c;
            } else {
                return hana::llong_c<_Size>;
            }
        } else {
            return hana::fold(hana::values(DefaultMap()), 0_c, [](auto size, auto value) {
                if constexpr (is_std_array_v<decltype(value)>) {
                    if constexpr (value.empty()) {
                        return size;
                    } else {
                        return size + hana::llong_c<static_cast<long long>(value.size())> *
                                          value.front().SizeC();
                    }
                } else {
                    return size + value.SizeC();
                }
            });
        }
    }

    template <typename _N, typename... _Args>
    constexpr decltype(auto) GetImpl0(_N name, _Args... args) const {
        if constexpr (static_cast<bool>(sizeof...(args))) {
            return BypassImpl(name).value().get()(args...);
        } else {
            return BypassImpl(name).value().get();
        }
    }

    template <typename _N, typename... _Args>
    constexpr decltype(auto) GetImpl1(_N name, const index_t i, _Args... args) const {
        if constexpr (static_cast<bool>(sizeof...(args))) {
            return BypassImpl(name, i).value().get()(args...);
        } else {
            return BypassImpl(name, i).value().get();
        }
    }

    template <typename _V, typename... _Args>
    constexpr decltype(auto) GetImpl2(_V name,
                                      const index_t i1,
                                      const index_t i2,
                                      _Args... args) const {
        if constexpr (static_cast<bool>(sizeof...(args))) {
            return BypassImpl(name, i1, i2).value().get()(args...);
        } else {
            return BypassImpl(name, i1, i2).value().get();
        }
    }

    template <typename _N, typename... _Args>
    constexpr decltype(auto) GetImpl3(
        _N name, const index_t i1, const index_t i2, const index_t i3, _Args... args) const {
        if constexpr (static_cast<bool>(sizeof...(args))) {
            return BypassImpl(name, i1, i2, i3).value().get()(args...);
        } else {
            return BypassImpl(name, i1, i2, i3).value().get();
        }
    }

    template <typename _F>
    constexpr void ForEach(_F&& f) {
        std::forward<_F>(f)(*this);

        if constexpr (IsLeaf()) {
            return;
        } else {
            hana::for_each(hana::keys(_map), [&](const auto& key) {
                auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                    for (auto& var : value) {
                        var.ForEach(std::forward<_F>(f));
                    }
                } else {
                    value.ForEach(std::forward<_F>(f));
                }
            });
        }
    }

    template <typename _State, typename _F>
    auto FoldUnique(_State&& state, _F&& f) const {
        return hana::fold(hana::values(_map),
                          std::forward<_F>(f)(std::forward<_State>(state), *this),
                          [&](auto&& s1, const auto& value) {
                              if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                                  if (!value.empty()) {
                                      return value.front().FoldUnique(
                                          std::forward<decltype(s1)>(s1), std::forward<_F>(f));
                                  } else {
                                      Unreachable();
                                  }
                              } else {
                                  return value.FoldUnique(std::forward<decltype(s1)>(s1),
                                                          std::forward<_F>(f));
                              }
                          });
    }

    template <typename _F>
    constexpr void ForEachUnique(_F&& f) const {
        std::forward<_F>(f)(*this);
        hana::for_each(hana::values(_map), [&](const auto& value) {
            if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                if (!value.empty()) {
                    value.front().ForEach(std::forward<_F>(f));
                }
            } else {
                value.ForEach(std::forward<_F>(f));
            }
        });
    }

    template <typename _Set1, typename _Set2, typename... _Others>
    static constexpr auto Union(_Set1 s1, _Set2 s2, _Others... others) {
        return Union(hana::union_(s1, s2), others...);
    }

    template <typename _Set>
    static constexpr auto Union(_Set set) {
        return set;
    }

    template <typename _Proj = hana::id_t>
    auto Unique(_Proj&& proj = hana::id) const {
        if constexpr (IsLeaf()) {
            return hana::make_set(std::forward<_Proj>(proj)(Variable{}));
        } else {
            return hana::unpack(hana::values(_map), [&](auto... values) {
                auto l = [&](auto v) {
                    if constexpr (is_std_array_v<remove_cvref_t<decltype(v)>>) {
                        if constexpr (!decltype(v){}.empty()) {
                            return v.front().Unique(std::forward<_Proj>(proj));
                        } else {
                            Unreachable();
                        }
                    } else {
                        return v.Unique(std::forward<_Proj>(proj));
                    }
                };
                return Union(hana::make_set(std::forward<_Proj>(proj)(Variable{})), l(values)...);
            });
        }
    }

    template <typename _N>
    constexpr auto BypassImpl(_N name) const {
        if constexpr (Name() == name) {
            return hana::just(Ungar::cref(*this));
        } else if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                    return hana::nothing;
                } else {
                    return value.BypassImpl(name);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    template <typename _N>
    static constexpr auto IsParentVariableOf(_N name) {
        if constexpr (Name() == name) {
            return hana::true_c;
        } else if constexpr (IsLeaf()) {
            return hana::false_c;
        } else {
            auto candidates = hana::transform(hana::values(DefaultMap()), [=](auto value) {
                if constexpr (is_std_array_v<decltype(value)>) {
                    return hana::false_c;
                } else {
                    return value.IsParentVariableOf(name);
                }
            });
            static_assert(hana::count_if(candidates, hana::id) <= hana::size_c<1UL>,
                          "There is a unique sub-variable only if such a variable appears once.");
            return hana::is_just(hana::find(candidates, hana::true_c));
        }
    }

    template <typename _N>
    constexpr auto BypassImpl(_N name, const index_t i) const {
        if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                    if constexpr (remove_cvref_t<decltype(value)>::value_type::IsParentVariableOf(
                                      remove_cvref_t<decltype(name)>{})) {
                        return value[static_cast<size_t>(i)].BypassImpl(name);
                    } else {
                        return hana::nothing;
                    }
                } else {
                    return value.BypassImpl(name, i);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    template <typename _N>
    constexpr auto BypassImpl(_N name, const index_t i1, const index_t i2) const {
        if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                    if constexpr (remove_cvref_t<decltype(value)>::value_type::IsParentVariableOf(
                                      remove_cvref_t<decltype(name)>{}, decltype(i2){})) {
                        return value[static_cast<size_t>(i1)].BypassImpl(name, i2);
                    } else {
                        return hana::nothing;
                    }
                } else {
                    return value.BypassImpl(name, i1, i2);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    template <typename _N>
    constexpr auto BypassImpl(_N name, const index_t i1, const index_t i2, const index_t i3) const {
        if constexpr (IsLeaf()) {
            return hana::nothing;
        } else {
            auto candidates = hana::transform(hana::keys(_map), [&](const auto& key) {
                const auto& value = hana::at_key(_map, key);
                if constexpr (is_std_array_v<remove_cvref_t<decltype(value)>>) {
                    if constexpr (remove_cvref_t<decltype(value)>::value_type::IsParentVariableOf(
                                      remove_cvref_t<decltype(name)>{},
                                      decltype(i2){},
                                      decltype(i3){})) {
                        return value[static_cast<size_t>(i1)].BypassImpl(name, i2, i3);
                    } else {
                        return hana::nothing;
                    }
                } else {
                    return value.BypassImpl(name, i1, i2, i3);
                }
            });
            static_assert(hana::count_if(candidates, hana::is_just) <= hana::size_c<1UL>,
                          "To prevent ambiguous bypasses, a variable must appear only once as a "
                          "sub-variable.");
            return hana::flatten(hana::find_if(candidates, hana::is_just));
        }
    }

    template <typename _N, typename... _Integrals>
    static constexpr auto IsParentVariableOf(_N name, const index_t i, const _Integrals... is) {
        if constexpr (IsLeaf()) {
            return hana::false_c;
        } else {
            auto candidates = hana::transform(hana::values(DefaultMap()), [=](auto value) {
                if constexpr (is_std_array_v<decltype(value)>) {
                    if constexpr (std_array_empty_v<decltype(value)>) {
                        return hana::false_c;
                    } else {
                        return decltype(value)::value_type::IsParentVariableOf(name, is...);
                    }
                } else {
                    return value.IsParentVariableOf(name, i, is...);
                }
            });
            static_assert(hana::count_if(candidates, hana::id) <= hana::size_c<1UL>,
                          "There is a unique sub-variable only if such a variable appears once.");
            return hana::is_just(hana::find(candidates, hana::true_c));
        }
    }

    _Name _name;
    _Map _map;

    index_t _index;
};

template <typename _Name>
Variable(_Name name) -> Variable<_Name, decltype(hana::make_map()), 0_idx>;

template <typename _Name, typename _Size>
Variable(_Name name, _Size size) -> Variable<_Name, decltype(hana::make_map()), _Size::value>;

static_assert(Q != std::numeric_limits<index_t>::max() &&
                  Q_c.value != static_cast<long long>(std::numeric_limits<index_t>::max()),
              "The special value denoting the size of a unit quaternion must be different from the "
              "maximum value that index_t integers can take.");

template <typename _Count,
          typename _N,
          typename _M,
          index_t _S,
          std::enable_if_t<is_hana_integral_constant_v<_Count>, bool> = true>
constexpr auto operator*(_Count cnt, Variable<_N, _M, _S> var) {
    return VariableProductExpr{cnt, var};
}

template <typename _El1,
          typename _El2,
          std::enable_if_t<(is_variable_v<_El1> || is_variable_product_expr_v<_El1>)&&(
                               is_variable_v<_El2> || is_variable_product_expr_v<_El2>),
                           bool> = true>
constexpr auto operator,(_El1 el1, _El2 el2) {
    return hana::make_tuple(el1, el2);
}

template <typename _El,
          typename _Tuple,
          std::enable_if_t<(is_variable_v<_El> ||
                            is_variable_product_expr_v<_El>)&&is_hana_tuple_v<_Tuple>,
                           bool> = true>
constexpr auto operator,(_El el, _Tuple tuple) {
    return hana::prepend(tuple, el);
}

template <typename _Tuple,
          typename _El,
          std::enable_if_t<is_hana_tuple_v<_Tuple> &&
                               (is_variable_v<_El> || is_variable_product_expr_v<_El>),
                           bool> = true>
constexpr auto operator,(_Tuple tuple, _El el) {
    return hana::append(tuple, el);
}

namespace Internal {

template <index_t _SIZE = std::numeric_limits<index_t>::max()>
inline constexpr auto var_helper = [](auto name) {
    if constexpr (_SIZE == std::numeric_limits<index_t>::max()) {
        return Variable{name};
    } else {
        return Variable{name, hana::llong_c<static_cast<long long>(_SIZE)>};
    }
};

}

#define UNGAR_LEAF_VARIABLE(name, size)                                       \
    constexpr auto name = ::Ungar::Variable {                                 \
        BOOST_HANA_STRING(#name), hana::llong_c<static_cast<long long>(size)> \
    }
#define UNGAR_BRANCH_VARIABLE(name)           \
    constexpr auto name = ::Ungar::Variable { \
        BOOST_HANA_STRING(#name)              \
    }
#define _UNGAR_STRINGIZE_FIRST_ARG(first, ...) #first
#define _UNGAR_FIRST_ARG(first, ...) first
#define _UNGAR_SECOND_ARG(first, second, ...) second
#define UNGAR_VARIABLE(...)                                            \
    constexpr auto _UNGAR_FIRST_ARG(__VA_ARGS__, dummy) =              \
        ::Ungar::Internal::var_helper<_UNGAR_SECOND_ARG(               \
            __VA_ARGS__, std::numeric_limits<index_t>::max(), dummy)>( \
            BOOST_HANA_STRING(_UNGAR_STRINGIZE_FIRST_ARG(__VA_ARGS__, dummy)))

}  // namespace Ungar

#endif /* _UNGAR__VARIABLE_HPP_ */
