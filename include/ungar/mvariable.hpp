/******************************************************************************
 *
 * @file ungar/mvariable.hpp
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

#ifndef _UNGAR__MVARIABLE_HPP_
#define _UNGAR__MVARIABLE_HPP_

#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/pop_front.hpp>
#include <boost/preprocessor/seq/size.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "ungar/utils/macros/for_each.hpp"
#include "ungar/utils/utils.hpp"

namespace Ungar {

enum class MVariableKind { LEAF, BRANCH, ARRAY };
enum class MVariableSpace { EUCLIDEAN, UNIT_QUATERNION };

template <typename _Var, size_t _N>
constexpr auto make_mvariable_array(index_t index) {
    return [index]<size_t... _IS>(std::index_sequence<_IS...>) {
        return std::array<_Var, _N>{(index + _Var::Size() * static_cast<index_t>(_IS))...};
    }
    (std::make_index_sequence<_N>());
}

#define UNGAR_LEAF_MVARIABLE(name, size)                                                        \
    inline constexpr struct name##_t {                                                          \
      private:                                                                                  \
        static constexpr auto _name = #name;                                                    \
        static constexpr auto _size =                                                           \
            ::Ungar::hana::if_(static_cast<::Ungar::index_t>(size) == ::Ungar::Utils::Q,        \
                               static_cast<::Ungar::index_t>(4),                                \
                               static_cast<::Ungar::index_t>(size));                            \
        static constexpr auto _kind = ::Ungar::MVariableKind::LEAF;                             \
        static constexpr auto _space =                                                          \
            ::Ungar::hana::if_(static_cast<::Ungar::index_t>(size) == ::Ungar::Utils::Q,        \
                               ::Ungar::MVariableSpace::UNIT_QUATERNION,                        \
                               ::Ungar::MVariableSpace::EUCLIDEAN);                             \
        ::Ungar::index_t _index;                                                                \
                                                                                                \
      public:                                                                                   \
        static constexpr const char* Name() {                                                   \
            return _name;                                                                       \
        }                                                                                       \
                                                                                                \
        static constexpr ::Ungar::index_t Size() {                                              \
            return _size;                                                                       \
        }                                                                                       \
                                                                                                \
        static constexpr ::Ungar::MVariableSpace Space() {                                      \
            return _space;                                                                      \
        }                                                                                       \
                                                                                                \
        constexpr ::Ungar::index_t Index() const {                                              \
            return _index;                                                                      \
        }                                                                                       \
                                                                                                \
        constexpr name##_t(const ::Ungar::index_t _index) : _index{_index} {                    \
        }                                                                                       \
                                                                                                \
        constexpr const auto& Get(auto _var, auto... _args) const {                             \
            if constexpr (sizeof...(_args)) {                                                   \
                return GetOpt(_var)->get().Get(_args...);                                       \
            } else {                                                                            \
                return GetOpt(_var).value().get();                                              \
            }                                                                                   \
        }                                                                                       \
                                                                                                \
        constexpr const auto& Get(auto _var,                                                    \
                                  std::convertible_to<::Ungar::index_t> auto i,                 \
                                  auto... _args) const {                                        \
            if constexpr (sizeof...(_args)) {                                                   \
                return GetOpt(_var, i)->get().Get(_args...);                                    \
            } else {                                                                            \
                return GetOpt(_var, i).value().get();                                           \
            }                                                                                   \
        }                                                                                       \
                                                                                                \
        constexpr const auto& Get(auto _var,                                                    \
                                  std::convertible_to<::Ungar::index_t> auto _i1,               \
                                  std::convertible_to<::Ungar::index_t> auto _i2,               \
                                  auto... _args) const {                                        \
            if constexpr (sizeof...(_args)) {                                                   \
                return GetOpt(_var, _i1, _i2)->get().Get(_args...);                             \
            } else {                                                                            \
                return GetOpt(_var, _i1, _i2).value().get();                                    \
            }                                                                                   \
        }                                                                                       \
                                                                                                \
        constexpr const auto& Get(auto _var,                                                    \
                                  std::convertible_to<::Ungar::index_t> auto _i1,               \
                                  std::convertible_to<::Ungar::index_t> auto _i2,               \
                                  std::convertible_to<::Ungar::index_t> auto _i3,               \
                                  auto... _args) const {                                        \
            if constexpr (sizeof...(_args)) {                                                   \
                return GetOpt(_var, _i1, _i2, _i3)->get().Get(_args...);                        \
            } else {                                                                            \
                return GetOpt(_var, _i1, _i2, _i3).value().get();                               \
            }                                                                                   \
        }                                                                                       \
                                                                                                \
        constexpr decltype(auto) GetOpt(auto _var) const {                                      \
            if constexpr (std::same_as<decltype(_var), name##_t>) {                             \
                return ::Ungar::hana::just(std::cref(*this));                                   \
            } else {                                                                            \
                return ::Ungar::hana::nothing;                                                  \
            }                                                                                   \
        }                                                                                       \
                                                                                                \
        constexpr decltype(auto) GetOpt(auto _var,                                              \
                                        std::convertible_to<::Ungar::index_t> auto _i) const {  \
            return ::Ungar::hana::nothing;                                                      \
        }                                                                                       \
                                                                                                \
        constexpr decltype(auto) GetOpt(auto _var,                                              \
                                        std::convertible_to<::Ungar::index_t> auto _i1,         \
                                        std::convertible_to<::Ungar::index_t> auto _i2) const { \
            return ::Ungar::hana::nothing;                                                      \
        }                                                                                       \
                                                                                                \
        constexpr decltype(auto) GetOpt(auto _var,                                              \
                                        std::convertible_to<::Ungar::index_t> auto _i1,         \
                                        std::convertible_to<::Ungar::index_t> auto _i2,         \
                                        std::convertible_to<::Ungar::index_t> auto _i3) const { \
            return ::Ungar::hana::nothing;                                                      \
        }                                                                                       \
                                                                                                \
        constexpr decltype(auto) GetOpt(auto... _args) const {                                  \
            return ::Ungar::hana::nothing;                                                      \
        }                                                                                       \
    } name {                                                                                    \
        static_cast<::Ungar::index_t>(0)                                                        \
    }

#define UNGAR_MVARIABLE_ARRAY(name, var, size)                                                     \
    inline constexpr struct name##_t {                                                             \
      private:                                                                                     \
        static constexpr auto _name  = #name;                                                      \
        static constexpr auto _size  = static_cast<::Ungar::index_t>(size) * var##_t::Size();      \
        static constexpr auto _kind  = ::Ungar::MVariableKind::ARRAY;                              \
        static constexpr auto _space = ::Ungar::MVariableSpace::EUCLIDEAN;                         \
        ::Ungar::index_t _index;                                                                   \
        ::std::array<var##_t, static_cast<::std::size_t>(size)> _data;                             \
                                                                                                   \
      public:                                                                                      \
        static constexpr const char* Name() {                                                      \
            return _name;                                                                          \
        }                                                                                          \
                                                                                                   \
        static constexpr ::Ungar::index_t Size() {                                                 \
            return _size;                                                                          \
        }                                                                                          \
                                                                                                   \
        static constexpr ::Ungar::MVariableSpace Space() {                                         \
            return _space;                                                                         \
        }                                                                                          \
                                                                                                   \
        constexpr ::Ungar::index_t Index() const {                                                 \
            return _index;                                                                         \
        }                                                                                          \
                                                                                                   \
        constexpr name##_t(const ::Ungar::index_t _index)                                          \
            : _index{_index},                                                                      \
              _data{make_mvariable_array<var##_t, static_cast<::std::size_t>(size)>(_index)} {     \
        }                                                                                          \
                                                                                                   \
        constexpr decltype(auto) operator[](std::convertible_to<::Ungar::index_t> auto i) const {  \
            return _data[static_cast<::std::size_t>(i)];                                           \
        }                                                                                          \
                                                                                                   \
        constexpr const auto& Get(auto _var, auto... _args) const {                                \
            if constexpr (sizeof...(_args)) {                                                      \
                return GetOpt(_var)->get().Get(_args...);                                          \
            } else {                                                                               \
                return GetOpt(_var).value().get();                                                 \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        constexpr const auto& Get(auto _var,                                                       \
                                  std::convertible_to<::Ungar::index_t> auto i,                    \
                                  auto... _args) const {                                           \
            if constexpr (sizeof...(_args)) {                                                      \
                return GetOpt(_var, i)->get().Get(_args...);                                       \
            } else {                                                                               \
                return GetOpt(_var, i).value().get();                                              \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        constexpr const auto& Get(auto _var,                                                       \
                                  std::convertible_to<::Ungar::index_t> auto _i1,                  \
                                  std::convertible_to<::Ungar::index_t> auto _i2,                  \
                                  auto... _args) const {                                           \
            if constexpr (sizeof...(_args)) {                                                      \
                return GetOpt(_var, _i1, _i2)->get().Get(_args...);                                \
            } else {                                                                               \
                return GetOpt(_var, _i1, _i2).value().get();                                       \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        constexpr const auto& Get(auto _var,                                                       \
                                  std::convertible_to<::Ungar::index_t> auto _i1,                  \
                                  std::convertible_to<::Ungar::index_t> auto _i2,                  \
                                  std::convertible_to<::Ungar::index_t> auto _i3,                  \
                                  auto... _args) const {                                           \
            if constexpr (sizeof...(_args)) {                                                      \
                return GetOpt(_var, _i1, _i2, _i3)->get().Get(_args...);                           \
            } else {                                                                               \
                return GetOpt(_var, _i1, _i2, _i3).value().get();                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        constexpr decltype(auto) GetOpt(auto _var) const {                                         \
            if constexpr (std::same_as<decltype(_var), name##_t>) {                                \
                return ::Ungar::hana::just(std::cref(*this));                                      \
            } else {                                                                               \
                return ::Ungar::hana::nothing;                                                     \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        constexpr decltype(auto) GetOpt(auto _var,                                                 \
                                        std::convertible_to<::Ungar::index_t> auto _i) const {     \
            if constexpr (std::same_as<std::remove_cvref_t<decltype(_data.front().GetOpt(_var))>,  \
                                       ::Ungar::hana::optional<>>) {                               \
                return ::Ungar::hana::nothing;                                                     \
            } else {                                                                               \
                return _data[static_cast<::std::size_t>(_i)].GetOpt(_var);                         \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        constexpr decltype(auto) GetOpt(auto _var,                                                 \
                                        std::convertible_to<::Ungar::index_t> auto _i1,            \
                                        std::convertible_to<::Ungar::index_t> auto _i2) const {    \
            if constexpr (std::same_as<                                                            \
                              std::remove_cvref_t<decltype(_data.front().GetOpt(_var, _i2))>,      \
                              ::Ungar::hana::optional<>>) {                                        \
                return ::Ungar::hana::nothing;                                                     \
            } else {                                                                               \
                return _data[static_cast<::std::size_t>(_i1)].GetOpt(_var, _i2);                   \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        constexpr decltype(auto) GetOpt(auto _var,                                                 \
                                        std::convertible_to<::Ungar::index_t> auto _i1,            \
                                        std::convertible_to<::Ungar::index_t> auto _i2,            \
                                        std::convertible_to<::Ungar::index_t> auto _i3) const {    \
            if constexpr (std::same_as<                                                            \
                              std::remove_cvref_t<decltype(_data.front().GetOpt(_var, _i2, _i3))>, \
                              ::Ungar::hana::optional<>>) {                                        \
                return ::Ungar::hana::nothing;                                                     \
            } else {                                                                               \
                return _data[static_cast<::std::size_t>(_i1)].GetOpt(_var, _i2, _i3);              \
            }                                                                                      \
        }                                                                                          \
    } name {                                                                                       \
        static_cast<::Ungar::index_t>(0)                                                           \
    }

#define _UNGAR_BRANCH_MVARIABLE_HELPER_1_PRED(r, state) \
    BOOST_PP_NOT_EQUAL(BOOST_PP_SEQ_SIZE(state), 1)
#define _UNGAR_BRANCH_MVARIABLE_HELPER_1_OP(r, state) BOOST_PP_SEQ_POP_FRONT(state)
#define _UNGAR_BRANCH_MVARIABLE_HELPER_1(r, state)                               \
    , BOOST_PP_SEQ_ELEM(1, state) {                                              \
        BOOST_PP_SEQ_ELEM(0, state).Index() + BOOST_PP_SEQ_ELEM(0, state).Size() \
    }
#define _UNGAR_BRANCH_MVARIABLE_HELPER_2(name) +name##_t::Size()
#define _UNGAR_BRANCH_MVARIABLE_HELPER_3(name) , name.GetOpt(_var)
#define _UNGAR_BRANCH_MVARIABLE_HELPER_4(name) , name.GetOpt(_var, _i)
#define _UNGAR_BRANCH_MVARIABLE_HELPER_5(name) , name.GetOpt(_var, _i1, _i2)
#define _UNGAR_BRANCH_MVARIABLE_HELPER_6(name) , name.GetOpt(_var, _i1, _i2, _i3)
#define _UNGAR_BRANCH_MVARIABLE_HELPER_7(name) name##_t name;
#define UNGAR_BRANCH_MVARIABLE(name, firstSubVariableName, ...)                                   \
    inline constexpr struct name##_t {                                                            \
      private:                                                                                    \
        static constexpr auto _name = #name;                                                      \
        static constexpr auto _size = firstSubVariableName##_t::Size()                            \
            UNGAR_FOR_EACH(_UNGAR_BRANCH_MVARIABLE_HELPER_2, __VA_ARGS__);                        \
        static constexpr auto _kind  = ::Ungar::MVariableKind::BRANCH;                            \
        static constexpr auto _space = ::Ungar::MVariableSpace::EUCLIDEAN;                        \
        ::Ungar::index_t _index;                                                                  \
                                                                                                  \
      public:                                                                                     \
        UNGAR_FOR_EACH(_UNGAR_BRANCH_MVARIABLE_HELPER_7, firstSubVariableName, __VA_ARGS__)       \
                                                                                                  \
        constexpr name##_t(const ::Ungar::index_t _index)                                         \
            : _index{_index},                                                                     \
              firstSubVariableName{_index} BOOST_PP_FOR(                                          \
                  (firstSubVariableName)BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__),                    \
                  _UNGAR_BRANCH_MVARIABLE_HELPER_1_PRED,                                          \
                  _UNGAR_BRANCH_MVARIABLE_HELPER_1_OP,                                            \
                  _UNGAR_BRANCH_MVARIABLE_HELPER_1) {                                             \
        }                                                                                         \
                                                                                                  \
        static constexpr const char* Name() {                                                     \
            return _name;                                                                         \
        }                                                                                         \
                                                                                                  \
        static constexpr ::Ungar::index_t Size() {                                                \
            return _size;                                                                         \
        }                                                                                         \
                                                                                                  \
        static constexpr ::Ungar::MVariableSpace Space() {                                        \
            return _space;                                                                        \
        }                                                                                         \
                                                                                                  \
        constexpr ::Ungar::index_t Index() const {                                                \
            return _index;                                                                        \
        }                                                                                         \
                                                                                                  \
        constexpr const auto& Get(auto _var, auto... _args) const {                               \
            if constexpr (sizeof...(_args)) {                                                     \
                return GetOpt(_var)->get().Get(_args...);                                         \
            } else {                                                                              \
                return GetOpt(_var).value().get();                                                \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        constexpr const auto& Get(auto _var,                                                      \
                                  std::convertible_to<::Ungar::index_t> auto i,                   \
                                  auto... _args) const {                                          \
            if constexpr (sizeof...(_args)) {                                                     \
                return GetOpt(_var, i)->get().Get(_args...);                                      \
            } else {                                                                              \
                return GetOpt(_var, i).value().get();                                             \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        constexpr const auto& Get(auto _var,                                                      \
                                  std::convertible_to<::Ungar::index_t> auto _i1,                 \
                                  std::convertible_to<::Ungar::index_t> auto _i2,                 \
                                  auto... _args) const {                                          \
            if constexpr (sizeof...(_args)) {                                                     \
                return GetOpt(_var, _i1, _i2)->get().Get(_args...);                               \
            } else {                                                                              \
                return GetOpt(_var, _i1, _i2).value().get();                                      \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        constexpr const auto& Get(auto _var,                                                      \
                                  std::convertible_to<::Ungar::index_t> auto _i1,                 \
                                  std::convertible_to<::Ungar::index_t> auto _i2,                 \
                                  std::convertible_to<::Ungar::index_t> auto _i3,                 \
                                  auto... _args) const {                                          \
            if constexpr (sizeof...(_args)) {                                                     \
                return GetOpt(_var, _i1, _i2, _i3)->get().Get(_args...);                          \
            } else {                                                                              \
                return GetOpt(_var, _i1, _i2, _i3).value().get();                                 \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        constexpr decltype(auto) GetOpt(auto _var) const {                                        \
            if constexpr (std::same_as<decltype(_var), name##_t>) {                               \
                return ::Ungar::hana::just(std::cref(*this));                                     \
            } else {                                                                              \
                auto candidates = ::Ungar::hana::make_tuple(firstSubVariableName.GetOpt(          \
                    _var) UNGAR_FOR_EACH(_UNGAR_BRANCH_MVARIABLE_HELPER_3, __VA_ARGS__));         \
                static_assert(hana::count_if(candidates, ::Ungar::hana::is_just) <=               \
                                  ::Ungar::hana::size_c<1UL>,                                     \
                              "To prevent ambiguous bypasses, a variable must appear at most "    \
                              "once as a sub-variable.");                                         \
                return ::Ungar::hana::flatten(hana::find_if(candidates, ::Ungar::hana::is_just)); \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        constexpr decltype(auto) GetOpt(auto _var,                                                \
                                        std::convertible_to<::Ungar::index_t> auto _i) const {    \
            auto candidates = ::Ungar::hana::make_tuple(firstSubVariableName.GetOpt(              \
                _var, _i) UNGAR_FOR_EACH(_UNGAR_BRANCH_MVARIABLE_HELPER_4, __VA_ARGS__));         \
            static_assert(                                                                        \
                hana::count_if(candidates, ::Ungar::hana::is_just) <= ::Ungar::hana::size_c<1UL>, \
                "To prevent ambiguous bypasses, a variable must appear at most once as "          \
                "a sub-variable.");                                                               \
            return ::Ungar::hana::flatten(hana::find_if(candidates, ::Ungar::hana::is_just));     \
        }                                                                                         \
                                                                                                  \
        constexpr decltype(auto) GetOpt(auto _var,                                                \
                                        std::convertible_to<::Ungar::index_t> auto _i1,           \
                                        std::convertible_to<::Ungar::index_t> auto _i2) const {   \
            auto candidates = ::Ungar::hana::make_tuple(firstSubVariableName.GetOpt(              \
                _var, _i1, _i2) UNGAR_FOR_EACH(_UNGAR_BRANCH_MVARIABLE_HELPER_5, __VA_ARGS__));   \
            static_assert(                                                                        \
                hana::count_if(candidates, ::Ungar::hana::is_just) <= ::Ungar::hana::size_c<1UL>, \
                "To prevent ambiguous bypasses, a variable must appear at most once as "          \
                "a sub-variable.");                                                               \
            return ::Ungar::hana::flatten(hana::find_if(candidates, ::Ungar::hana::is_just));     \
        }                                                                                         \
                                                                                                  \
        constexpr decltype(auto) GetOpt(auto _var,                                                \
                                        std::convertible_to<::Ungar::index_t> auto _i1,           \
                                        std::convertible_to<::Ungar::index_t> auto _i2,           \
                                        std::convertible_to<::Ungar::index_t> auto _i3) const {   \
            auto candidates = ::Ungar::hana::make_tuple(                                          \
                firstSubVariableName.GetOpt(_var, _i1, _i2, _i3)                                  \
                    UNGAR_FOR_EACH(_UNGAR_BRANCH_MVARIABLE_HELPER_6, __VA_ARGS__));               \
            static_assert(                                                                        \
                hana::count_if(candidates, ::Ungar::hana::is_just) <= ::Ungar::hana::size_c<1UL>, \
                "To prevent ambiguous bypasses, a variable must appear at most once as "          \
                "a sub-variable.");                                                               \
            return ::Ungar::hana::flatten(hana::find_if(candidates, ::Ungar::hana::is_just));     \
        }                                                                                         \
    } name {                                                                                      \
        static_cast<::Ungar::index_t>(0)                                                          \
    }

}  // namespace Ungar

#endif /* _UNGAR__MVARIABLE_HPP_ */
