/******************************************************************************
 *
 * @file ungar/utils/macros/for_each.hpp
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

#ifndef _UNGAR__UTILS__MACROS__FOR_EACH_HPP_
#define _UNGAR__UTILS__MACROS__FOR_EACH_HPP_

#define _UNGAR_PARENS ()

#define _UNGAR_EXPAND(...) \
    _UNGAR_EXPAND4(_UNGAR_EXPAND4(_UNGAR_EXPAND4(_UNGAR_EXPAND4(__VA_ARGS__))))
#define _UNGAR_EXPAND4(...) \
    _UNGAR_EXPAND3(_UNGAR_EXPAND3(_UNGAR_EXPAND3(_UNGAR_EXPAND3(__VA_ARGS__))))
#define _UNGAR_EXPAND3(...) \
    _UNGAR_EXPAND2(_UNGAR_EXPAND2(_UNGAR_EXPAND2(_UNGAR_EXPAND2(__VA_ARGS__))))
#define _UNGAR_EXPAND2(...) \
    _UNGAR_EXPAND1(_UNGAR_EXPAND1(_UNGAR_EXPAND1(_UNGAR_EXPAND1(__VA_ARGS__))))
#define _UNGAR_EXPAND1(...) __VA_ARGS__

#define UNGAR_FOR_EACH(macro, ...) \
    __VA_OPT__(_UNGAR_EXPAND(_UNGAR_FOR_EACH_IMPL_1(macro, __VA_ARGS__)))
#define _UNGAR_FOR_EACH_IMPL_1(macro, first, ...) \
    macro(first) __VA_OPT__(_UNGAR_FOR_EACH_IMPL_2 _UNGAR_PARENS(macro, __VA_ARGS__))
#define _UNGAR_FOR_EACH_IMPL_2() _UNGAR_FOR_EACH_IMPL_1

#endif /* _UNGAR__UTILS__MACROS__FOR_EACH_HPP_ */
