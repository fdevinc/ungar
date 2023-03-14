/******************************************************************************
 *
 * @file ungar/assert.hpp
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

#ifndef _UNGAR__ASSERT_HPP_
#define _UNGAR__ASSERT_HPP_

#include <source_location>

#include "ungar/io/logging.hpp"

namespace Ungar {

inline constexpr long expect(long exp, long c) {
    if (exp == c) [[likely]] {
        return c;
    } else {
        return exp;
    }
}

inline void assertion_failed(
    char const* expr, const std::source_location location = std::source_location::current()) {
    UNGAR_LOG(error,
              "{}:{}: {}: Assertion `{}` failed.",
              location.file_name(),
              location.line(),
              location.function_name(),
              expr);
    std::abort();
}

inline void assertion_failed_msg(
    char const* expr,
    char const* msg,
    const std::source_location location = std::source_location::current()) {
    UNGAR_LOG(error,
              "{}:{}: {}: Assertion `{} && \"{}\"` failed.",
              location.file_name(),
              location.line(),
              location.function_name(),
              expr,
              msg);
    std::abort();
}

}  // namespace Ungar

#define UNGAR_LIKELY(x) ::Ungar::expect(x, 1)
#define UNGAR_UNLIKELY(x) ::Ungar::expect(x, 0)

#ifdef UNGAR_RELEASE
#define UNGAR_ASSERT(expr) ((void)0)
#define UNGAR_ASSERT_MSG(expr, msg) ((void)0)
#define UNGAR_ASSERT_EXPLICIT(expr) ((void)0)
#define UNGAR_ASSERT_EXPLICIT_MSG(expr, msg) ((void)0)
#else
#define UNGAR_ASSERT(expr) (UNGAR_LIKELY(!!(expr)) ? ((void)0) : ::Ungar::assertion_failed(#expr))
#define UNGAR_ASSERT_MSG(expr, msg) \
    (UNGAR_LIKELY(!!(expr)) ? ((void)0) : ::Ungar::assertion_failed_msg(#expr, msg))
#define UNGAR_ASSERT_EXPLICIT(expr) UNGAR_ASSERT(static_cast<bool>(expr))
#define UNGAR_ASSERT_EXPLICIT_MSG(expr, msg) UNGAR_ASSERT_MSG(static_cast<bool>(expr), msg)
#endif

#endif /* _UNGAR__ASSERT_HPP_ */
