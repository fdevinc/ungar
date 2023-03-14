/******************************************************************************
 *
 * @file ungar/io/logging.hpp
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

#ifndef _UNGAR__IO__LOGGING_HPP_
#define _UNGAR__IO__LOGGING_HPP_

#ifndef UNGAR_RELEASE
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <boost/hana.hpp>

#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace Ungar {

class Logger {
  public:
    static constexpr const char* NAME = "ungar";

  private:
    Logger()                         = delete;
    Logger(const Logger&)            = delete;
    Logger(Logger&&)                 = delete;
    Logger& operator=(const Logger&) = delete;
    Logger& operator=(Logger&&)      = delete;

    static auto InitializeLogger() {
        static bool initialized = false;
        if (initialized) {
            throw std::logic_error("Can initialize Ungar's logger only once.");
        }

        auto logger = spdlog::stdout_color_mt(NAME);
#ifndef UNGAR_RELEASE
        logger->set_level(spdlog::level::trace);
#endif
        logger->set_pattern("[%H:%M:%S.%e] [%n] [%^%l%$] %v");

        initialized = true;
        return logger;
    }

  public:
    static auto& Get() {
        static auto logger = InitializeLogger();
        return logger;
    }
};

}  // namespace Ungar

#define SPDLOG_LOGGER_trace SPDLOG_LOGGER_TRACE
#define SPDLOG_LOGGER_debug SPDLOG_LOGGER_DEBUG
#define SPDLOG_LOGGER_info SPDLOG_LOGGER_INFO
#define SPDLOG_LOGGER_warn SPDLOG_LOGGER_WARN
#define SPDLOG_LOGGER_error SPDLOG_LOGGER_ERROR
#define SPDLOG_LOGGER_critical SPDLOG_LOGGER_CRITICAL
#define UNGAR_LOG(level, ...) SPDLOG_LOGGER_##level(::Ungar::Logger::Get(), __VA_ARGS__)

namespace fmt {

template <typename _HanaStruct>  // clang-format off
requires boost::hana::Struct<_HanaStruct>::value
struct  formatter<_HanaStruct> {  // clang-format on
    char presentation = 'v';

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it        = ctx.begin();
        const auto end = ctx.end();
        if (it != end && (*it == 'v' || *it == 'c')) {
            presentation = *(it++);
        }

        if (it != end && *it != '}') {
            throw format_error(
                "Invalid format: the options available are 'v' (\"verbose\") or 'c' "
                "(\"compact\").");
        }

        return it;
    }

    template <typename _FormatContext>
    auto format(const _HanaStruct& hanaStruct, _FormatContext& ctx) -> decltype(ctx.out()) {
        namespace hana = boost::hana;

        auto separator = [&](auto i) -> const char* {
            if constexpr (i == 0UL) {
                return presentation == 'v' ? "\t" : "";
            } else {
                return presentation == 'v' ? ",\n\t" : ", ";
            }
        };

        std::string buffer;
        hana::length(hanaStruct).times.with_index([&](auto i) {
            std::ignore = presentation == 'v' ? format_to(std::back_inserter(buffer),
                                                          "{}{} = {}",
                                                          separator(i),
                                                          hana::keys(hanaStruct)[i].c_str(),
                                                          hana::members(hanaStruct)[i])
                                              : format_to(std::back_inserter(buffer),
                                                          "{}{}",
                                                          separator(i),
                                                          hana::members(hanaStruct)[i]);
        });

        if (presentation == 'v') {
            return format_to(ctx.out(), "\n{{\n{}\n}}", buffer);
        } else {
            return format_to(ctx.out(), "{{ {} }}", buffer);
        }
    }
};

}  // namespace fmt

#endif /* _UNGAR__IO__LOGGING_HPP_ */
