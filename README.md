# Ungar
[![Language](https://img.shields.io/badge/c%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![Language](https://img.shields.io/badge/c%2B%2B-17-orange.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/fdevinc/ungar/actions/workflows/ungar-ci.yml/badge.svg)](https://github.com/fdevinc/ungar/actions/workflows/ungar-ci.yml)

> Simplifying optimal control with metaprogramming
## Overview
```cpp
#include "ungar/variable_map.hpp"
using namespace Ungar;

int main() {
    /***************                     QUADROTOR MODEL                      ***************/
    // Define numeric invariants as integral constants.
    constexpr auto N = 30_c;  // Discrete time horizon.
    constexpr auto NUM_ROTORS = 4_c;

    // Define variables.
    UNGAR_VARIABLE(position, 3);                                    // := p
    UNGAR_VARIABLE(orientation, Q);                                 // := q
    UNGAR_VARIABLE(linear_velocity, 3);                             // := pDot
    UNGAR_VARIABLE(angular_velocity, 3);                            // := omega
    UNGAR_VARIABLE(x) <<=
        (position, orientation, linear_velocity, angular_velocity); // x := [p q pDot omega]
    UNGAR_VARIABLE(X) <<= (N + 1_c) * x;                            // X := [x0 x1 ... xN]

    UNGAR_VARIABLE(rotor_speed, 1);                                 // := r
    UNGAR_VARIABLE(u) <<= NUM_ROTORS * rotor_speed;                 // u := [r0 r1 r2 r3]
    UNGAR_VARIABLE(U) <<= N * u;                                    // U := [u0 u1 ... uN-1]

    UNGAR_VARIABLE(variables) <<= (X, U);

    // Access information about variables at compile time.
    static_assert(x.Size() == 13);
    static_assert(variables(x, 0).Index() == 0);         // [{x0} x1  x2  ...  xN  u0  ... ]
    static_assert(variables(x, 1).Index() == 13);        // [ x0 {x1} x2  ...  xN  u0  ... ]
    static_assert(variables(x, 2).Index() == 26);        // [ x0  x1 {x2} ...  xN  u0  ... ]
    static_assert(variables(u, 0).Index() == X.Size());  // [ x0  x1  x2  ...  xN {u0} ... ]

    // Create maps over contiguous arrays of data.
    auto vars = MakeVariableMap<real_t>(variables);          // [ X   u0    u1   ...  uN-1 ]
    vars.Get(u, 0).setZero();                                // [ X  0000   u1   ...  uN-1 ]
    vars.Get(u, 1).setOnes();                                // [ X  0000  1111  ...  uN-1 ]
    vars.Get(u, N - 1).setLinSpaced(2.0, 8.0);               // [ X  0000  1111  ...  2468 ]

    static_assert(std::same_as<decltype(vars.Get(rotor_speed, 0, 0)), real_t&>);
    static_assert(std::same_as<decltype(vars.Get(position, 0)), Eigen::Map<Vector3r>&>);
    static_assert(std::same_as<decltype(vars.Get(orientation, 0)), Eigen::Map<Quaternionr>&>);
    static_assert(std::same_as<decltype(vars.Get(x, 0)), Eigen::Map<Vector<real_t, 13>>&>);
}

```

### Getting Started
To get started with Ungar, explore the `example/mpc` folder, which contains thoroughly documented code for three nonlinear model predictive control (MPC) implementations: one for a [quadrotor](example/mpc/quadrotor.example.cpp), another for a [radio-controlled miniature car](example/mpc/rc_car.example.cpp), and the third for a [quadruped robot](example/mpc/quadruped.example.cpp). These examples will help you quickly grasp how to utilize the library for your own control systems.

## Using Ungar
Ungar is designed to be easily integrated in C++ projects using [CMake][]. Since it is a C++20 library, it requires compilers with C++20 support, such as [GCC 11][], [Clang 16][], [MSVC 19][], etc. For users who need C++17 support, a dedicated version of Ungar is available on the `cxx17` branch, which offers all the major features of the main version.

> :warning: **[GCC 12](https://gcc.gnu.org/gcc-12/) is not supported!** When attempting to create Ungar's variables, you may encounter [internal compiler errors (ICEs)](https://stackoverflow.com/a/12405680). Fortunately, this issue is absent in both [GCC 11](https://gcc.gnu.org/gcc-11/) and [GCC 13](https://gcc.gnu.org/gcc-13/), making them suitable alternatives.

### FetchContent
The recommended way to use Ungar is through [CMake][]'s [FetchContent][] module. To use Ungar in your CMake-based project, add the following lines to your project's _CMakeLists.txt_ file:
```CMake
include(FetchContent)

message(STATUS "Fetching ungar...")
FetchContent_Declare(
  ungar
  GIT_REPOSITORY https://github.com/fdevinc/ungar.git
  GIT_TAG main  # For C++17: cxx17
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

# Configure Ungar build options (all options are OFF by default).
set(UNGAR_BUILD_TESTS OFF)            # Turn ON to build Ungar's tests.
set(UNGAR_BUILD_EXAMPLES ON)          # Turn OFF to not build Ungar's examples.
set(UNGAR_ENABLE_LOGGING ON)          # Turn OFF to disable logging in Ungar.
set(UNGAR_ENABLE_AUTODIFF ON)         # Not supported on Windows.
set(UNGAR_ENABLE_OPTIMIZATION ON)     # Not supported on Windows.
FetchContent_MakeAvailable(ungar)
```
The above lines automatically handle the download of all the required dependencies if the corresponding targets are not defined. This simplifies the integration process and ensures that your project has everything it needs to use Ungar seamlessly.

Once you have added Ungar to your [CMake][] project using [FetchContent][], you can link your project to Ungar by adding the following line to your _CMakeLists.txt_:
```CMake
target_link_libraries(PROJECT ungar::ungar)
```

### Embedded
Alternatively, you can directly embed the library into an existing [CMake][] project. To this end, copy and paste the entire source tree in a subdirectory and call `add_subdirectory` in your _CMakeLists.txt_ file:
```CMake
add_subdirectory(ungar)
...
target_link_libraries(PROJECT ungar::ungar)
```
Once again, make sure to configure the various Ungar's build options before adding its directory.

## Project Organization
The project is organized in the following subdirectories:
- The [`example`](example) directory contains various examples showing the basic functionalities of the library.
- The [`external`](external) directory contains all the third-party libraries. [Eigen][] and
  [Boost.Hana][] are the only required dependencies: if their targets are not found, Ungar will use a bundled version
  of each. All remaining dependencies are optional and bring logging, automatic differentiation, and nonlinear optimization functionalities to the
  library. If enabled, they are downloaded using [CMake][].
- The [`include`](include) directory contains the library itself. The optional modules are located in the [`io`](include/ungar/io),
  [`autodiff`](include/ungar/autodiff), and [`optimization`](include/ungar/optimization) subfolders.
- The [`test`](test) directory contains the source code of all the unit tests.

## Development Status
Ungar is a project meant for long-term maintenance. While the core API of the library is stable, the optional modules will likely see breaking changes
over time. The current implementations will be enhanced to achieve faster compile times. Finally, new algorithms and types will be introduced to
make the creation of optimal control problems as straightforward as possible.

## Citing Ungar
If you are using Ungar for your work, we encourage you to cite the [related paper](https://arxiv.org/abs/2309.06783):
```
@INPROCEEDINGS{DeVincenti-IROS-23,
  author={De Vincenti, Flavio and Coros, Stelian},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Ungar - A C++ Framework for Real-Time Optimal Control Using Template Metaprogramming}, 
  year={2023},
  volume={},
  number={},
  pages={6297-6303},
  doi={10.1109/IROS55552.2023.10341365}}
```
Ungar was used to implement the centralized MPC controller presented in the [following publication](https://www.roboticsproceedings.org/rss19/p050.html):
```
@INPROCEEDINGS{DeVincenti-RSS-23, 
    AUTHOR    = {Flavio De Vincenti AND Stelian Coros}, 
    TITLE     = {{Centralized Model Predictive Control for Collaborative Loco-Manipulation}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2023}, 
    ADDRESS   = {Daegu, Republic of Korea}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2023.XIX.050} 
}
```

## License
Please see [LICENSE.md](LICENSE.md).

<!-- Links -->
[Eigen]: https://eigen.tuxfamily.org/index.php?title=Main_Page
[Boost.Hana]: https://github.com/boostorg/hana
[CMake]: http://www.cmake.org
[FetchContent]: https://cmake.org/cmake/help/latest/module/FetchContent.html
[GCC 11]: https://gcc.gnu.org/
[Clang 16]: https://clang.llvm.org/
[MSVC 19]: https://visualstudio.microsoft.com/vs/features/cplusplus/
