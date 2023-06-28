/******************************************************************************
 *
 * @file ungar/autodiff/function.hpp
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

#ifndef _UNGAR__AUTODIFF__FUNCTION_HPP_
#define _UNGAR__AUTODIFF__FUNCTION_HPP_

#include <finitediff.hpp>

#include "ungar/autodiff/support/quaternion.hpp"
#include "ungar/io/logging.hpp"
#include "ungar/utils/passkey.hpp"
#include "ungar/utils/utils.hpp"

namespace Ungar {
namespace Autodiff {

class FunctionFactory;

class Function {  // clang-format on
  public:
    struct Blueprint {
        Blueprint(const ADFunction& functionImpl_,
                  const index_t independentVariableSize_,
                  const index_t parameterSize_,
                  std::string_view name_,
                  const EnabledDerivatives enabledDerivatives_ = EnabledDerivatives::ALL,
                  const std::filesystem::path& folder_         = UNGAR_CODEGEN_FOLDER)
            : independentVariableSize{independentVariableSize_},
              parameterSize{parameterSize_},
              dependentVariableSize{[&functionImpl_, independentVariableSize_, parameterSize_]() {
                  VectorXad dependentVariable;
                  functionImpl_(VectorXad::Ones(independentVariableSize_ + parameterSize_),
                                dependentVariable);
                  return dependentVariable.size();
              }()},
              name{name_},
              folder{folder_},
              enabledDerivatives{enabledDerivatives_},
              functionImpl{functionImpl_} {
        }

        index_t independentVariableSize;
        index_t parameterSize;
        index_t dependentVariableSize;

        std::string name;
        std::filesystem::path folder;

        EnabledDerivatives enabledDerivatives;

        ADFunction functionImpl;
    };

    Function(Passkey<FunctionFactory>,
             std::unique_ptr<CppAD::cg::DynamicLib<real_t>> dynamicLib,
             std::unique_ptr<CppAD::cg::GenericModel<real_t>> model,
             const index_t independentVariableSize,
             const index_t parameterSize,
             const index_t dependentVariableSize)
        : _dynamicLib{std::move(dynamicLib)},
          _model{std::move(model)},
          _jacobianInnerStarts{},
          _jacobianOuterIndices{},
          _hessianInnerStarts{},
          _hessianOuterIndices{},
          _jacobianData{},
          _jacobianMap{0_idx, 0_idx, 0_idx, nullptr, nullptr, nullptr},
          _hessianData{},
          _hessianMap{0_idx, 0_idx, 0_idx, nullptr, nullptr, nullptr},
          _independentVariableSize{independentVariableSize},
          _parameterSize{parameterSize},
          _dependentVariableSize{dependentVariableSize},
          _nnzJacobian{0UL},
          _nnzHessian{0UL} {
        if (_model->isJacobianSparsityAvailable()) {
            for (const auto& row : _model->JacobianSparsitySet()) {
                _nnzJacobian += row.size();
            }

            std::vector<size_t> rows{};
            std::vector<size_t> cols{};
            _model->JacobianSparsity(rows, cols);

            _jacobianOuterIndices.clear();
            _jacobianOuterIndices.reserve(_nnzJacobian);
            for (const auto col : cols) {
                _jacobianOuterIndices.emplace_back(static_cast<int>(col));
            }
            _jacobianInnerStarts.clear();
            _jacobianInnerStarts.reserve(_dependentVariableSize + 1UL);
            _jacobianInnerStarts.emplace_back(0);
            {
                int i = 0;
                for (const int j : enumerate(static_cast<int>(_dependentVariableSize))) {
                    if (i < static_cast<int>(_nnzJacobian)) {
                        while (i < static_cast<int>(_nnzJacobian) &&
                               static_cast<int>(rows[i]) == j) {
                            ++i;
                        }
                        _jacobianInnerStarts.emplace_back(i);
                    } else {
                        _jacobianInnerStarts.emplace_back(_jacobianInnerStarts.back());
                    }
                }
            }

            _jacobianData.resize(_nnzJacobian);
            new (&_jacobianMap) Eigen::Map<const Eigen::SparseMatrix<real_t, Eigen::RowMajor>>{
                _dependentVariableSize,
                _independentVariableSize,
                static_cast<index_t>(_nnzJacobian),
                _jacobianInnerStarts.data(),
                _jacobianOuterIndices.data(),
                _jacobianData.data()};
        }
        if (_model->isHessianSparsityAvailable()) {
            UNGAR_ASSERT_MSG(_dependentVariableSize == 1_idx,
                             "The Hessian is implemented only for scalar functions.");

            for (const auto& row : _model->HessianSparsitySet()) {
                _nnzHessian += row.size();
            }

            std::vector<size_t> rows{};
            std::vector<size_t> cols{};
            _model->HessianSparsity(0UL, rows, cols);

            _hessianOuterIndices.clear();
            _hessianOuterIndices.reserve(_nnzHessian);
            for (const auto col : cols) {
                _hessianOuterIndices.emplace_back(static_cast<int>(col));
            }
            _hessianInnerStarts.clear();
            _hessianInnerStarts.reserve(_independentVariableSize + 1UL);  // n rows
            _hessianInnerStarts.emplace_back(0);
            {
                int i = 0;
                for (const int j : enumerate(static_cast<int>(_independentVariableSize))) {
                    if (i < static_cast<int>(_nnzHessian)) {
                        while (i < static_cast<int>(_nnzHessian) &&
                               static_cast<int>(rows[i]) == j) {
                            ++i;
                        }
                        _hessianInnerStarts.emplace_back(i);
                    } else {
                        _hessianInnerStarts.emplace_back(_hessianInnerStarts.back());
                    }
                }
            }

            _hessianData.resize(_nnzHessian);
            new (&_hessianMap) Eigen::Map<const Eigen::SparseMatrix<real_t, Eigen::RowMajor>>{
                _independentVariableSize,
                _independentVariableSize,
                static_cast<index_t>(_nnzHessian),
                _hessianInnerStarts.data(),
                _hessianOuterIndices.data(),
                _hessianData.data()};
        }
    }

    Function(Function&& rhs) = default;
    Function& operator=(Function&& rhs) = default;

    template <
        typename _XP,
        typename _Y,
        std::enable_if_t<same_v<real_t, typename _XP::Scalar, typename _Y::Scalar>, bool> = true>
    void Evaluate(const Eigen::MatrixBase<_XP>& xp, Eigen::MatrixBase<_Y> const& y) const {
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);
        UNGAR_ASSERT(y.size() == _dependentVariableSize);

        _model->ForwardZero(
            CppAD::cg::ArrayView<const real_t>{xp.derived().data(), static_cast<size_t>(xp.size())},
            CppAD::cg::ArrayView<real_t>{y.const_cast_derived().data(),
                                         static_cast<size_t>(y.size())});
    }

    template <typename _XP,
              typename _Y,
              std::enable_if_t<contiguous_range_of_v<_Y, real_t> &&
                                   std::is_same_v<real_t, typename _XP::Scalar>,
                               bool> = true>  // clang-format off
    void Evaluate(const Eigen::MatrixBase<_XP>& xp, _Y&& y) const {  // clang-format on
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);
        UNGAR_ASSERT(static_cast<index_t>(nano::ranges::size(std::forward<_Y>(y))) ==
                     _dependentVariableSize);

        _model->ForwardZero(
            CppAD::cg::ArrayView<const real_t>{xp.derived().data(), static_cast<size_t>(xp.size())},
            CppAD::cg::ArrayView<real_t>{
                nano::ranges::data(std::forward<_Y>(y)),
                static_cast<size_t>(nano::ranges::size(std::forward<_Y>(y)))});
    }

    template <typename _XP,
              std::enable_if_t<std::is_same_v<typename _XP::Scalar, real_t>, bool> = true>
    VectorXr operator()(const Eigen::MatrixBase<_XP>& xp) const {
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);

        VectorXr y{_dependentVariableSize};
        Evaluate(xp, y);
        return y;
    }

    template <typename _XP>
    const Eigen::Map<const Eigen::SparseMatrix<real_t, Eigen::RowMajor>>& Jacobian(
        const Eigen::MatrixBase<_XP>& xp) const {
        UNGAR_ASSERT(ImplementsJacobian());
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);

        const size_t* rows;
        const size_t* cols;
        _model->SparseJacobian(
            CppAD::cg::ArrayView<const real_t>{xp.derived().data(), static_cast<size_t>(xp.size())},
            CppAD::cg::ArrayView<real_t>{_jacobianData},
            &rows,
            &cols);
        return _jacobianMap;
    }

    /**
     * @brief Returns an upper triangular view of the Hessian matrix for a given
     *        dependent variable.
     */
    template <typename _XP>
    const Eigen::Map<const Eigen::SparseMatrix<real_t, Eigen::RowMajor>>& Hessian(
        const index_t dependentVariableIndex, const Eigen::MatrixBase<_XP>& xp) const {
        UNGAR_ASSERT(ImplementsHessian());
        UNGAR_ASSERT_MSG(_dependentVariableSize == 1_idx,
                         "Although this method is meant for general multivariable functions, it is "
                         "currently implemented only for scalar functions.");
        UNGAR_ASSERT(dependentVariableIndex >= 0_idx &&
                     dependentVariableIndex < _dependentVariableSize);
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);

        const size_t* rows;
        const size_t* cols;
        const VectorXr mask = VectorXr::Unit(_dependentVariableSize, dependentVariableIndex);
        CppAD::cg::ArrayView<const real_t> maskArrayView{mask.data(),
                                                         static_cast<size_t>(mask.size())};
        _model->SparseHessian(
            CppAD::cg::ArrayView<const real_t>{xp.derived().data(), static_cast<size_t>(xp.size())},
            maskArrayView,
            CppAD::cg::ArrayView<real_t>{_hessianData},
            &rows,
            &cols);
        return _hessianMap;
    }

    /**
     * @brief Returns an upper triangular view of the Hessian matrix.
     *
     * @note This method can only be used for scalar functions.
     */
    template <typename _XP>
    const Eigen::Map<const Eigen::SparseMatrix<real_t, Eigen::RowMajor>>& Hessian(
        const Eigen::MatrixBase<_XP>& xp) const {
        UNGAR_ASSERT(ImplementsHessian());
        UNGAR_ASSERT(_dependentVariableSize == 1_idx);
        UNGAR_ASSERT(xp.size() == _independentVariableSize + _parameterSize);

        return Hessian(0_idx, xp);
    }

    template <typename _XP>
    [[nodiscard]] bool TestFunction(
        const Eigen::MatrixBase<_XP>& xp,
        const std::function<VectorXr(const VectorXr&)>& groundTruthFunction) const {
        UNGAR_ASSERT(ImplementsFunction());
        return Utils::CompareMatrices(
            (*this)(xp), "Autodiff function"sv, groundTruthFunction(xp), "Ground truth"sv);
    }

    template <typename _XP>
    [[nodiscard]] bool TestJacobian(
        const Eigen::MatrixBase<_XP>& xp,
        const fd::AccuracyOrder accuracyOrder = fd::AccuracyOrder::SECOND,
        const real_t epsilon                  = 1e-8) const {
        UNGAR_ASSERT(ImplementsJacobian());
        MatrixXr fdJacobian;
        const VectorXr p{xp.tail(_parameterSize)};
        fd::finite_jacobian(
            xp.head(_independentVariableSize),
            [&](const VectorXr& xFD) {
                return (*this)((VectorXr{xFD.size() + p.size()} << xFD, p).finished());
            },
            fdJacobian,
            accuracyOrder,
            epsilon);
        return Utils::CompareMatrices(
            Jacobian(xp).toDense(), "Autodiff Jacobian"sv, fdJacobian, "FD Jacobian"sv);
    }

    template <typename _XP>
    [[nodiscard]] bool TestHessian(
        const index_t dependentVariableIndex,
        const Eigen::MatrixBase<_XP>& xp,
        const fd::AccuracyOrder accuracyOrder = fd::AccuracyOrder::SECOND,
        const real_t epsilon                  = 1e-5) const {
        UNGAR_ASSERT(dependentVariableIndex >= 0_idx &&
                     dependentVariableIndex < _dependentVariableSize);
        MatrixXr fdHessian;
        const VectorXr p{xp.tail(_parameterSize)};
        fd::finite_hessian(
            xp.head(_independentVariableSize),
            [&](const VectorXr& xFD) {
                return (*this)(
                    (VectorXr{xFD.size() + p.size()} << xFD, p).finished())[dependentVariableIndex];
            },
            fdHessian,
            accuracyOrder,
            epsilon);
        return Utils::CompareMatrices(Hessian(dependentVariableIndex, xp).toDense(),
                                      "Autodiff Hessian"sv,
                                      fdHessian,
                                      "FD Hessian"sv);
    }

    /**
     * @note This method can only be used for scalar functions.
     */
    template <typename _XP>
    [[nodiscard]] bool TestHessian(
        const Eigen::MatrixBase<_XP>& xp,
        const fd::AccuracyOrder accuracyOrder = fd::AccuracyOrder::SECOND,
        const real_t epsilon                  = 1e-5) const {
        UNGAR_ASSERT(_dependentVariableSize == 1_idx);
        return TestHessian(0_idx, xp, accuracyOrder, epsilon);
    }

    bool ImplementsFunction() const {
        return _model->isForwardZeroAvailable();
    }

    bool ImplementsJacobian() const {
        return _model->isSparseJacobianAvailable();
    }

    bool ImplementsHessian() const {
        return _model->isSparseHessianAvailable();
    }

    index_t IndependentVariableSize() const {
        return _independentVariableSize;
    }

    index_t ParameterSize() const {
        return _parameterSize;
    }

    index_t DependentVariableSize() const {
        return _dependentVariableSize;
    }

  private:
    std::unique_ptr<CppAD::cg::DynamicLib<real_t>> _dynamicLib;
    std::unique_ptr<CppAD::cg::GenericModel<real_t>> _model;

    /// @note For both the Jacobian and the Hessian, we store maps to arrays of data
    ///       that are manipulated by the \c Autodiff::Function generated code.
    ///       Unfortunately, \a CppADCodeGen generates sparse matrices in row major
    ///       representation with outer indices sorted in decreasing order, while
    ///       \a Eigen sorts them in increasing order. This means that the maps
    ///       we create cannot be used reliably for sparse operations. However,
    ///       assigning them to column major sparse matrices results in well-defined
    ///       conversions.
    std::vector<int> _jacobianInnerStarts;
    std::vector<int> _jacobianOuterIndices;
    std::vector<int> _hessianInnerStarts;
    std::vector<int> _hessianOuterIndices;

    mutable std::vector<real_t> _jacobianData;
    Eigen::Map<const Eigen::SparseMatrix<real_t, Eigen::RowMajor>> _jacobianMap;
    mutable std::vector<real_t> _hessianData;
    Eigen::Map<const Eigen::SparseMatrix<real_t, Eigen::RowMajor>> _hessianMap;

    index_t _independentVariableSize;
    index_t _parameterSize;
    index_t _dependentVariableSize;
    size_t _nnzJacobian;
    size_t _nnzHessian;
};

class FunctionFactory {
  private:
    class Worker {
      public:
        Worker(const Function::Blueprint& blueprint, std::vector<std::string> compilerFlags)
            : _blueprint{blueprint}, _compilerFlags{std::move(compilerFlags)} {
            SetUpFolders();
        }

        auto CreateModel() {
            CreateModelDirectories();

            UNGAR_LOG(info, "Creating internal model...");
            CreateModelsImpl(_blueprint.name + "_internal"s, _internalLibrary, false);
            UNGAR_LOG(info, "Creating model...");
            CreateModelsImpl(_blueprint.name, _library, true);

            UNGAR_LOG(info,
                      "Removing internal model folder {}...",
                      _internalLibrary.parent_path().parent_path());
            std::filesystem::remove_all(_internalLibrary.parent_path().parent_path());

            UNGAR_LOG(info, "Success.");

            return std::pair{std::unique_ptr<CppAD::cg::DynamicLib<real_t>>{_dynamicLib.release()},
                             std::unique_ptr<CppAD::cg::GenericModel<real_t>>{_model.release()}};
        }

        auto TryCreateModel() {
            if (IsLibraryAvailable()) {
                LoadModel();
                return std::pair{
                    std::unique_ptr<CppAD::cg::DynamicLib<real_t>>{_dynamicLib.release()},
                    std::unique_ptr<CppAD::cg::GenericModel<real_t>>{_model.release()}};
            } else {
                return CreateModel();
            }
        }

      private:
        void SetUpFolders() {
            _library = _blueprint.folder / _blueprint.name / "cppad_cg"s /
                       (_blueprint.name + "_lib"s +
                        CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION);
            _internalLibrary = _blueprint.folder / (_blueprint.name + "_internal/cppad_cg"s) /
                               (_blueprint.name + "_internal_lib"s +
                                CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION);

            _tmpFolder = Utils::TemporaryDirectoryPath("cppadcg_tmp"s);
        }

        void CreateModelDirectories() const {
            std::filesystem::create_directories(_library.parent_path());
            std::filesystem::create_directories(_internalLibrary.parent_path());
            std::filesystem::create_directories(_tmpFolder);
        }

        bool IsLibraryAvailable() const {
            return std::filesystem::exists(_library);
        }

        void CreateModelsImpl(const std::string& modelName,
                              const std::filesystem::path& library,
                              const bool trimInternalModelSparsity) {
            VectorXad xp =
                VectorXad::Ones(_blueprint.independentVariableSize + _blueprint.parameterSize);
            CppAD::Independent(xp);

            VectorXad y;

            _blueprint.functionImpl(xp, y);
            UNGAR_ASSERT(y.size() && y.size() == _blueprint.dependentVariableSize);

            ADFun adFun(xp, y);
            adFun.optimize();

            CppAD::cg::ModelCSourceGen<real_t> sourceGen(adFun, modelName);
            if (_blueprint.enabledDerivatives & EnabledDerivatives::JACOBIAN) {
                sourceGen.setCreateSparseJacobian(true);

                if (trimInternalModelSparsity) {
                    TrimParameterEntriesFromInternalModelJacobianSparsity(sourceGen);
                }
            }
            if (_blueprint.enabledDerivatives & EnabledDerivatives::HESSIAN) {
                sourceGen.setCreateSparseHessian(true);
                sourceGen.setCreateHessianSparsityByEquation(true);

                if (trimInternalModelSparsity) {
                    TrimParameterEntriesFromInternalModelHessianSparsity(sourceGen);
                }
            }

            const std::filesystem::path tmpLibrary{
                library.parent_path() / (library.stem().string() + _tmpFolder.stem().string() +
                                         library.extension().string())};

            CppAD::cg::ModelLibraryCSourceGen<real_t> libraryCSourceGen{sourceGen};
            CppAD::cg::GccCompiler<real_t> gccCompiler;
            CppAD::cg::DynamicModelLibraryProcessor<real_t> libraryProcessor{
                libraryCSourceGen, (tmpLibrary.parent_path() / tmpLibrary.stem()).string()};
            SetCompilerOptions(library.parent_path(), gccCompiler);

            UNGAR_LOG(info, "Compiling shared library {}...", tmpLibrary);
            _dynamicLib = libraryProcessor.createDynamicLibrary(gccCompiler);
            _model      = _dynamicLib->model(modelName);
            UNGAR_ASSERT(_dynamicLib);
            UNGAR_ASSERT(_model);

            UNGAR_LOG(info, "Renaming {} to {}...", tmpLibrary, library);
            std::filesystem::rename(tmpLibrary, library);
        }

        void LoadModel() {
            UNGAR_LOG(info, "Loading shared library {}...", _library);

            _dynamicLib.reset(new CppAD::cg::LinuxDynamicLib<real_t>(_library));
            _model = _dynamicLib->model(_blueprint.name);
            UNGAR_ASSERT(_dynamicLib);
            UNGAR_ASSERT(_model);

            UNGAR_LOG(info, "Success.");
        }

        void SetCompilerOptions(const std::filesystem::path& libraryFolder,
                                CppAD::cg::GccCompiler<real_t>& compiler) const {
            if (!_compilerFlags.empty()) {
                compiler.setCompileLibFlags(_compilerFlags);
                compiler.addCompileLibFlag("-shared");
                compiler.addCompileLibFlag("-rdynamic");
            }

            compiler.setTemporaryFolder(_tmpFolder);
            compiler.setSourcesFolder(libraryFolder);
            compiler.setSaveToDiskFirst(true);
        }

        void TrimParameterEntriesFromInternalModelJacobianSparsity(
            CppAD::cg::ModelCSourceGen<real_t>& sourceGen) const {
            if (!_model || _model->getName() != _blueprint.name + "_internal"s) {
                throw std::runtime_error(
                    "Cannot trim Jacobian sparsity for nonexistent internal model.");
            }

            const auto fullJacobianSparsity = _model->JacobianSparsitySet();

            SparsityPattern jacobianSparsity{static_cast<size_t>(_blueprint.dependentVariableSize)};
            for (const auto row : enumerate(_blueprint.dependentVariableSize)) {
                for (const auto col :
                     enumerate(_blueprint.independentVariableSize) |
                         nano::views::filter([&fullJacobianSparsity, row](const auto col) {
                             return fullJacobianSparsity[static_cast<size_t>(row)].find(
                                        static_cast<size_t>(col)) !=
                                    fullJacobianSparsity[static_cast<size_t>(row)].end();
                         })) {
                    jacobianSparsity[static_cast<size_t>(row)].insert(static_cast<size_t>(col));
                }
            }
            sourceGen.setCustomSparseJacobianElements(jacobianSparsity);
        }

        void TrimParameterEntriesFromInternalModelHessianSparsity(
            CppAD::cg::ModelCSourceGen<real_t>& sourceGen) const {
            if (!_model || _model->getName() != _blueprint.name + "_internal"s) {
                throw std::runtime_error(
                    "Cannot trim Hessian sparsity for nonexistent internal model.");
            }

            const auto fullHessianSparsity = _model->HessianSparsitySet();

            SparsityPattern hessianSparsity{
                static_cast<size_t>(_blueprint.independentVariableSize + _blueprint.parameterSize)};
            for (const auto row : enumerate(_blueprint.independentVariableSize)) {
                for (const auto col :
                     nano::views::iota(row, _blueprint.independentVariableSize) |
                         nano::views::filter([&fullHessianSparsity, row](const auto col) {
                             return fullHessianSparsity[static_cast<size_t>(row)].find(
                                        static_cast<size_t>(col)) !=
                                    fullHessianSparsity[static_cast<size_t>(row)].end();
                         })) {
                    hessianSparsity[static_cast<size_t>(row)].insert(static_cast<size_t>(col));
                }
            }
            sourceGen.setCustomSparseHessianElements(hessianSparsity);
        }

        Function::Blueprint _blueprint;

        std::unique_ptr<CppAD::cg::DynamicLib<real_t>> _dynamicLib;
        std::unique_ptr<CppAD::cg::GenericModel<real_t>> _model;
        std::vector<std::string> _compilerFlags;

        std::filesystem::path _library;
        std::filesystem::path _internalLibrary;

        std::filesystem::path _tmpFolder;
    };

  public:
    static Function Make(const Function::Blueprint& blueprint,
                         const bool recompileLibraries,
                         std::vector<std::string> compilerFlags) {
        Worker worker{blueprint, std::move(compilerFlags)};

        UNGAR_LOG(info, "{}", Utils::DASH_LINE_SEPARATOR);
        auto [dynamicLib, model] =
            recompileLibraries ? worker.CreateModel() : worker.TryCreateModel();

        return {UNGAR_PASSKEY,
                std::move(dynamicLib),
                std::move(model),
                blueprint.independentVariableSize,
                blueprint.parameterSize,
                blueprint.dependentVariableSize};
    }
};

inline Function MakeFunction(
    const Function::Blueprint& blueprint,
    const bool recompileLibraries          = false,
    std::vector<std::string> compilerFlags = {
        "-O3"s, "-g"s, "-march=native"s, "-mtune=native"s, "-ffast-math"s}) {
    return FunctionFactory::Make(blueprint, recompileLibraries, std::move(compilerFlags));
}

}  // namespace Autodiff
}  // namespace Ungar

#endif /* _UNGAR__AUTODIFF__FUNCTION_HPP_ */
