

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class ModuleOp;
class Pass;
template <typename OpT>
class OperationPass;
} // namespace mlir

namespace assera::transforms::value
{

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFunctionPointerResolutionPass();
} // namespace assera::transforms::value
