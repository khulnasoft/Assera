////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class ModuleOp;

template <typename OpT>
class OperationPass;

} // namespace mlir

namespace assera::transforms
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEmitDebugFunctionPass();
} // namespace assera::transforms
