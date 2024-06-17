////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>

#include <functional>
#include <optional>
#include <string>

namespace assera::value
{
class Value;
}

namespace assera
{
namespace ir
{
    mlir::ModuleOp ConvertToLLVM(
        mlir::ModuleOp module,
        std::function<void(mlir::PassManager& pm)> addStdPassesFn,
        std::function<void(mlir::PassManager& pm)> addLLVMPassesFn);

} // namespace mlirHelpers
} // namespace assera
