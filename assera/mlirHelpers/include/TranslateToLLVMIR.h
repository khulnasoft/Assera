////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <llvm/IR/Module.h>

#include <memory>

namespace mlir
{
class MLIRContext;

template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace assera
{
namespace ir
{
    std::unique_ptr<llvm::Module> TranslateToLLVMIR(mlir::OwningOpRef<mlir::ModuleOp>& module, llvm::LLVMContext& context);
} // namespace mlirHelpers
} // namespace assera
