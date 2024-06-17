////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include "TranslateToLLVMIR.h"

#include <llvm/IR/Module.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Target/LLVMIR/Export.h>

#ifdef _MSC_VER
#pragma warning(enable : 4146)
#endif

namespace assera::ir
{
std::unique_ptr<llvm::Module> TranslateToLLVMIR(mlir::OwningOpRef<mlir::ModuleOp>& module, llvm::LLVMContext& context)
{
    return mlir::translateModuleToLLVMIR(*module, context);
}

} // namespace assera::ir
