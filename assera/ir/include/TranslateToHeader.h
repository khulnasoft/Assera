////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

#include <iosfwd>
#include <string>
#include <vector>

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
    mlir::LogicalResult TranslateToHeader(mlir::ModuleOp, std::ostream& os);
    mlir::LogicalResult TranslateToHeader(mlir::ModuleOp, llvm::raw_ostream& os);
    mlir::LogicalResult TranslateToHeader(std::vector<mlir::ModuleOp>& modules, const std::string& libraryName, llvm::raw_ostream& os);
} // namespace ir
} // namespace assera
