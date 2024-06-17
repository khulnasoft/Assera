////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DIALECT_ARGO_AMDGPU_H_
#define DIALECT_ARGO_AMDGPU_H_

#include <mlir/IR/Types.h>
#include <string>

namespace mlir
{
namespace cpp_printer
{

    llvm::Optional<std::string> GetAMDMFMAOpName(const mlir::Type& aTy, const mlir::Type& bTy, const mlir::Type& cTy, const mlir::Type& resTy);

} // namespace cpp_printer
} // namespace mlir

#endif // DIALECT_ARGO_AMDGPU_H_
