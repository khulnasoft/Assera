////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ir/include/intrinsics/AsseraIntrinsicsDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;

#include "intrinsics/AsseraIntrinsicsDialect.cpp.inc"

namespace assera::ir::intrinsics
{

void AsseraIntrinsicsDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "intrinsics/AsseraIntrinsics.cpp.inc"
        >();
}

} // namespace assera::ir::intrinsics

#define GET_OP_CLASSES
#include "intrinsics/AsseraIntrinsics.cpp.inc"
