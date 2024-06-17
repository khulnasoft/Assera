////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MLIR_TARGET_CPP_EMITTER_H
#define MLIR_TARGET_CPP_EMITTER_H

#include "mlir/Support/LogicalResult.h"

// Forward-declare LLVM classes
namespace llvm
{
class raw_ostream;
} // namespace llvm

namespace mlir
{
class Operation;

/// Convert the given model operation into C++ code.
LogicalResult translateModuleToCpp(Operation* m, raw_ostream& os, int indexBitwidth);

} // namespace mlir

#endif // MLIR_TARGET_CPP_EMITTER_H
