////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "assera/AsseraOps.h"
#include "assera/AsseraDialect.cpp.inc"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/StringSwitch.h>

namespace assera::ir::rc
{
void AsseraDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "assera/AsseraOps.cpp.inc"
        >();
}
} // namespace assera::ir::rc

using namespace llvm;
using namespace mlir;
using namespace assera::ir;
using namespace assera::ir::rc;

// TableGen'd op method definitions
#define GET_OP_CLASSES
#include "assera/AsseraOps.cpp.inc"
