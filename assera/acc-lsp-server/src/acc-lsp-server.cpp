////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <ir/include/DialectRegistry.h> 
#include <transforms/include/AsseraPasses.h>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char** argv)
{
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
    assera::ir::GetDialectRegistry().appendTo(registry);
    assera::transforms::RegisterAllPasses();
    return failed(MlirLspServerMain(argc, argv, registry));
}