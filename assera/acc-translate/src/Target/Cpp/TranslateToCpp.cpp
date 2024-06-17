////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "transforms/include/AsseraPasses.h"

#include "TranslateToCpp.h"
#include "CppPrinter.h"

#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

using namespace llvm;

namespace mlir
{

LogicalResult translateModuleToCpp(Operation* m, raw_ostream& os, int indexBitwidth)
{
    cpp_printer::CppPrinter printer(os, indexBitwidth);
#if 1
    auto context = m->getContext();

    PassManager pm(context);
    auto& optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(memref::createFoldSubViewOpsPass());
    optPM.addPass(createAffineScalarReplacementPass());
    pm.addPass(assera::transforms::value::createRangeValueOptimizePass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    if (failed(pm.run(m)))
    {
        return failure();
    }
#endif
    return printer.process(m);
}

} // namespace mlir
