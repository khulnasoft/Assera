////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <transforms/include/util/SnapshotUtilities.h>

#include <memory>

// fwd decls
namespace mlir
{
class MLIRContext;
class Pass;
template <typename OpT>
class OperationPass;

class RewritePatternSet;

} // namespace mlir

namespace assera
{
namespace ir::value
{
    class ValueFuncOp;
}
namespace transforms::loopnest
{
    struct LoopNestToValueFuncOptions
    {
        assera::transforms::IntraPassSnapshotOptions snapshotOptions;
        bool printLoops = false;
        bool printVecOpDetails = false;
    };

    void populateLoopnestToValueFuncPatterns(mlir::RewritePatternSet& patterns);
    std::unique_ptr<mlir::OperationPass<assera::ir::value::ValueFuncOp>> createLoopNestToValueFuncPass(const LoopNestToValueFuncOptions& options);
    std::unique_ptr<mlir::OperationPass<assera::ir::value::ValueFuncOp>> createLoopNestToValueFuncPass();
} // namespace transforms::loopnest
} // namespace assera
