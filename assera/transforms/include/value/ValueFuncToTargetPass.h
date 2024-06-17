////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class MLIRContext;
class ModuleOp;
class Pass;
template <typename OpT> class OperationPass;

class RewritePatternSet;
} // namespace mlir

namespace assera::transforms::value
{

void populateValueLambdaToFuncPatterns(mlir::MLIRContext* context, mlir::RewritePatternSet& patterns);
void populateValueFuncToTargetPatterns(mlir::MLIRContext* context, mlir::RewritePatternSet& patterns);
void populateValueLaunchFuncInlinerPatterns(mlir::MLIRContext*, mlir::RewritePatternSet&);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueFuncToTargetPass(const IntraPassSnapshotOptions& snapshotOptions);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueFuncToTargetPass();
} // namespace assera::transforms::value
