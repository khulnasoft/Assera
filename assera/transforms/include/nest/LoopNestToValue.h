////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace mlir
{
class RewritePatternSet;
} // namespace mlir

namespace assera::transforms
{
void populateRangeResolutionPatterns(mlir::RewritePatternSet& patterns);
void populateScheduleScaffoldingPatterns(bool printLoops, mlir::RewritePatternSet& patterns);
void populateScheduledOperationsPatterns(mlir::RewritePatternSet& patterns);
void populateScheduleToValueRewritePatterns(mlir::RewritePatternSet& patterns);
void populateScheduleToValuePatterns(mlir::RewritePatternSet& patterns);
void populateUnlinkSymbolicIndicesPatterns(mlir::RewritePatternSet& patterns);
void populateSymIndexCleanupPatterns(mlir::RewritePatternSet& patterns);
void populateGPUIndexMappingRewritePatterns(mlir::RewritePatternSet& patterns);
void populateLoopOptimizationPatterns(mlir::RewritePatternSet& patterns);
void populateLoopMergingPatterns(mlir::RewritePatternSet& patterns);
void populateLoopSimplificationPatterns(mlir::RewritePatternSet& patterns);
} // namespace assera::transforms
