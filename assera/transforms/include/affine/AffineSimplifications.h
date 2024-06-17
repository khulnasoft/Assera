////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class Pass;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;
} // namespace mlir

namespace assera::transforms::affine
{
void populateAsseraAffineExprSimplificationPatterns(mlir::OwningRewritePatternList& patterns);
void populateAsseraAffineLoopSimplificationPatterns(mlir::OwningRewritePatternList& patterns);
std::unique_ptr<mlir::Pass> createAffineSimplificationPass();
} // namespace assera::transforms::affine
