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
std::unique_ptr<mlir::Pass> createAsseraAffineLoopNormalizePass();
} // namespace assera::transforms::affine
