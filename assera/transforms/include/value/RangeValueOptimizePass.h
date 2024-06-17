////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace assera::transforms::value
{
void populateRangeValueOptimizePatterns(mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::Pass> createRangeValueOptimizePass();
} // namespace assera::transforms::value
