////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa, Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class Pass;

class RewritePatternSet;
using RewritePatternSet = RewritePatternSet;
} // namespace mlir

namespace assera::transforms::value
{
void populateValueSimplifyPatterns(mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::Pass> createValueSimplifyPass();
} // namespace assera::transforms::value
