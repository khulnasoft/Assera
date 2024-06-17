////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

// fwd decls
namespace mlir
{
class FuncOp;

template <typename OpT>
class OperationPass;

class RewritePatternSet;
} // namespace mlir

namespace
{
struct ProfileRegions;
}

const char kTimerRegionTypeIdentifier[] = "timer_region_type";

enum class TimerRegionType
{
    enterRegion = 0,
    exitRegion = 1,
};

namespace assera::transforms::value
{
inline std::string GetSplitSizeAttrName()
{
    return "split_size";
}
} // namespace assera::transforms::value

namespace assera::transforms::value
{
void populateVectorizeValueOpPatterns(mlir::RewritePatternSet& patterns);
[[maybe_unused]] void populateValueToStandardPatterns(bool enableProfiling, ProfileRegions& profileRegions, mlir::RewritePatternSet& patterns);
void populateValueLaunchFuncPatterns(mlir::RewritePatternSet& patterns);
void populateValueModuleRewritePatterns(mlir::RewritePatternSet& patterns);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToStdPass(bool enableProfiling = false);
} // namespace assera::transforms::value
