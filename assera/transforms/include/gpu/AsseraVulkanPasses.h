////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace assera::transforms::vulkan
{
struct VulkanTimingOptions
{
    bool printTimings = false;
    int64_t warmupCount = 0;
    int64_t runCount = 1;

    static const VulkanTimingOptions& getDefaultOptions()
    {
        static VulkanTimingOptions options;
        return options;
    }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertVulkanLaunchFuncToVulkanCallsWithTimingPass(const VulkanTimingOptions& options = VulkanTimingOptions::getDefaultOptions());
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEmitVulkanWrapperPass();
} // namespace assera::transforms::vulkan
