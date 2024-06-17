////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors:  Mason Remy
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AsseraPasses.h"

#include <value/include/MLIREmitterContext.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/STLExtras.h>

using namespace mlir;

namespace
{

static constexpr const char* kVulkanLaunch = "vulkanLaunch";

/// A pass to mark the vulkanLaunch function to emit a C wrapper
class EmitVulkanWrapperPass : public assera::transforms::EmitVulkanWrapperBase<EmitVulkanWrapperPass>
{
public:
    void runOnModule() override
    {
        auto moduleOp = getOperation();
        SymbolTable symbolTable = SymbolTable::getNearestSymbolTable(moduleOp);
        auto vulkanLaunchFuncOp = dyn_cast_or_null<mlir::FuncOp>(symbolTable.lookup(kVulkanLaunch));
        if (vulkanLaunchFuncOp)
        {
            OpBuilder builder(vulkanLaunchFuncOp);
            vulkanLaunchFuncOp->setAttr(assera::ir::CInterfaceAttrName, builder.getUnitAttr());
        }
    }
};

} // namespace

namespace assera::transforms::vulkan
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEmitVulkanWrapperPass()
{
    return std::make_unique<EmitVulkanWrapperPass>();
}
} // namespace assera::transforms::vulkan
