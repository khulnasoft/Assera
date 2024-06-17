////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AsseraPasses.h"

#include <value/include/MLIREmitterContext.h>

#include <mlir/Dialect/GPU/Passes.h>

#include <memory>

class SerializeToHsacoPass : public assera::transforms::SerializeToHSACOBase<SerializeToHsacoPass>
{
public:
    void runOnOperation() override
    {
        // noop
    }
};

namespace assera::transforms
{
std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUModuleOp>> createSerializeToHSACOPass()
{
    return std::make_unique<SerializeToHsacoPass>();
}
} // namespace assera::transforms
