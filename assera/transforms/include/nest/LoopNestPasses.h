////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace mlir
{
class Pass;
class PassManager;
} // namespace mlir

namespace assera::transforms
{
namespace loopnest
{
    std::unique_ptr<mlir::Pass> createScheduledOperationsPass();
    std::unique_ptr<mlir::Pass> createScheduleToValuePass();
    std::unique_ptr<mlir::Pass> createLoopNestOptPass();

    void addLoopNestLoweringPasses(mlir::PassManager& pm);
    void addLoopNestStructureLoweringPasses(mlir::PassManager& pm);
    void addLoopNestFinalLoweringPasses(mlir::PassManager& pm);
    void addLoopNestCleanupLoweringPasses(mlir::PassManager& pm);
} // namespace loopnest
} // namespace assera::transforms
