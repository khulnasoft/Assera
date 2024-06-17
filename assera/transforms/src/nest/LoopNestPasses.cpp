////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/LoopNestPasses.h"
#include "AsseraPasses.h"
#include "nest/LoopNestToValue.h"

#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

using namespace assera::ir;
using namespace assera::ir::loopnest;
namespace v = assera::ir::value;
namespace xp = assera::ir::executionPlan;
using namespace assera::transforms;
using namespace mlir;

namespace
{

struct ScheduledOperationsLoweringPass : public ConvertScheduledOperationsBase<ScheduledOperationsLoweringPass>
{
    void runOnOperation() final;
};

struct ScheduleToValueLoweringPass : public ConvertScheduleToValueBase<ScheduleToValueLoweringPass>
{
    void runOnOperation() final;
};

struct LoopNestOptPass : public ConvertLoopNestOptBase<LoopNestOptPass>
{
    void runOnOperation() final;
};

} // end anonymous namespace.

void ScheduledOperationsLoweringPass::runOnOperation()
{
    {
        RewritePatternSet patterns(&getContext());
        populateRangeResolutionPatterns(patterns);
        (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    {
        RewritePatternSet patterns(&getContext());
        populateScheduleScaffoldingPatterns(false, patterns);
        (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    ConversionTarget target(getContext());

    target.addLegalDialect<LoopNestDialect,
                           mlir::AffineDialect,
                           mlir::arith::ArithmeticDialect,
                           mlir::math::MathDialect,
                           mlir::memref::MemRefDialect,
                           mlir::StandardOpsDialect,
                           v::ValueDialect,
                           xp::ExecutionPlanDialect>();

    target.addDynamicallyLegalOp<ScheduleOp>([](Operation* op) {
        // ScheduleOps still inside of kernels should be left alone for now
        return isa<KernelOp>(op->getParentOp());
    });

    target.addDynamicallyLegalOp<ScheduledLoopOp>([](ScheduledLoopOp op) {
        bool found = false;
        auto index = op.index().getValue();
        op.walk([&](Operation* innerOp) {
            for (auto operand : innerOp->getOperands())
            {
                if (operand)
                {
                    if (auto indexOp = dyn_cast_or_null<SymbolicIndexOp>(operand.getDefiningOp()); indexOp && index == indexOp.getValue())
                    {
                        found = true;
                    }
                }
            }
        });
        return !found;
    });

    target.addDynamicallyLegalOp<DimSizeOp>([](DimSizeOp op) {
        // DimSizeOps still inside of kernels should be left alone for now
        auto parentOp = op.getOperation()->getParentOp();
        return isa<KernelOp>(parentOp);
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the operations.
    RewritePatternSet patterns(&getContext());
    populateScheduledOperationsPatterns(patterns);

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
        llvm::errs() << "ScheduledOperationsLoweringPass failed\n";
        llvm::errs().flush();

        signalPassFailure();
    }
}

void ScheduleToValueLoweringPass::runOnOperation()
{
    auto function = getOperation();

    {
        RewritePatternSet foldPatterns(&getContext());
        populateScheduleToValueRewritePatterns(foldPatterns);
        (void)applyPatternsAndFoldGreedily(function, std::move(foldPatterns));
    }

    ConversionTarget target(getContext());
    target.addLegalDialect<mlir::AffineDialect,
                           mlir::arith::ArithmeticDialect,
                           mlir::math::MathDialect,
                           mlir::memref::MemRefDialect,
                           mlir::StandardOpsDialect,
                           v::ValueDialect,
                           xp::ExecutionPlanDialect>();

    // Now we only allow terminators and symbolic indices
    target.addIllegalDialect<LoopNestDialect>();
    target.addLegalOp<SymbolicIndexOp>();

    // Remove predicates if they aren't used anymore
    target.addDynamicallyLegalOp<ScheduledKernelOp,
                                 NullPredicateOp,
                                 ProloguePredicateOp,
                                 EpiloguePredicateOp,
                                 ConstantPredicateOp,
                                 FragmentTypePredicateOp,
                                 PlacementPredicateOp,
                                 IndexDefinedPredicateOp,
                                 ConjunctionPredicateOp,
                                 DisjunctionPredicateOp>([](Operation* op) {
        return !op->use_empty();
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the operations.
    RewritePatternSet patterns(&getContext());
    populateScheduleToValuePatterns(patterns);

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(function, target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

void LoopNestOptPass::runOnOperation()
{
    auto func = getOperation();

    func.walk([&](AffineForOp op) {
        if (op->getAttrOfType<UnitAttr>("accv_unrolled"))
        {
            auto tripCount = getConstantTripCount(op);
            if (tripCount && *tripCount >= 1)
                (void)loopUnrollFull(op);
        }
    });
}

namespace assera::transforms::loopnest
{
std::unique_ptr<Pass> createScheduledOperationsPass()
{
    return std::make_unique<ScheduledOperationsLoweringPass>();
}

std::unique_ptr<Pass> createScheduleToValuePass()
{
    return std::make_unique<ScheduleToValueLoweringPass>();
}

std::unique_ptr<Pass> createLoopNestOptPass()
{
    return std::make_unique<LoopNestOptPass>();
}

void addLoopNestStructureLoweringPasses(mlir::PassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createScheduledOperationsPass());
}

void addLoopNestFinalLoweringPasses(mlir::PassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createScheduleToValuePass());
    pm.addNestedPass<ir::value::ValueFuncOp>(createLoopNestOptPass());
}

void addLoopNestCleanupLoweringPasses(mlir::PassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
}

void addLoopNestLoweringPasses(mlir::PassManager& pm)
{
    addLoopNestStructureLoweringPasses(pm);
    addLoopNestFinalLoweringPasses(pm);
    addLoopNestCleanupLoweringPasses(pm);
}

} // namespace assera::transforms::loopnest
