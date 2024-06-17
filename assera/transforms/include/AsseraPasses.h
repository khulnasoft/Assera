////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "affine/AffineSimplifications.h"
#include "affine/AffineLoopNormalize.h"
#include "affine/CheckBoundsPass.h"
#include "exec/ExecutionPlanToAffineLoweringPass.h"
#include "gpu/AsseraToGPUPass.h"
#include "gpu/AsseraVulkanPasses.h"
#include "ir/include/value/ValueEnums.h"
#include "nest/LoopNestPasses.h"
#include "nest/LoopNestToValueFunc.h"
#include "util/DebugFunctionPass.h"
#include "value/BarrierOptPass.h"
#include "value/FunctionPointerResolutionPass.h"
#include "value/RangeValueOptimizePass.h"
#include "value/ValueFuncToTargetPass.h"
#include "value/ValueUnrollingPass.h"
#include "value/ValueSimplifyPass.h"
#include "value/ValueToLLVMLoweringPass.h"
#include "value/ValueToStandardLoweringPass.h"
#include "vectorization/VectorizationPass.h"

#include <ir/include/intrinsics/AsseraIntrinsicsDialect.h>
#include <ir/include/exec/ExecutionPlanOps.h>
#include <ir/include/nest/LoopNestOps.h>
#include <ir/include/value/ValueDialect.h>
#include <ir/include/value/ValueEnums.h>

#include <value/include/ExecutionOptions.h>

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>

namespace assera::transforms
{

using mlir::Pass;

/// A model for providing module pass specific utilities.
///
/// Derived module passes are expected to provide the following:
///   - A 'void runOnModule()' method.
class ModulePass : public ::mlir::OperationPass<::mlir::ModuleOp>
{
public:
    using ::mlir::OperationPass<::mlir::ModuleOp>::OperationPass;

    /// The polymorphic API that runs the pass over the currently held function.
    virtual void runOnModule() = 0;

    /// The polymorphic API that runs the pass over the currently held operation.
    void runOnOperation() final
    {
        runOnModule();
    }

    /// Return the current module being transformed.
    ::mlir::ModuleOp getModule() { return this->getOperation(); }
};

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "AsseraPasses.h.inc"

struct AsseraPassPipelineOptions : mlir::PassPipelineOptions<AsseraPassPipelineOptions>
{
    Option<bool> dumpPasses{ *this, "dump-passes", llvm::cl::init(false) };
    Option<bool> gpuOnly{ *this, "gpu-only", llvm::cl::init(false) };
    Option<bool> dumpIntraPassIR{ *this, "dump-intra-pass-ir", llvm::cl::init(false) };
    Option<std::string> basename{ *this, "basename", llvm::cl::init(std::string{}) };
    Option<std::string> target{ *this, "target", llvm::cl::init("host") };
    Option<assera::value::ExecutionRuntime> runtime{
        *this,
        "runtime",
        llvm::cl::desc("Execution runtime"),
        llvm::cl::values(
            clEnumValN(assera::value::ExecutionRuntime::NONE, "none", "No runtimes"),
            clEnumValN(assera::value::ExecutionRuntime::CUDA, "cuda", "CUDA runtime"),
            clEnumValN(assera::value::ExecutionRuntime::ROCM, "rocm", "ROCm runtime"),
            clEnumValN(assera::value::ExecutionRuntime::VULKAN, "vulkan", "Vulkan runtime"),
            clEnumValN(assera::value::ExecutionRuntime::OPENMP, "openmp", "OpenMP runtime"),
            clEnumValN(assera::value::ExecutionRuntime::DEFAULT, "default", "default runtime")),
        llvm::cl::init(assera::value::ExecutionRuntime::DEFAULT)
    };
    Option<bool> enableAsync{ *this, "enable-async", llvm::cl::init(false) };
    Option<bool> enableProfile{ *this, "enable-profiling", llvm::cl::init(false) };
    Option<bool> printLoops{ *this, "print-loops", llvm::cl::init(false) };
    Option<bool> printVecOpDetails{ *this, "print-vec-details", llvm::cl::init(false) };
    Option<bool> writeBarrierGraph{ *this, "barrier-opt-dot", llvm::cl::init(false) };
    Option<std::string> barrierGraphFilename{ *this, "barrier-opt-dot-filename", llvm::cl::init(std::string{}) };
};

void addAsseraToLLVMPassPipeline(mlir::OpPassManager& pm, const AsseraPassPipelineOptions& options);

void registerAsseraToLLVMPipeline();

inline void RegisterAllPasses()
{
    mlir::registerAllPasses();
    registerPasses();
    registerAsseraToLLVMPipeline();
}

} // namespace assera::transforms
