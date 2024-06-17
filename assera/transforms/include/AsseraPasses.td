////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ASSERA_CONVERSION_PASSES
#define ASSERA_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

class accModulePass<string passArg> : PassBase<passArg, "::assera::transforms::ModulePass">;

//===----------------------------------------------------------------------===//
// ExecutionPlanCacheRegion
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanCacheRegion : Pass<"convert-exec-plan-cache-region"> {
  let summary = "Lower ExecPlan cache region and mapping ops";
  let constructor = "assera::transforms::executionPlan::createExecutionPlanCacheRegionLoweringPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanParallelization
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanParallelization : Pass<"convert-exec-plan-parallelize"> {
  let summary = "Match and parallelize ExecPlan ops";
  let constructor = "assera::transforms::executionPlan::createExecutionPlanParallelizationPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanTensorization
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanTensorization : Pass<"convert-exec-plan-tensorize"> {
  let summary = "Match and tensorize ExecPlan ops";
  let constructor = "assera::transforms::executionPlan::createExecutionPlanTensorizationPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanMakeCache
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanMakeCache : Pass<"convert-exec-make-cache", "mlir::FuncOp"> {
  let summary = "Lower ExecPlan make cache op";
  let constructor = "assera::transforms::executionPlan::createExecutionPlanMakeCachePass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanCopyReduce
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanCopyReduce : Pass<"convert-exec-copy-reduce"> {
  let summary = "Lower ExecPlan cache copy and reduce ops";
  let constructor = "assera::transforms::executionPlan::createExecutionPlanCopyReducePass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "assera::ir::loopnest::LoopNestDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExecutionPlanScaleHoisting
//===----------------------------------------------------------------------===//

def ConvertExecutionPlanScaleHoisting : Pass<"hoist-cache-scaling", "mlir::FuncOp"> {
  let summary = "Hoist scale ops in kernels to cache reduce ops";
  let constructor = "assera::transforms::executionPlan::createExecutionPlanScaleHoistingPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "assera::ir::loopnest::LoopNestDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// HandleOutOfBoundsAccess
//===----------------------------------------------------------------------===//

def HandleOutOfBoundsAccess : Pass<"handle-out-of-bounds-access", "mlir::FuncOp"> {
  let summary = "Detect potential out-of-bounds affine loads and replace them with a conditional access and default value";
  let constructor = "assera::transforms::executionPlan::createOutOfBoundsAccessHandlingPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "assera::ir::loopnest::LoopNestDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ScheduledOperations
//===----------------------------------------------------------------------===//

def ConvertScheduledOperations : Pass<"convert-loopnest-scheduledops", "mlir::FuncOp"> {
  let summary = "Lower LoopNest ops";
  let constructor = "assera::transforms::loopnest::createScheduledOperationsPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ScheduleToValue
//===----------------------------------------------------------------------===//

def ConvertScheduleToValue : Pass<"convert-loopnest-to-value", "mlir::FuncOp"> {
  let summary = "Lower LoopNest ops to Value dialect ops";
  let constructor = "assera::transforms::loopnest::createScheduleToValuePass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LoopNestOpt
//===----------------------------------------------------------------------===//

def ConvertLoopNestOpt : Pass<"loopnest-opt", "assera::ir::value::ValueFuncOp"> {
  let summary = "Optimize LoopNest ops";
  let constructor = "assera::transforms::loopnest::createLoopNestOptPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ValueSimplify
//===----------------------------------------------------------------------===//

def ConvertValueSimplify : Pass<"simplify-value"> {
  let summary = "Simplify Value ops";
  let constructor = "assera::transforms::value::createValueSimplifyPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    // "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect"
  ];
}

//===----------------------------------------------------------------------===//
// RangeValueOptimization
//===----------------------------------------------------------------------===//

def ConvertRangeValueOptimize : Pass<"optimize-range-value"> {
  let summary = "Performing optimization using Range Value analysis";
  let constructor = "assera::transforms::value::createRangeValueOptimizePass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    // "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// AffineSimplification
//===----------------------------------------------------------------------===//

// use Assera / acc- prefix to distinguish from the built-in MLIR SimplifyAffineStructures pass
def AsseraAffineSimplification : Pass<"acc-affine-simplify"> {
  let summary = "Simplify affine expressions using Range Value analysis";
  let constructor = "assera::transforms::affine::createAffineSimplificationPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::gpu::GPUDialect"
  ];
}


//===----------------------------------------------------------------------===//
// AsseraAffineLoopNormalize
//===----------------------------------------------------------------------===//

def AsseraAffineLoopNormalize : Pass<"acc-affine-loop-normalize"> {
  let summary = "Normalize affine for ops";
  let constructor = "assera::transforms::affine::createAsseraAffineLoopNormalizePass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::gpu::GPUDialect"
  ];
}
//===----------------------------------------------------------------------===//
// BarrierOpt
//===----------------------------------------------------------------------===//

def BarrierOpt : Pass<"optimize-barriers"> {
  let summary = "Optimize Barrier ops";
  let constructor = "assera::transforms::value::createBarrierOptPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"writeBarrierGraph", "barrier-opt-dot", "bool", /*default=*/"false",
           "Write out barrier analysis graph in graphviz format">,
    Option<"barrierGraphFilename", "barrier-opt-dot-filename", "std::string",
           /*default=*/"\"\"",
           "Filename for barrier analysis graph in graphviz format">
  ];
}

//===----------------------------------------------------------------------===//
// ValueToLLVM
//===----------------------------------------------------------------------===//

def ConvertValueToLLVM : accModulePass<"value-to-llvm"> {
  let summary = "Lower Value ops to LLVM";
  let constructor = "assera::transforms::value::createValueToLLVMPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "assera::ir::intrinsics::AsseraIntrinsicsDialect",
    "mlir::LLVM::LLVMDialect"
  ];
  // Match std-to-llvm options so we can pass through arguments
  let options = [
    // TODO : evaluate if we want to converge with MLIR and default this to false once the sample emitters catch up
    Option<"useAlignedAlloc", "use-aligned-alloc", "bool", /*default=*/"true",// "false",
           "Use aligned_alloc in place of malloc for heap allocations">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">
  ];
}

//===----------------------------------------------------------------------===//
// ValueToStd
//===----------------------------------------------------------------------===//

def ConvertValueToStd : accModulePass<"convert-value-to-std"> {
  let summary = "Lower Value ops to standard MLIR dialects";
  let constructor = "assera::transforms::value::createValueToStdPass()";
  let options = [
    Option<"enableProfiling", "enable-profiling", "bool", /*default=*/"false",
           "Enable profiling">
  ];
  let dependentDialects = [
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect"
    // "mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ValueFuncToTarget
//===----------------------------------------------------------------------===//

def ValueFuncToTarget : Pass<"value-func-to-target", "::mlir::ModuleOp"> {
  let constructor = "assera::transforms::value::createValueFuncToTargetPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ValueUnrolling
//===----------------------------------------------------------------------===//

def ValueUnrolling : Pass<"value-unroll", "::mlir::ModuleOp"> {
  let constructor = "assera::transforms::value::createValueUnrollingPass()";
  let dependentDialects = [
    "mlir::StandardOpsDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LoopNestToValueFunc
//===----------------------------------------------------------------------===//

def LoopNestToValueFunc : Pass<"loopnest-to-value-func", "assera::ir::value::ValueFuncOp"> {
  let summary = "Lower loop nests";
  let description = [{
      This pass lowers loop nests into value functions.
    }];
  let constructor = "assera::transforms::loopnest::createLoopNestToValueFuncPass()";
  let options = [
    Option<"printVecOpDetails", "print-vec-details", "bool", /*default=*/"false",
           "Print details about op vectorization">,
    Option<"printLoops", "print-loops", "bool", /*default=*/"false",
           "Print loop structure">
  ];
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::gpu::GPUDialect"
    // "mlir::linalg::LinalgDialect"
  ];
}


//===----------------------------------------------------------------------===//
// AsseraVectorizationPass
//===----------------------------------------------------------------------===//

def AsseraVectorizationPass : Pass<"acc-vectorize", "::mlir::ModuleOp"> {
  let summary = "Vectorize AffineForOps marked for vectorization";
  let description = [{
      This pass finds AffineForOps marked for vectorization and vectorizes the ops in those loops
    }];
  let constructor = "assera::transforms::vectorization::createVectorizationPass()";
  let options = [
    Option<"printVecOpDetails", "print-vec-details", "bool", /*default=*/"false",
           "Print details about op vectorization">
  ];
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect"
  ];
}


//===----------------------------------------------------------------------===//
// AsseraVectorizationUnrollPass
//===----------------------------------------------------------------------===//

def AsseraVectorizationUnrollPass : Pass<"acc-vectorize-unroll", "::mlir::ModuleOp"> {
  let summary = "Unroll AffineForOps marked for vectorized unroll";
  let description = [{
      This pass finds AffineForOps marked for vectorized unroll and unrolls those loops op-by-op
    }];
  let constructor = "assera::transforms::vectorization::createVectorizationUnrollPass()";
  let options = [
    Option<"printVecOpDetails", "print-vec-details", "bool", /*default=*/"false",
           "Print details about op vectorization">
  ];
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect"
  ];
}


//===----------------------------------------------------------------------===//
// ConvertVulkanLaunchFuncToVulkanCallsWithTiming
//===----------------------------------------------------------------------===//

def ConvertVulkanLaunchFuncToVulkanCallsWithTiming : accModulePass<"launch-func-to-vulkan-with-timing"> {
  let summary = "Convert vulkanLaunch external call to Vulkan runtime external "
                "calls with custom timing configuration runtime calls";
  let constructor = "assera::transforms::vulkan::createConvertVulkanLaunchFuncToVulkanCallsWithTimingPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let options = [
    Option<"printTimings", "print-timings", "bool", /*default=*/"true",
           "Print device timestamp averages">,
    Option<"warmupCount", "warmup-count", "unsigned",
           /*default=*/"0",
           "Number of warmup rounds to run when timing">,
    Option<"runCount", "run-count", "unsigned",
           /*default=*/"1",
           "Number of timed rounds to run when timing">
  ];
}

//===----------------------------------------------------------------------===//
// EmitVulkanWrappers
//===----------------------------------------------------------------------===//

def EmitVulkanWrapper : accModulePass<"emit-vulkan-wrappers"> {
  let summary = "Mark vulkanLaunch calls to emit C wrappers";
  let constructor = "assera::transforms::vulkan::createEmitVulkanWrapperPass()";
  let dependentDialects = ["mlir::StandardOpsDialect"];
}

//===----------------------------------------------------------------------===//
// FunctionPointerResolution
//===----------------------------------------------------------------------===//

def FunctionPointerResolution : accModulePass<"resolve-function-pointers"> {
  let summary = "Resolve Assera temporary functions to their fully-realized counterparts";
  let constructor = "assera::transforms::value::createFunctionPointerResolutionPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// SerializeToHSACO
//===----------------------------------------------------------------------===//

def SerializeToHSACO : Pass<"serialize-to-hsaco", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Serializes the GPU kernel to HSACO object (WIP)";
  let constructor = "assera::transforms::createSerializeToHSACOPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::ROCDL::ROCDLDialect"
  ];
  let options = [
    Option<"chip", "chip", "std::string",
          //  TODO: Should this default to something else?
           "\"gfx908\"",
           "The GPU target architecture.">
  ];
}

//===----------------------------------------------------------------------===//
// GPUSimplification
//===----------------------------------------------------------------------===//

def GPUSimplification : Pass<"gpu-simplification", "::mlir::ModuleOp"> {
  let summary = "Simplifies GPU operations";
  let constructor = "assera::transforms::createGPUSimplificationPass()";
  let dependentDialects = ["assera::ir::value::ValueDialect"];
}

//===----------------------------------------------------------------------===//
// AsseraToSPIRV
//===----------------------------------------------------------------------===//

def ConvertAsseraToSPIRV : Pass<"convert-assera-to-spirv", "::mlir::ModuleOp"> {
  let summary = "Convert Assera dialects to SPIR-V dialect";
  let constructor = "assera::transforms::createAsseraToSPIRVPass()";
  let dependentDialects = ["mlir::spirv::SPIRVDialect", "mlir::gpu::GPUDialect"];
}

//===----------------------------------------------------------------------===//
// AsseraToROCDL
//===----------------------------------------------------------------------===//

def ConvertAsseraToROCDL : Pass<"convert-assera-to-rocdl", "::mlir::ModuleOp"> {
  let summary = "Convert Assera dialects to ROCDL dialect";
  let constructor = "assera::transforms::createAsseraToROCDLPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
    "mlir::ROCDL::ROCDLDialect"
  ];
}


//===----------------------------------------------------------------------===//
// GPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertGPUToROCDL : Pass<"convert-gpu-ops-to-rocdl", "::mlir::ModuleOp"> {
  let summary = "Convert GPU dialects to ROCDL dialect";
  let constructor = "assera::transforms::createGPUToROCDLPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
    "mlir::ROCDL::ROCDLDialect"
  ];
}


//===----------------------------------------------------------------------===//
// AsseraToNVVM
//===----------------------------------------------------------------------===//

def ConvertAsseraToNVVM : Pass<"convert-assera-to-nvvm", "::mlir::ModuleOp"> {
  let summary = "Convert Assera dialects to NVVM dialect";
  let constructor = "assera::transforms::createAsseraToNVVMPass()";
  let dependentDialects = [
    "assera::ir::value::ValueDialect",
    "mlir::StandardOpsDialect",
    "mlir::AffineDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
    "mlir::NVVM::NVVMDialect"
  ];
}


//===----------------------------------------------------------------------===//
// EmitDebugFunction
//===----------------------------------------------------------------------===//

def EmitDebugFunction : accModulePass<"emit-debug-func"> {
  let summary = "Emits a debug function with correctness-checking against default schedules";
  let constructor = "assera::transforms::createEmitDebugFunctionPass()";
  let dependentDialects = ["assera::ir::value::ValueDialect"];
}


#endif // ASSERA_CONVERSION_PASSES
