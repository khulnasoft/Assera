////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DialectRegistry.h"
#include "exec/ExecutionPlanOps.h"
#include "nest/LoopNestOps.h"
#include "assera/AsseraOps.h"
#include "value/ValueDialect.h"
#include "intrinsics/AsseraIntrinsicsDialect.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
// #include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Dialect.h>

#include <mlir/Target/LLVMIR/Dialect/All.h>

using namespace mlir;

namespace assera::ir
{

mlir::DialectRegistry& GetDialectRegistry()
{
    static mlir::DialectRegistry registry;
    [[maybe_unused]] static bool init_once = [&]() {
        registry.insert<value::ValueDialect,
                        loopnest::LoopNestDialect,
                        executionPlan::ExecutionPlanDialect,
                        intrinsics::AsseraIntrinsicsDialect,
                        rc::AsseraDialect,

                        // MLIR dialects
                        StandardOpsDialect,
                        AffineDialect,
                        arith::ArithmeticDialect,
                        memref::MemRefDialect,
                        math::MathDialect,
                        gpu::GPUDialect,
                        // linalg::LinalgDialect,
                        LLVM::LLVMDialect,
                        NVVM::NVVMDialect,
                        ROCDL::ROCDLDialect,
                        spirv::SPIRVDialect,
                        scf::SCFDialect,
                        vector::VectorDialect>();
        mlir::registerLLVMDialectTranslation(registry);
        //mlir::registerNVVMDialectTranslation(registry);
        return true;
    }();
    return registry;
}

} // namespace assera::ir
