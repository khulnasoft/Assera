////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors:  Abdul Dakkak, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ir/include/value/ValueEnums.h"
#include "value/include/FunctionDeclaration.h"
#include <memory>
#include <mlir/Dialect/GPU/GPUDialect.h>

namespace mlir
{
class MLIRContext;
class ModuleOp;
class RewritePatternSet;
class Pass;
class PassManager;
class SPIRVTypeConverter;
class LLVMTypeConverter;
class TypeConverter;

template <typename OpT>
class OperationPass;

namespace gpu
{
    class GPUModuleOp;
};

} // namespace mlir

namespace assera::transforms
{
void populateGPUSimplificationPatterns(mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGPUSimplificationPass();

void populateAsseraToSPIRVPatterns(
    mlir::SPIRVTypeConverter& typeConverter,
    mlir::MLIRContext* context,
    mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAsseraToSPIRVPass();

void populateAsseraToNVVMPatterns(mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAsseraToNVVMPass();

void populateAsseraToROCDLPatterns(mlir::RewritePatternSet& patterns);
void populateGPUToROCDLPatterns(mlir::LLVMTypeConverter& converter, mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAsseraToROCDLPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGPUToROCDLPass();

std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUModuleOp>> createSerializeToHSACOPass();

// Abstract method which dispatches to SPIRV, NVVM, or ROCDL depending on the execution environment's runtime
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAsseraToGPUPass(assera::value::ExecutionRuntime runtime);

} // namespace assera::transforms
