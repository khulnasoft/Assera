////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

#include <transforms/include/util/SnapshotUtilities.h>

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>


// fwd decls
namespace llvm
{
class DataLayout;    
}

namespace mlir
{
class MLIRContext;
class ModuleOp;
class Pass;
class LLVMTypeConverter;

template <typename OpT>
class OperationPass;

class RewritePatternSet;
} // namespace mlir

namespace assera::transforms::value
{
void populateValueToLLVMNonMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns, assera::value::TargetDevice deviceInfo);
void populateGlobalValueToLLVMNonMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateLocalValueToLLVMNonMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns, assera::value::TargetDevice deviceInfo);
void populateValueToLLVMMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
void populateReshapeOpToLLVMMemPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns);

const mlir::LowerToLLVMOptions& GetDefaultAsseraLLVMOptions(mlir::MLIRContext* context);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass(mlir::LowerToLLVMOptions options);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass(mlir::MLIRContext* context);
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createValueToLLVMPass(bool useBasePtrCallConv,
                                                                           bool emitCWrappers,
                                                                           unsigned indexBitwidth,
                                                                           bool useAlignedAlloc,
                                                                           llvm::DataLayout dataLayout,
                                                                           assera::value::TargetDevice deviceInfo = {},
                                                                           const IntraPassSnapshotOptions& options = {});
} // namespace assera::transforms::value
