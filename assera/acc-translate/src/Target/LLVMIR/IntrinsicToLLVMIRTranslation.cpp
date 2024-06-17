////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////


#include "IntrinsicToLLVMIRTranslation.h"

#include <ir/include/intrinsics/AsseraIntrinsicsDialect.h>

#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsX86.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace assera::transforms::intrinsics;

namespace {
class IntrinsicsDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "intrinsics/AsseraIntrinsicsConversions.inc"

    return failure();
  }
};
} // namespace

void assera::transforms::intrinsics::registerIntrinsicsDialectTranslation(DialectRegistry &registry) {
  registry.insert<assera::ir::intrinsics::AsseraIntrinsicsDialect>();
  registry.addDialectInterface<assera::ir::intrinsics::AsseraIntrinsicsDialect,
                               IntrinsicsDialectLLVMIRTranslationInterface>();
}

void assera::transforms::intrinsics::registerIntrinsicsDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerIntrinsicsDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
