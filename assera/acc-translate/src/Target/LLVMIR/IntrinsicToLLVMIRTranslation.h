////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace mlir
{

class DialectRegistry;
class MLIRContext;

} // namespace mlir

namespace assera::transforms::intrinsics
{

/// Register the Intrinsic dialect and the translation from it to the LLVM IR
/// in the given registry;
void registerIntrinsicsDialectTranslation(mlir::DialectRegistry& registry);

/// Register the Intrinsic dialect and the translation from it in the registry
/// associated with the given context.
void registerIntrinsicsDialectTranslation(mlir::MLIRContext& context);

} // namespace assera::transforms::intrinsics
