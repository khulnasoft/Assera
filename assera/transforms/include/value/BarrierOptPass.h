////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <string>

// fwd decls
namespace mlir
{
class Pass;
} // namespace mlir

namespace assera::transforms::value
{
std::unique_ptr<mlir::Pass> createBarrierOptPass(bool writeBarrierGraph, std::string barrierGraphFilename);
std::unique_ptr<mlir::Pass> createBarrierOptPass();
} // namespace assera::transforms::value
