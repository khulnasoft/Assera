////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace mlir
{
class DialectRegistry;
}

namespace assera::ir
{

mlir::DialectRegistry& GetDialectRegistry();

} // namespace assera::ir
