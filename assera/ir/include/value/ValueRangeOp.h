////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace assera::ir::value
{

using mlir::Type;
using mlir::TypeStorage;

/// A RangeType represents a minimal range abstraction (min, max, step).
/// It is constructed by calling the accv.range op with three values index of
/// index type:
///
/// ```mlir
///    func @foo(%arg0 : index, %arg1 : index, %arg2 : index) {
///      %0 = accv.range %arg0:%arg1:%arg2 : !accv.range
///    }
/// ```
class  RangeType : public Type::TypeBase<RangeType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

} // assera::ir::value