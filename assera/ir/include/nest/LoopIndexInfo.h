////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors:  Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Index.h"
#include "Range.h"

#include <mlir/IR/Value.h>

#include <unordered_map>

namespace assera::ir
{
namespace loopnest
{
    enum class LoopIndexState
    {
        notVisited,
        inProgress,
        done
    };

    struct LoopIndexSymbolTableEntry
    {
        mlir::Value value;
        Range loopRange;
        LoopIndexState state;
    };
    using LoopIndexSymbolTable = std::unordered_map<Index, LoopIndexSymbolTableEntry>;
} // namespace loopnest
} // namespace assera::ir
