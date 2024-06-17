////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ir/include/nest/Range.h>

namespace assera
{
namespace value
{
    namespace loopnests
    {
        /// <summary>
        /// A class representing the half-open interval `[begin, end)`, with an increment between points of _increment.
        /// </summary>
        using Range = assera::ir::loopnest::Range;
    } // namespace loopnests
} // namespace value
} // namespace assera
