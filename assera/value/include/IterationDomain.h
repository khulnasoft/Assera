////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ir/include/nest/IterationDomain.h>

namespace assera
{
namespace value
{
    namespace loopnests
    {
        /// <summary>
        /// The set of all points (IterationVectors) to be visited by a loop or loop nest.
        /// </summary>
        using IterationDomain = assera::ir::loopnest::IterationDomain;
    } // namespace loopnests
} // namespace value
} // namespace assera
