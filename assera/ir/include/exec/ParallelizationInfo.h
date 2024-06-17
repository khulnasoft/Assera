////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

namespace assera::ir
{
namespace executionPlan
{
    struct ParallelizationInfo
    {
        int64_t numThreads = 4;
        bool isDynamicPolicy = false;
        // TODO: pinning

    private:
        friend inline bool operator==(const ParallelizationInfo& p1, const ParallelizationInfo& p2)
        {
            return (p1.numThreads == p2.numThreads) && (p1.isDynamicPolicy == p2.isDynamicPolicy);
        }
        friend inline bool operator!=(const ParallelizationInfo& p1, const ParallelizationInfo& p2)
        {
            return !(p1 == p2);
        }
    };
} // namespace executionPlan
} // namespace assera::ir
