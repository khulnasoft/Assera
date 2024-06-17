////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../value/ValueMMAOp.h"
#include <array>

namespace assera::ir
{
namespace executionPlan
{
    struct TensorizationInfo
    {
        assera::ir::value::MMAShapeType dim;
        int numTotalPasses{ 1 };
        bool useStaticOffsets{};
        int numFusedPasses{ -1 };
        assera::ir::value::MMASchedulingPolicyType schedulingPolicy{};
        assera::ir::value::MMAFragmentOpType prologueOp{};
        double prologueArg{};
        assera::ir::value::MMAFragmentOpType epilogueOp{};
        double epilogueArg{};
        bool _useRocWMMA{};

    private:
        friend inline bool operator==(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return p1.dim == p2.dim && p1.useStaticOffsets == p2.useStaticOffsets && p1.numTotalPasses == p2.numTotalPasses && p1.numFusedPasses == p2.numFusedPasses && p1.schedulingPolicy == p2.schedulingPolicy && p1.prologueOp == p2.prologueOp && p1.prologueArg == p2.prologueArg && p1.epilogueOp == p2.epilogueOp && p1.epilogueArg == p2.epilogueArg && p1._useRocWMMA == p2._useRocWMMA;
        }
        friend inline bool operator!=(const TensorizationInfo& p1, const TensorizationInfo& p2)
        {
            return !(p1 == p2);
        }
    };
} // namespace executionPlan
} // namespace assera::ir
