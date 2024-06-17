////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Scalar.h"

#include <memory>

namespace assera::ir::loopnest
{
class KernelPredicateOpInterface;
}

namespace assera
{
namespace value
{
    class KernelPredicateImpl;

    using ScalarIndex = Scalar;

    class KernelPredicate
    {
    public:
        KernelPredicate();
        KernelPredicate(KernelPredicate&& other);
        ~KernelPredicate();

        assera::ir::loopnest::KernelPredicateOpInterface GetOp() const;

        void dump();

        KernelPredicate(const assera::ir::loopnest::KernelPredicateOpInterface& predicate);
    private:

        std::unique_ptr<KernelPredicateImpl> _impl;
    };

    // Helper functions
    KernelPredicate First(ScalarIndex index);
    KernelPredicate Last(ScalarIndex index);
    KernelPredicate EndBoundary(ScalarIndex index);
    KernelPredicate Before(ScalarIndex index);
    KernelPredicate After(ScalarIndex index);
    KernelPredicate IsDefined(ScalarIndex index);

    KernelPredicate operator&&(const KernelPredicate& lhs, const KernelPredicate& rhs);
    KernelPredicate operator||(const KernelPredicate& lhs, const KernelPredicate& rhs);

} // namespace value
} // namespace assera
#pragma region implementation
